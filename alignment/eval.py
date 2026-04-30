from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from datasets import load_dataset

try:
    from .prompts import COT_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE
    from .rewards import answer_tag_reward_fn, majority_vote_tagged_answers
except ImportError:  # pragma: no cover - allows `python alignment/eval.py`
    from alignment.prompts import COT_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE
    from alignment.rewards import answer_tag_reward_fn, majority_vote_tagged_answers


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_VALIDATION_SIZE = 256
DEFAULT_GSM8K_CONFIG = "main"
DEFAULT_OUTPUT_DIR = Path("alignment/results")
DEFAULT_MAX_TOKENS = 256


def _prompt_style_name(prompt_template: str) -> str:
    return "cot" if prompt_template == COT_PROMPT_TEMPLATE else "direct"


def _extract_gsm8k_final_answer(answer_text: str) -> str:
    final_answer = answer_text.rsplit("####", maxsplit=1)[-1].strip()
    return re.sub(r"(?<=\d),(?=\d)", "", final_answer)


def _average_metric_dicts(metric_dicts: Sequence[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}

    metric_totals: dict[str, float] = defaultdict(float)
    for metric_dict in metric_dicts:
        for key, value in metric_dict.items():
            metric_totals[key] += float(value)

    count = float(len(metric_dicts))
    return {key: value / count for key, value in metric_totals.items()}


def _build_reward_fn(examples: Sequence[dict[str, Any]]) -> Callable[[str, str], dict[str, float]]:
    prompt_to_ground_truth = {example["prompt"]: example["ground_truth"] for example in examples}

    def reward_fn(response: str, prompt: str) -> dict[str, float]:
        return answer_tag_reward_fn(response, prompt_to_ground_truth[prompt])

    return reward_fn


def _attach_example_metadata(
    examples: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched_records: list[dict[str, Any]] = []
    for example, record in zip(examples, records, strict=True):
        enriched_records.append(
            {
                "index": example["index"],
                "question": example["question"],
                "ground_truth": example["ground_truth"],
                "gold_solution": example["answer"],
                "prompt": record["prompt"],
                "generation": record["generation"],
                "candidate_generations": record["candidate_generations"],
                "scores": record["scores"],
            }
        )
    return enriched_records


def _default_output_path(
    *,
    prompt_template: str,
    split: str,
    limit: int,
    self_consistency: bool,
) -> Path:
    prompt_style = _prompt_style_name(prompt_template)
    suffix = "self_consistency" if self_consistency else "baseline"
    return DEFAULT_OUTPUT_DIR / f"gsm8k_{split}_{prompt_style}_{suffix}_limit{limit}_vllm.json"


def load_gsm8k_examples(split: str) -> list[dict[str, Any]]:
    """Load GSM8K examples from HuggingFace datasets."""
    dataset = load_dataset("openai/gsm8k", DEFAULT_GSM8K_CONFIG, split=split)

    examples: list[dict[str, Any]] = []
    for index, record in enumerate(dataset):
        examples.append(
            {
                "index": index,
                "question": record["question"],
                "answer": record["answer"],
                "ground_truth": _extract_gsm8k_final_answer(record["answer"]),
            }
        )
    return examples


def build_prompts(examples: Sequence[dict[str, Any]], prompt_template: str) -> list[str]:
    """Format raw GSM8K examples into prompt strings."""
    return [prompt_template.format(question=example["question"]) for example in examples]


def evaluate_vllm(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    eval_sampling_params,
) -> dict[str, Any]:
    """Generate model outputs, score them, and return serializable evaluation artifacts."""
    request_outputs = vllm_model.generate(list(prompts), sampling_params=eval_sampling_params)

    records: list[dict[str, Any]] = []
    per_example_scores: list[dict[str, float]] = []
    for prompt, request_output in zip(prompts, request_outputs, strict=True):
        candidate_generations = [completion.text for completion in request_output.outputs]
        generation = candidate_generations[0] if candidate_generations else ""
        scores = reward_fn(generation, prompt)
        records.append(
            {
                "prompt": prompt,
                "generation": generation,
                "candidate_generations": candidate_generations,
                "scores": scores,
            }
        )
        per_example_scores.append(scores)

    return {
        "metrics": {
            "num_examples": len(records),
            **_average_metric_dicts(per_example_scores),
        },
        "records": records,
    }


def write_evaluation_results(results: dict[str, Any], output_path: Path) -> None:
    """Serialize generations and scores for later analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def _run_vllm_baseline(
    *,
    output_path: Path,
    prompt_template: str,
    model_name: str,
    split: str,
    limit: int,
    offset: int,
    max_tokens: int,
) -> None:
    from vllm import LLM, SamplingParams

    examples = load_gsm8k_examples(split)[offset : offset + limit]
    prompts = build_prompts(examples, prompt_template)
    for example, prompt in zip(examples, prompts, strict=True):
        example["prompt"] = prompt

    reward_fn = _build_reward_fn(examples)
    model = LLM(model=model_name)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, n=1)
    raw_results = evaluate_vllm(model, reward_fn, prompts, sampling_params)

    write_evaluation_results(
        {
            "metadata": {
                "baseline": _prompt_style_name(prompt_template),
                "prompt_style": _prompt_style_name(prompt_template),
                "split": split,
                "offset": offset,
                "model_name": model_name,
                "backend": "vllm",
                "num_examples": len(examples),
            },
            "metrics": raw_results["metrics"],
            "records": _attach_example_metadata(examples, raw_results["records"]),
        },
        output_path,
    )


def _run_vllm_self_consistency(
    *,
    output_path: Path,
    prompt_template: str,
    model_name: str,
    split: str,
    limit: int,
    offset: int,
    max_tokens: int,
    k: int,
) -> None:
    from vllm import LLM, SamplingParams

    examples = load_gsm8k_examples(split)[offset : offset + limit]
    prompts = build_prompts(examples, prompt_template)
    for example, prompt in zip(examples, prompts, strict=True):
        example["prompt"] = prompt

    reward_fn = _build_reward_fn(examples)
    model = LLM(model=model_name)
    sampling_params = SamplingParams(temperature=0.7, top_p=1.0, max_tokens=max_tokens, n=k)
    raw_results = evaluate_vllm(model, reward_fn, prompts, sampling_params)

    voted_records: list[dict[str, Any]] = []
    voted_scores: list[dict[str, float]] = []
    for example, record in zip(examples, raw_results["records"], strict=True):
        voted_answer = majority_vote_tagged_answers(record["candidate_generations"])
        voted_generation = f"<answer>{voted_answer}</answer>" if voted_answer is not None else ""
        scores = answer_tag_reward_fn(voted_generation, example["ground_truth"])
        voted_scores.append(scores)
        voted_records.append(
            {
                "index": example["index"],
                "question": example["question"],
                "ground_truth": example["ground_truth"],
                "gold_solution": example["answer"],
                "prompt": record["prompt"],
                "generation": voted_generation,
                "candidate_generations": record["candidate_generations"],
                "scores": scores,
            }
        )

    write_evaluation_results(
        {
            "metadata": {
                "baseline": f"{_prompt_style_name(prompt_template)}_self_consistency",
                "prompt_style": _prompt_style_name(prompt_template),
                "split": split,
                "offset": offset,
                "model_name": model_name,
                "backend": "vllm",
                "num_examples": len(examples),
                "num_samples_per_prompt": k,
            },
            "metrics": {
                "num_examples": len(voted_records),
                **_average_metric_dicts(voted_scores),
            },
            "records": voted_records,
        },
        output_path,
    )


def run_direct_baseline(output_path: Path) -> None:
    """Evaluate the direct-prediction GSM8K baseline from Section 3.1."""
    _run_vllm_baseline(
        output_path=output_path,
        prompt_template=DIRECT_PROMPT_TEMPLATE,
        model_name=DEFAULT_MODEL_NAME,
        split="train",
        limit=DEFAULT_VALIDATION_SIZE,
        offset=0,
        max_tokens=DEFAULT_MAX_TOKENS,
    )


def run_cot_baseline(output_path: Path) -> None:
    """Evaluate the chain-of-thought baseline from Section 3.2."""
    _run_vllm_baseline(
        output_path=output_path,
        prompt_template=COT_PROMPT_TEMPLATE,
        model_name=DEFAULT_MODEL_NAME,
        split="train",
        limit=DEFAULT_VALIDATION_SIZE,
        offset=0,
        max_tokens=DEFAULT_MAX_TOKENS,
    )


def run_self_consistency_baseline(output_path: Path, k: int = 5) -> None:
    """Evaluate the self-consistency baseline from Section 3.2."""
    _run_vllm_self_consistency(
        output_path=output_path,
        prompt_template=COT_PROMPT_TEMPLATE,
        model_name=DEFAULT_MODEL_NAME,
        split="train",
        limit=DEFAULT_VALIDATION_SIZE,
        offset=0,
        max_tokens=DEFAULT_MAX_TOKENS,
        k=k,
    )


def get_prompt_template(use_cot: bool) -> str:
    return COT_PROMPT_TEMPLATE if use_cot else DIRECT_PROMPT_TEMPLATE


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a vLLM GSM8K baseline.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--use-cot", action="store_true")
    parser.add_argument("--self-consistency", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prompt_template = get_prompt_template(args.use_cot or args.self_consistency)
    output_path = args.output_path or _default_output_path(
        prompt_template=prompt_template,
        split=args.split,
        limit=args.limit,
        self_consistency=args.self_consistency,
    )

    if args.self_consistency:
        _run_vllm_self_consistency(
            output_path=output_path,
            prompt_template=prompt_template,
            model_name=args.model_name,
            split=args.split,
            limit=args.limit,
            offset=args.offset,
            max_tokens=args.max_tokens,
            k=args.k,
        )
        return

    _run_vllm_baseline(
        output_path=output_path,
        prompt_template=prompt_template,
        model_name=args.model_name,
        split=args.split,
        limit=args.limit,
        offset=args.offset,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
