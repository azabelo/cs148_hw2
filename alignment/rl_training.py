"""GRPO RL training script (HF policy; rollouts match CoT + </answer> stop behavior).

For large-scale runs, swap HF ``generate`` for vLLM with
``SamplingParams(..., stop=["</answer>"])`` and the same temperature / min / max tokens.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

try:
    from .eval import _average_metric_dicts, build_prompts, load_gsm8k_examples
    from .grpo import (
        compute_entropy,
        compute_group_normalized_rewards,
        grpo_microbatch_train_step,
        get_response_log_probs,
        tokenize_prompt_and_output,
    )
    from .prompts import COT_PROMPT_TEMPLATE
    from .rewards import answer_tag_reward_fn
except ImportError:  # pragma: no cover
    from alignment.eval import _average_metric_dicts, build_prompts, load_gsm8k_examples
    from alignment.grpo import (
        compute_entropy,
        compute_group_normalized_rewards,
        grpo_microbatch_train_step,
        get_response_log_probs,
        tokenize_prompt_and_output,
    )
    from alignment.prompts import COT_PROMPT_TEMPLATE
    from alignment.rewards import answer_tag_reward_fn

logger = logging.getLogger(__name__)


def _apply_hf_ssl_cert_workaround() -> None:
    """Match scripts/run_alignment_eval_vllm.sh: bad SSL_CERT_DIR breaks hub TLS; prefer certifi CA bundle."""
    ssl_dir = os.environ.get("SSL_CERT_DIR")
    if ssl_dir and not os.path.isdir(ssl_dir):
        os.environ.pop("SSL_CERT_DIR", None)
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        fallback = "/etc/ssl/certs/ca-certificates.crt"
        if os.path.isfile(fallback):
            os.environ["SSL_CERT_FILE"] = fallback


@dataclass
class GRPOHyperparams:
    n_grpo_steps: int = 8
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 32
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 256
    epochs_per_rollout_batch: int = 1
    train_batch_size: int = 32
    gradient_accumulation_steps: int = 16
    cliprange: float = 1.0
    normalize_by_std: bool = True
    val_interval: int = 8
    val_size: int = 256
    # Validation prompts per ``generate`` call (left-padded batch). Lower if GPU OOM.
    val_batch_size: int = 256
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        assert self.train_batch_size % self.gradient_accumulation_steps == 0, (
            "train_batch_size must be divisible by gradient_accumulation_steps"
        )
        self.micro_train_batch_size = self.train_batch_size // self.gradient_accumulation_steps
        assert self.rollout_batch_size % self.group_size == 0, (
            "rollout_batch_size must be divisible by group_size"
        )
        self.n_prompts_per_rollout_batch = self.rollout_batch_size // self.group_size
        assert self.train_batch_size >= self.group_size, (
            "train_batch_size must be greater than or equal to group_size"
        )
        assert self.rollout_batch_size % self.micro_train_batch_size == 0, (
            "rollout_batch_size must be divisible by micro_train_batch_size"
        )
        self.n_microbatches_per_rollout_batch = self.rollout_batch_size // self.micro_train_batch_size
        assert self.n_microbatches_per_rollout_batch == self.gradient_accumulation_steps, (
            "Expected rollout_batch_size / micro_train_batch_size == gradient_accumulation_steps "
            "for the default on-policy schedule."
        )


class StopOnAnswerClose(StoppingCriteria):
    """Stop when ``</answer>`` appears in newly generated tokens only.

    The CoT system prompt already contains ``</answer>`` in its instructions, so checking
    the full decoded sequence would stop immediately (often one token after ``min_new_tokens``).
    """

    def __init__(self, tokenizer: Any, initial_prompt_len: int, stop: str = "</answer>") -> None:
        self.tokenizer = tokenizer
        self.initial_prompt_len = initial_prompt_len
        self.stop = stop

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        done: list[bool] = []
        for row in input_ids:
            gen_ids = row[self.initial_prompt_len :]
            if gen_ids.numel() == 0:
                done.append(False)
                continue
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
            done.append(self.stop in text)
        return torch.tensor(done, device=input_ids.device, dtype=torch.bool)


def _policy_forward_log_probs(
    model: torch.nn.Module, input_ids: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return log_probs, logits


def _rollout_batch_multi_prompt(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    group_size: int,
    hp: GRPOHyperparams,
    device: torch.device,
) -> list[str]:
    """Sample one rollout group per prompt, all in a single batched ``generate`` call."""
    enc = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device).repeat_interleave(group_size, dim=0)
    attn = enc["attention_mask"].to(device).repeat_interleave(group_size, dim=0)
    prompt_len0 = input_ids.shape[1]
    stop_criteria = StoppingCriteriaList([StopOnAnswerClose(tokenizer, prompt_len0)])
    with torch.inference_mode():
        out = model.generate(
            input_ids,
            attention_mask=attn,
            max_new_tokens=hp.sampling_max_tokens,
            min_new_tokens=hp.sampling_min_tokens,
            do_sample=True,
            temperature=hp.sampling_temperature,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stop_criteria,
        )
    completions: list[str] = []
    for i in range(out.shape[0]):
        completions.append(tokenizer.decode(out[i, prompt_len0:], skip_special_tokens=True))
    return completions


def _run_validation(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    device: torch.device,
    hp: GRPOHyperparams,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    prompts = build_prompts(examples, str(COT_PROMPT_TEMPLATE))
    scores: list[dict[str, float]] = []
    rollouts: list[dict[str, Any]] = []
    bs = max(1, hp.val_batch_size)
    n = len(examples)
    for start in tqdm(range(0, n, bs), desc="validation", leave=False, total=(n + bs - 1) // bs):
        chunk_ex = examples[start : start + bs]
        chunk_pr = prompts[start : start + bs]
        enc = tokenizer(
            chunk_pr,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        prompt_len0 = input_ids.shape[1]
        stop_criteria = StoppingCriteriaList([StopOnAnswerClose(tokenizer, prompt_len0)])
        with torch.inference_mode():
            out = model.generate(
                input_ids,
                attention_mask=attn,
                max_new_tokens=hp.sampling_max_tokens,
                min_new_tokens=hp.sampling_min_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=stop_criteria,
            )
        for i, (ex, prompt) in enumerate(zip(chunk_ex, chunk_pr, strict=True)):
            gen = tokenizer.decode(out[i, prompt_len0:], skip_special_tokens=True)
            metrics = answer_tag_reward_fn(gen, ex["ground_truth"])
            scores.append(metrics)
            rollouts.append(
                {
                    "question": ex["question"],
                    "ground_truth": ex["ground_truth"],
                    "prompt": prompt,
                    "completion": gen,
                    "reward": metrics.get("reward", 0.0),
                    "format_reward": metrics.get("format_reward", 0.0),
                    "answer_reward": metrics.get("answer_reward", 0.0),
                }
            )
    model.train()
    return {"val_" + k: v for k, v in _average_metric_dicts(scores).items()}, rollouts


def _save_model_and_tokenizer(
    model: torch.nn.Module,
    tokenizer: Any,
    save_dir: Path,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    logger.info("Saved model and tokenizer to %s", save_dir.resolve())


def train_grpo_loop(
    model_name: str,
    output_dir: Path,
    device: torch.device | None = None,
    seed: int = 42,
    train_limit: int = 512,
    use_wandb: bool = False,
    save_model_dir: Path | None = None,
    n_grpo_steps: int | None = None,
    normalize_by_std: bool = True,
    val_batch_size: int | None = None,
) -> None:
    hp = GRPOHyperparams()
    if n_grpo_steps is not None:
        hp.n_grpo_steps = n_grpo_steps
    hp.normalize_by_std = normalize_by_std
    if val_batch_size is not None:
        hp.val_batch_size = val_batch_size
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hp.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    train_examples = load_gsm8k_examples("train")[:train_limit]
    val_examples = load_gsm8k_examples("test")[: hp.val_size]

    reward_fn = answer_tag_reward_fn
    global_step = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_wandb:
        import wandb

        wandb.init(project="cs148-grpo", config=vars(hp))

    shuffled = list(range(len(train_examples)))
    random.shuffle(shuffled)
    ptr = 0

    for grpo_step in tqdm(range(hp.n_grpo_steps), desc="grpo_steps"):
        batch_prompts: list[str] = []
        batch_completions: list[str] = []
        batch_gts: list[str] = []

        model.eval()
        step_prompts: list[str] = []
        step_gts: list[str] = []
        for _ in range(hp.n_prompts_per_rollout_batch):
            if ptr >= len(shuffled):
                random.shuffle(shuffled)
                ptr = 0
            ex = train_examples[shuffled[ptr]]
            ptr += 1
            step_prompts.append(str(COT_PROMPT_TEMPLATE).format(question=ex["question"]))
            step_gts.append(ex["ground_truth"])
        batch_completions = _rollout_batch_multi_prompt(
            model, tokenizer, step_prompts, hp.group_size, hp, device
        )
        for p, g in zip(step_prompts, step_gts, strict=True):
            batch_prompts.extend([p] * hp.group_size)
            batch_gts.extend([g] * hp.group_size)

        batch_tok = tokenize_prompt_and_output(batch_prompts, batch_completions, tokenizer)
        input_ids = batch_tok["input_ids"].to(device)
        labels = batch_tok["labels"].to(device)
        response_mask = batch_tok["response_mask"].to(device)

        model.eval()
        with torch.inference_mode():
            old_out = get_response_log_probs(
                model, input_ids, labels, return_token_entropy=False
            )
        old_log_probs = old_out["log_probs"].detach()
        model.train()

        advantages_1d, _raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn,
            batch_completions,
            batch_gts,
            hp.group_size,
            hp.advantage_eps,
            hp.normalize_by_std,
        )
        advantages = advantages_1d.to(device).unsqueeze(-1)

        train_scores = [answer_tag_reward_fn(c, g) for c, g in zip(batch_completions, batch_gts, strict=True)]
        train_metrics = _average_metric_dicts(train_scores)

        last_grad_norm = 0.0
        last_mb_losses: list[float] = []
        last_mb_entropies: list[float] = []
        last_mb_clip_fracs: list[float] = []

        for _epoch in range(hp.epochs_per_rollout_batch):
            optimizer.zero_grad(set_to_none=True)
            mb_losses: list[float] = []
            mb_entropies: list[float] = []
            mb_clip_fracs: list[float] = []

            for mb in range(hp.n_microbatches_per_rollout_batch):
                s = mb * hp.micro_train_batch_size
                e = s + hp.micro_train_batch_size
                ids_mb = input_ids[s:e]
                lab_mb = labels[s:e]
                mask_mb = response_mask[s:e]
                adv_mb = advantages[s:e]
                old_mb = old_log_probs[s:e]

                policy_log_probs, logits_mb = _policy_forward_log_probs(model, ids_mb, lab_mb)
                ent = compute_entropy(logits_mb)
                resp_tokens = mask_mb.sum().clamp(min=1)
                mb_entropies.append(float(((ent * mask_mb.float()).sum() / resp_tokens).item()))

                with torch.no_grad():
                    ratios = torch.exp(policy_log_probs.detach() - old_mb)
                    clipped = (ratios < 1.0 - hp.cliprange) | (ratios > 1.0 + hp.cliprange)
                    mb_clip_fracs.append(
                        float((clipped.float() * mask_mb.float()).sum() / resp_tokens)
                    )

                loss_tensor, _meta = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=mask_mb,
                    gradient_accumulation_steps=hp.gradient_accumulation_steps,
                    advantages=adv_mb,
                    old_log_probs=old_mb,
                    cliprange=hp.cliprange,
                )
                mb_losses.append(float(loss_tensor.item()))

            last_grad_norm = float(clip_grad_norm_(model.parameters(), hp.max_grad_norm))
            optimizer.step()
            last_mb_losses = mb_losses
            last_mb_entropies = mb_entropies
            last_mb_clip_fracs = mb_clip_fracs

        global_step += 1
        log_payload: dict[str, Any] = {
            "step": global_step,
            "loss": sum(last_mb_losses) / len(last_mb_losses),
            "grad_norm": last_grad_norm,
            "token_entropy_mean": sum(last_mb_entropies) / len(last_mb_entropies),
            "clip_fraction": sum(last_mb_clip_fracs) / len(last_mb_clip_fracs),
            "train_reward": train_metrics.get("reward", 0.0),
            "train_format_reward": train_metrics.get("format_reward", 0.0),
            "train_answer_reward": train_metrics.get("answer_reward", 0.0),
            "raw_reward_mean": reward_meta.get("raw_mean", 0.0),
        }
        logger.info(json.dumps(log_payload))
        if use_wandb:
            import wandb

            wandb.log(log_payload, step=global_step)
        (output_dir / f"step_{global_step:04d}.json").write_text(json.dumps(log_payload, indent=2), encoding="utf-8")

        if (global_step % hp.val_interval == 0) or (global_step == hp.n_grpo_steps):
            val_metrics, val_rollouts = _run_validation(model, tokenizer, val_examples, device, hp)
            logger.info(json.dumps(val_metrics))
            if use_wandb:
                import wandb

                wandb.log(val_metrics, step=global_step)
            (output_dir / f"val_step_{global_step:04d}.json").write_text(
                json.dumps(val_metrics, indent=2), encoding="utf-8"
            )
            rollouts_path = output_dir / f"val_rollouts_{global_step:04d}.json"
            rollouts_path.write_text(json.dumps(val_rollouts, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Wrote validation rollouts to %s", rollouts_path.resolve())

    ckpt_dir = save_model_dir if save_model_dir is not None else output_dir / "hf_checkpoint"
    _save_model_and_tokenizer(model, tokenizer, ckpt_dir)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO RL training (HF).")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument("--output-dir", type=Path, default=Path("alignment/results/rl_grpo"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-limit", type=int, default=512, help="Max GSM8K train rows to index from.")
    p.add_argument("--n-grpo-steps", type=int, default=None, help="Rollout+train iterations (default: GRPOHyperparams).")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument(
        "--normalize-by-std",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Divide group advantages by per-group std (default: on). Use --no-normalize-by-std for mean-only (Eq. 31 style).",
    )
    p.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Validation prompts per batched generate (default: GRPOHyperparams.val_batch_size, usually all val). Lower if OOM.",
    )
    p.add_argument(
        "--save-model-dir",
        type=Path,
        default=None,
        help="Directory for final HF checkpoint (default: <output-dir>/hf_checkpoint).",
    )
    return p.parse_args()


def main() -> None:
    _apply_hf_ssl_cert_workaround()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    dev = torch.device(args.device) if args.device else None
    train_grpo_loop(
        model_name=args.model_name,
        output_dir=args.output_dir,
        device=dev,
        seed=args.seed,
        train_limit=args.train_limit,
        use_wandb=args.wandb,
        save_model_dir=args.save_model_dir,
        n_grpo_steps=args.n_grpo_steps,
        normalize_by_std=args.normalize_by_std,
        val_batch_size=args.val_batch_size,
    )


if __name__ == "__main__":
    main()
