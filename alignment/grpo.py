from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, Tensor]:
    """Tokenize prompt/output pairs and build a response mask over the labels."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0) or 0

    rows: list[tuple[list[int], list[int], list[int]]] = []
    for p, o in zip(prompt_strs, output_strs, strict=True):
        p_ids = tokenizer.encode(p, add_special_tokens=False)
        o_ids = tokenizer.encode(o, add_special_tokens=False)
        rows.append((p_ids, o_ids, p_ids + o_ids))

    max_len = max(len(seq) - 1 for *_, seq in rows)
    input_ids_list, labels_list, mask_list = [], [], []
    for p_ids, o_ids, seq in rows:
        pl, rl = len(p_ids), len(o_ids)
        sl = len(seq) - 1
        pad = [pad_id] * (max_len - sl)
        input_ids_list.append(seq[:-1] + pad)
        labels_list.append(seq[1:] + pad)
        mask_list.append([False] * (pl - 1) + [True] * rl + [False] * (max_len - sl))

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "response_mask": torch.tensor(mask_list, dtype=torch.bool),
    }


def compute_entropy(logits: Tensor) -> Tensor:
    """Compute per-token entropies over the vocabulary dimension."""
    log_p = F.log_softmax(logits, dim=-1)
    return -(log_p.exp() * log_p).sum(dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Score conditional log-probabilities for a batch of prompt/response examples."""
    with torch.inference_mode():
        logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    out: dict[str, Tensor] = {"log_probs": log_probs}
    if return_token_entropy:
        out["token_entropy"] = compute_entropy(logits)
    return out


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> Tensor:
    """Sum over masked elements and normalize by the provided constant."""
    m = mask.to(dtype=tensor.dtype)
    num = tensor * m
    if dim is None:
        return num.sum() / normalize_constant
    return num.sum(dim=dim) / normalize_constant


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Compute raw rewards and per-group normalized advantages for GRPO."""
    raw = torch.tensor(
        [float(reward_fn(r, gt)["reward"]) for r, gt in zip(rollout_responses, repeated_ground_truths, strict=True)],
        dtype=torch.float32,
    )
    g = raw.view(-1, group_size)
    mean = g.mean(dim=1, keepdim=True)
    centered = g - mean
    if normalize_by_std:
        adv = centered / (g.std(dim=1, keepdim=True, unbiased=False) + advantage_eps)
    else:
        adv = centered
    advantages = adv.reshape(-1)
    meta = {
        "raw_mean": float(raw.mean().item()),
        "raw_std": float(raw.std(unbiased=False).item()),
        "raw_min": float(raw.min().item()),
        "raw_max": float(raw.max().item()),
    }
    return advantages, raw, meta


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the per-token GRPO-Clip loss."""
    ratios = torch.exp(policy_log_probs - old_log_probs)
    clipped = torch.clamp(ratios, 1.0 - cliprange, 1.0 + cliprange)
    a = advantages.expand_as(policy_log_probs)
    unclipped = ratios * a
    clipped_term = clipped * a
    loss = -torch.minimum(unclipped, clipped_term)
    meta = {
        "used_clipped": (clipped_term <= unclipped).to(torch.bool),
        "ratios": ratios,
    }
    return loss, meta


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Backpropagate a single GRPO microbatch loss."""
    per_tok, clip_meta = compute_grpo_clip_loss(
        advantages, policy_log_probs, old_log_probs, cliprange
    )
    m = response_mask.to(dtype=per_tok.dtype)
    denom = response_mask.sum(dim=1).to(dtype=per_tok.dtype)
    per_ex = (per_tok * m).sum(dim=1) / denom
    loss = per_ex.mean() / float(gradient_accumulation_steps)
    loss.backward()
    return loss.detach(), clip_meta


def log_generations(
    prompts: Sequence[str],
    responses: Sequence[str],
    ground_truths: Sequence[str],
    reward_infos: Sequence[dict[str, float]],
    token_entropies: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Create serializable generation logs for debugging training runs."""
    logs: list[dict[str, Any]] = []
    lens: list[int] = []
    lens_ok: list[int] = []
    lens_bad: list[int] = []

    for i, (p, r, gt, rw) in enumerate(
        zip(prompts, responses, ground_truths, reward_infos, strict=True)
    ):
        fmt = float(rw.get("format_reward", 0.0))
        ans = float(rw.get("answer_reward", 0.0))
        tot = float(rw.get("reward", rw.get("total_reward", 0.0)))
        ntok = len(r.split())
        lens.append(ntok)
        correct = ans > 0.0
        (lens_ok if correct else lens_bad).append(ntok)

        row: dict[str, Any] = {
            "prompt": p,
            "response": r,
            "ground_truth": gt,
            "format_reward": fmt,
            "answer_reward": ans,
            "reward": tot,
            "response_length_tokens": ntok,
        }
        if token_entropies is not None:
            row["avg_response_token_entropy"] = float(token_entropies[i])
        logs.append(row)

    def _mean(xs: list[int]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    logs.append(
        {
            "batch_summary": True,
            "avg_response_length": _mean(lens),
            "avg_response_length_correct": _mean(lens_ok),
            "avg_response_length_incorrect": _mean(lens_bad),
        }
    )
    return logs


def train_grpo(*args, **kwargs) -> dict[str, Any]:
    """Run the full GRPO training loop from Section 3.5."""
    raise NotImplementedError
