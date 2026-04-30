from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from tqdm import tqdm


@dataclass(frozen=True)
class AttentionBenchmarkConfig:
    head_dims: tuple[int, ...] = (16, 32, 64, 128)
    sequence_lengths: tuple[int, ...] = (64, 128, 256, 512, 1024)
    batch_size: int = 8
    forward_passes: int = 100
    backward_passes: int = 100
    compile_attention: bool = False


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark attention implementations.")
    parser.add_argument("--compile-attention", action="store_true")
    return parser


def iter_benchmark_shapes(config: AttentionBenchmarkConfig) -> Iterable[tuple[int, int]]:
    for head_dim in config.head_dims:
        for sequence_length in config.sequence_lengths:
            yield head_dim, sequence_length


def make_qkv(batch_size: int, sequence_length: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Create random Q, K, and V tensors for the attention benchmark."""
    # We benchmark single-head attention: (batch, seq, d).
    q = torch.randn(batch_size, sequence_length, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch_size, sequence_length, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch_size, sequence_length, head_dim, device=device, dtype=torch.float32)
    return q, k, v


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Naive scaled dot-product attention (no mask)."""
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) * (d**-0.5)
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


class _AttentionModule(torch.nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return _attention(q, k, v)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _timeit_s(fn, iters: int, device: torch.device) -> list[float]:
    times: list[float] = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return times


def benchmark_attention_once(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> dict[str, float]:
    """Time the forward and backward pass for a single attention configuration."""
    device = q.device
    mod = _AttentionModule().to(device)

    # Warmup (also triggers compilation if mod is compiled by caller).
    with torch.inference_mode():
        for _ in range(5):
            _ = mod(q, k, v)
    _sync(device)

    # Forward-only timing.
    def fwd() -> None:
        with torch.inference_mode():
            _ = mod(q, k, v)

    # Backward timing (fresh graph each iter; use sum() to create scalar loss).
    def bwd() -> None:
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        out = mod(q_, k_, v_)
        out.sum().backward()

    fwd_times = _timeit_s(fwd, iters=100, device=device)
    bwd_times = _timeit_s(bwd, iters=100, device=device)
    return {
        "forward_mean_ms": statistics.mean(fwd_times) * 1_000.0,
        "forward_std_ms": statistics.pstdev(fwd_times) * 1_000.0,
        "backward_mean_ms": statistics.mean(bwd_times) * 1_000.0,
        "backward_std_ms": statistics.pstdev(bwd_times) * 1_000.0,
    }


def benchmark_attention_grid(config: AttentionBenchmarkConfig) -> list[dict[str, float | int | str]]:
    """Run the attention benchmark over the Section 2.7 Cartesian product of scales."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: list[dict[str, float | int | str]] = []
    shapes = list(iter_benchmark_shapes(config))
    for head_dim, seq_len in tqdm(shapes, desc=f"attention_benchmark[{device.type}]", unit="shape"):
        q, k, v = make_qkv(config.batch_size, seq_len, head_dim, device=device)

        # Uncompiled.
        base = _AttentionModule().to(device)

        def run(mod: torch.nn.Module) -> dict[str, float]:
            # Warmup / compilation trigger.
            # Use no_grad (not inference_mode) to avoid marking tensors as inference tensors
            # which can trip cudagraph capture in the compiled backward benchmark.
            with torch.no_grad():
                for _ in range(5):
                    _ = mod(q, k, v)
            _sync(device)

            def fwd() -> None:
                with torch.no_grad():
                    _ = mod(q, k, v)

            def bwd() -> None:
                # Clone to ensure these are normal (non-inference) tensors.
                q_ = q.detach().clone().requires_grad_(True)
                k_ = k.detach().clone().requires_grad_(True)
                v_ = v.detach().clone().requires_grad_(True)
                out = mod(q_, k_, v_)
                out.sum().backward()

            fwd_times = _timeit_s(fwd, iters=config.forward_passes, device=device)
            bwd_times = _timeit_s(bwd, iters=config.backward_passes, device=device)
            return {
                "forward_mean_ms": statistics.mean(fwd_times) * 1_000.0,
                "forward_std_ms": statistics.pstdev(fwd_times) * 1_000.0,
                "backward_mean_ms": statistics.mean(bwd_times) * 1_000.0,
                "backward_std_ms": statistics.pstdev(bwd_times) * 1_000.0,
            }

        base_stats = run(base)
        rows.append(
            {
                "compiled": "no",
                "device": device.type,
                "batch_size": config.batch_size,
                "sequence_length": seq_len,
                "head_dim": head_dim,
                **base_stats,
            }
        )

        if config.compile_attention:
            # Inductor may try to cudagraph capture; disable for reliability in this benchmark.
            if device.type == "cuda":
                try:  # pragma: no cover
                    import torch._inductor.config as inductor_config

                    inductor_config.triton.cudagraphs = False
                except Exception:
                    pass
            compiled = torch.compile(base, mode="reduce-overhead")
            comp_stats = run(compiled)
            rows.append(
                {
                    "compiled": "yes",
                    "device": device.type,
                    "batch_size": config.batch_size,
                    "sequence_length": seq_len,
                    "head_dim": head_dim,
                    **comp_stats,
                }
            )

    # Print a compact markdown table.
    headers = [
        "compiled",
        "device",
        "batch_size",
        "sequence_length",
        "head_dim",
        "forward_mean_ms",
        "backward_mean_ms",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        fwd_ms = float(row["forward_mean_ms"])
        bwd_ms = float(row["backward_mean_ms"])
        print(
            "| "
            + " | ".join(
                [
                    str(row["compiled"]),
                    str(row["device"]),
                    str(row["batch_size"]),
                    str(row["sequence_length"]),
                    str(row["head_dim"]),
                    f"{fwd_ms:.3f}",
                    f"{bwd_ms:.3f}",
                ]
            )
            + " |"
        )

    return rows


def main() -> None:
    args = build_argparser().parse_args()
    config = AttentionBenchmarkConfig(compile_attention=args.compile_attention)
    benchmark_attention_grid(config)


if __name__ == "__main__":
    main()
