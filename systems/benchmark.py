from __future__ import annotations

import argparse
import os
import statistics
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F

from basics.model import BasicsTransformerLM


@dataclass(frozen=True)
class ModelSpec:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "small": ModelSpec(d_model=512, d_ff=2048, num_layers=8, num_heads=8),
    "medium": ModelSpec(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "large": ModelSpec(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    model_size: str
    context_length: int = 128
    batch_size: int = 4
    vocab_size: int = 10_000
    warmup_steps: int = 5
    measure_steps: int = 10
    mode: Literal["forward", "forward-backward", "train-step"] = "forward"
    use_bf16: bool = False
    use_memory_profiler: bool = False
    compile_model: bool = False
    compare_compiled: bool = False
    output_dir: Path = Path("artifacts")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark and profile the Basics transformer.")
    parser.add_argument("--model-size", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--mode", choices=["forward", "forward-backward", "train-step"], default="forward")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--use-memory-profiler", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument(
        "--compare-compiled",
        action="store_true",
        help="Run both vanilla and torch.compile and print a comparison table.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser


def build_model(config: BenchmarkConfig) -> torch.nn.Module:
    """Instantiate the staff Basics transformer for the requested model size."""
    spec = MODEL_SPECS[config.model_size]
    return BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=spec.d_model,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        d_ff=spec.d_ff,
        rope_theta=10_000.0,
    )


def make_random_batch(config: BenchmarkConfig, device: torch.device) -> torch.Tensor:
    """Construct a random token batch for benchmarking and profiling."""
    # +1 because we create (input_ids, labels) via shift.
    return torch.randint(
        0,
        config.vocab_size,
        (config.batch_size, config.context_length + 1),
        device=device,
        dtype=torch.long,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_single_step(
    model: torch.nn.Module,
    batch: torch.Tensor,
    mode: Literal["forward", "forward-backward", "train-step"],
    autocast_context,
) -> None:
    """Execute one benchmark step and synchronize CUDA before returning."""
    input_ids, labels = batch[:, :-1], batch[:, 1:]
    device = input_ids.device

    if device.type == "cuda":
        # Helps avoid cudagraph output reuse hazards if cudagraphs are enabled.
        mark = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
        if mark is not None:
            mark()

    optimizer = None
    if mode == "train-step":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    model.train(mode != "forward")
    _sync(device)
    with autocast_context:
        logits = model(input_ids)
        if mode == "forward":
            _sync(device)
            return

        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    _sync(device)

    loss.backward()
    if optimizer is not None:
        optimizer.step()
    _sync(device)


def benchmark_model(config: BenchmarkConfig) -> dict[str, float]:
    """Run warmup steps followed by timed measurement steps."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_context = make_autocast_context(config.use_bf16)

    model = build_model(config).to(device)
    if config.compile_model:
        if device.type == "cuda":
            os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")
        model = torch.compile(
            model,
            options={"triton.cudagraphs": False, "triton.cudagraph_trees": False},
        )

    batch = make_random_batch(config, device=device)

    maybe_start_memory_history(config.use_memory_profiler)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(config.warmup_steps):
        run_single_step(model, batch, mode=config.mode, autocast_context=autocast_context)

    times: list[float] = []
    for _ in range(config.measure_steps):
        _sync(device)
        t0 = time.perf_counter()
        run_single_step(model, batch, mode=config.mode, autocast_context=autocast_context)
        _sync(device)
        times.append(time.perf_counter() - t0)

    maybe_dump_memory_snapshot(config.use_memory_profiler, config.output_dir / "memory_snapshot.pickle")

    mean_ms = statistics.mean(times) * 1_000.0
    std_ms = statistics.pstdev(times) * 1_000.0
    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "device": 1.0 if device.type == "cuda" else 0.0,
    }


def benchmark_model_instance(
    *,
    model: torch.nn.Module,
    batch: torch.Tensor,
    mode: Literal["forward", "forward-backward", "train-step"],
    warmup_steps: int,
    measure_steps: int,
    use_bf16: bool,
) -> dict[str, float]:
    device = batch.device
    autocast_context = make_autocast_context(use_bf16)

    for _ in range(warmup_steps):
        run_single_step(model, batch, mode=mode, autocast_context=autocast_context)

    times: list[float] = []
    for _ in range(measure_steps):
        _sync(device)
        t0 = time.perf_counter()
        run_single_step(model, batch, mode=mode, autocast_context=autocast_context)
        _sync(device)
        times.append(time.perf_counter() - t0)

    return {
        "mean_ms": statistics.mean(times) * 1_000.0,
        "std_ms": statistics.pstdev(times) * 1_000.0,
    }


def annotated_scaled_dot_product_attention(*args, **kwargs):
    """Optional NVTX-annotated attention path for Nsight Systems profiling."""
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push("scaled_dot_product_attention")
    try:
        return F.scaled_dot_product_attention(*args, **kwargs)
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


def maybe_start_memory_history(enabled: bool) -> None:
    if enabled:
        if torch.cuda.is_available():
            try:  # pragma: no cover
                torch.cuda.memory._record_memory_history(enabled=True)
            except Exception:
                pass


def maybe_dump_memory_snapshot(enabled: bool, output_path: Path) -> None:
    if enabled:
        if torch.cuda.is_available():
            try:  # pragma: no cover
                torch.cuda.memory._dump_snapshot(str(output_path))
            except Exception:
                pass


def make_autocast_context(use_bf16: bool):
    if use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def main() -> None:
    args = build_argparser().parse_args()
    config = BenchmarkConfig(
        model_size=args.model_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        mode=args.mode,
        use_bf16=args.use_bf16,
        use_memory_profiler=args.use_memory_profiler,
        compile_model=args.compile_model,
        compare_compiled=args.compare_compiled,
        output_dir=args.output_dir,
    )

    if config.compare_compiled:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = make_random_batch(config, device=device)

        vanilla = build_model(config).to(device)

        if device.type == "cuda":
            os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")
        print("Compiling model with torch.compile (this may take a while)...")
        compiled = torch.compile(
            build_model(config).to(device),
            options={"triton.cudagraphs": False, "triton.cudagraph_trees": False},
        )

        modes: list[Literal["forward", "forward-backward", "train-step"]] = ["forward", "forward-backward", "train-step"]
        rows: list[dict[str, str | float]] = []
        for mode in modes:
            base_out = benchmark_model_instance(
                model=vanilla,
                batch=batch,
                mode=mode,
                warmup_steps=config.warmup_steps,
                measure_steps=config.measure_steps,
                use_bf16=config.use_bf16,
            )
            comp_out = benchmark_model_instance(
                model=compiled,
                batch=batch,
                mode=mode,
                warmup_steps=config.warmup_steps,
                measure_steps=config.measure_steps,
                use_bf16=config.use_bf16,
            )
            rows.append({"mode": mode, "compiled": "no", "mean_ms": base_out["mean_ms"], "std_ms": base_out["std_ms"]})
            rows.append({"mode": mode, "compiled": "yes", "mean_ms": comp_out["mean_ms"], "std_ms": comp_out["std_ms"]})

        headers = ["mode", "compiled", "mean_ms", "std_ms"]
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in rows:
            print(f'| {r["mode"]} | {r["compiled"]} | {float(r["mean_ms"]):.3f} | {float(r["std_ms"]):.3f} |')
        return

    out = benchmark_model(config)
    print(f"mean_ms={out['mean_ms']:.3f} std_ms={out['std_ms']:.3f}")


if __name__ == "__main__":
    main()
