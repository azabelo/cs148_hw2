# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

EE/CS 148B HW 2 — a Python ML/AI educational project covering profiling/benchmarking (Section 2) and GRPO alignment training (Section 3). See `README.md` for full layout.

### Dependencies and package management

- Managed with **`uv`** (`uv sync` from repo root). Lock file: `uv.lock`.
- The `basics/` subdirectory is an editable local package installed via `[tool.uv.sources]`.
- Python requirement: `>=3.11, <3.13`. The VM ships Python 3.12.

### Running tests

```sh
uv run pytest -v ./tests
```

All 13 tests are CPU-only and use lightweight toy fixtures (no model downloads or GPU needed).

### Key caveats

- **No GPU on Cloud Agent VMs**: The `systems/` profiling scripts and full GRPO training require CUDA. Tests and helper-function validation work fine on CPU.
- **`BasicsTransformerLM` constructor** requires `rope_theta` as a positional/keyword argument (not `attn_pdrop`/`residual_pdrop`).
- **HuggingFace Hub warning**: Unauthenticated requests work for public models (e.g., `gpt2` tokenizer) but may be rate-limited. Set `HF_TOKEN` env var if needed.
- **vLLM** is intentionally excluded from the default env. Use the separate `vllm-eval/` sub-project if needed (see `README.md`).
- **No linter configured** in `pyproject.toml` — there are no lint commands to run.
