# AGENTS.md

mlx-heimr-sparse-down is a Python and MLX implementation of a sparse MoE
down projection and MTP speculative decoding for NVIDIA's Nemotron 3.5
Lightning 30B A3B on Apple silicon, from the stock mlx-community 4-bit
checkpoint. Follow the current code and project configuration rather than
assuming a fixed layout or toolchain.

## References

- Technical writeup (the layout permutation, the decode and prefill kernels,
  the MTP path, with measurements):
  https://sevren.ai/blog/heimr-sparse-down/
- mlx-lm's own model code (`mlx_lm.models.nemotron_h`, `switch_layers`, `ssm`)
  is the reference for block behaviour; this repository wraps it, it does not
  fork it. Dense mode must stay block-for-block what `mlx_lm.generate` runs.

## Development

- Read the relevant code, tests, `README.md`, and `pyproject.toml` before
  making changes. These are the source of truth for supported commands and
  conventions.
- `uv` is the toolchain: `uv sync --extra test` to set up, `uv run pytest -q`
  to test. Python 3.12. `mlx==0.32.2` and `mlx-lm==0.31.3` are pinned because
  the kernels are built against this exact pair's conventions; change the pins
  deliberately, never in passing.
- Prefer small, focused changes. Avoid speculative abstractions and
  compatibility layers for unreleased behavior.
- Use type annotations for public APIs and document non-obvious tensor
  shapes, layouts, dtypes, and numerical assumptions.
- Add dependencies only when existing dependencies are insufficient.
- Update tests and documentation with behavior changes. Run the narrowest
  relevant checks first, then the documented full suite when practical.

## MLX

- Keep compute paths in MLX. Avoid unnecessary NumPy, Python-scalar, or other
  framework conversions.
- Respect lazy execution. Use `mx.eval(...)` at intentional synchronization
  points, especially for correctness checks and benchmarks, not in hot paths.
- Be explicit about precision and accumulation dtypes. Do not silently cast
  public inputs.
- The neuron-major layout permutation must stay exact and self-inverse: no
  value may be re-quantised, only moved, so the dequantised products are the
  ones MLX itself would form and a result differs from the dense path only by
  fp32 summation order.
- Keep exactly one down layout resident. Sparse mode must never cost more
  memory than dense mode; a second layout or a per-prompt transpose defeats
  the point.
- Custom Metal kernels must state supported shapes and dtypes, handle
  boundaries, and remain testable against a reference implementation.

## Correctness

- The tests need the real 17.7 GB checkpoint, and the end-to-end test needs
  the MTP head; both skip with a message when absent rather than downloading.
- Compare optimized paths with a straightforward dequantised reference using
  dtype-appropriate tolerances, and against MLX's own dense quantised path.
- Dense, sparse and MTP modes must produce `mlx_lm.generate`'s greedy tokens
  exactly on the test prompts (MTP up to argmax near-ties, which is what
  greedy verify-and-rollback guarantees).
- A multi-row verify step must equal the same rows processed as single
  steps, and the rollback (attention caches trimmed, Mamba states recomputed
  over the accepted prefix) must leave every cache where single steps would.

## Performance

- One configuration per process; the GPU must never run two benchmarks at
  once.
- Warm the kernels up before timing: Metal kernel compilation and
  first-launch costs must land outside the timed region. Separate compilation
  from steady-state measurements.
- Record the device, the MLX version, this repository's commit and a UTC
  timestamp with every measurement (`hsd.runtime.record_meta` and the CLI's
  `--json`).
- Support performance claims with reproducible measurements; report memory
  and numerical tradeoffs where relevant.
