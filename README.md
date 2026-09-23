# mlx-heimr-sparse-down

Sparse MoE down projection and multi-token-prediction (MTP) speculative decoding
for NVIDIA's Nemotron 3.5 Lightning 30B A3B on Apple silicon with MLX, from the
stock [mlx-community 4-bit checkpoint](https://huggingface.co/mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit),
without re-quantising anything.

## Results

Measured on an Apple M5 Pro (48 GB): greedy decoding of 256 tokens after a
short chat prompt, each cell the mean of a prose and a code prompt, one
configuration per process.

| configuration | decode speed |
|---|---|
| stock mlx-lm (dense down) | 105 tok/s |
| sparse down | 120 tok/s |
| stock mlx-lm + MTP 2 drafts | 127 tok/s |
| sparse down + MTP 2 drafts | 153 tok/s |

With MTP 2 the head accepts 1.26 of its 2 drafts on the prose prompt and
1.68 on the code prompt; the MTP gain follows draft acceptance, which depends
on the text. The full measurements are in the technical writeup linked below.

## How it works

After the experts' ReLU² activation most neurons sit at exactly zero, so the
down projection is transposed at load time — a pure permutation of the
checkpoint's own bytes — and a Metal kernel reads only the rows of the neurons
that fired, never the rows behind a zero. A second kernel runs prefill on the
same transposed layout with Metal's matrix units, so the model needs no more
memory than stock mlx-lm, and the model's own MTP head adds greedy speculative
decoding, including the Mamba-2 state rollback that verifying drafted tokens
requires. We describe the technique in detail, with measurements, in [Skipping
dead neurons: a faster Nemotron 3.5 Lightning on Apple
silicon](https://sevren.ai/blog/heimr-sparse-down/), a technical writeup on
our website.

## Requirements

- Apple silicon, macOS 26 or newer (the prefill kernel's `matmul2d` needs it)
- 32 GB of memory recommended
- Python 3.12, `mlx==0.32.2`, `mlx-lm==0.31.3` (pinned; the kernels are built
  against this exact pair)

## Install

```sh
uv sync --extra test
```

## Getting the model

The default model id is
`mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit` (affine 4-bit, group
size 64, about 17.7 GB). A local path can be passed with `--model`; without
one, the checkpoint is resolved from LM Studio, the Hugging Face cache, or
downloaded on first use. To fetch it explicitly:

```sh
hf download mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit
```

## Getting the MTP head

The MTP head is the model's own (NVIDIA ships it as `mtp.layers.*`; the
mlx-community 4-bit export drops those tensors). We publish it separately,
quantised to the same affine 4-bit group-64 format, 772 MB:

```sh
hf download sevren-ai/nemotron-3.5-lightning-mtp-head-mlx-4bit
```

A local directory can be passed with `--mtp-head`; without one, the head is
resolved like the model. Without the head, plain and sparse generation still
work (the MTP path needs the file).

## Usage

```sh
uv run heimr-sparse-down --prompt "Why are whales not fish?"
uv run heimr-sparse-down --down dense --prompt "Why are whales not fish?"  # stock mlx-lm blocks
uv run heimr-sparse-down --mtp 2 --prompt-code                           # sparse + MTP, 2 drafts
uv run heimr-sparse-down --prompt-file texts/moby_dick.txt --prompt-tokens 2048 --raw --max-tokens 256
```

`python -m hsd` is the same entry point. Useful flags: `--max-tokens`, `--temp`
(plain decode only; MTP is greedy), `--chunk` (prefill chunk size), `--quiet`,
`--json results/run.json` (append a run record with device, versions and
timings). MTP is greedy only, and its benefit depends on how predictable the
text is; `--mtp 0` turns it off.

## Tests

```sh
uv run pytest -q
```

The tests need the checkpoint, and the end-to-end test needs the MTP head; both
skip with a message when absent. They check that the layout permutation is
exact and self-inverse, that the decode and prefill kernels match dequantised
references, that the multi-row verify step equals single steps and that the
rollback leaves the caches where single steps would, and that dense, sparse and
MTP modes all produce `mlx_lm.generate`'s greedy tokens.

## Layout

```text
hsd/      the package: layout permutation, Metal kernels, model wrapper, MTP drafter, CLI
bench/    measurement scripts (one configuration per process; output goes to results/)
scripts/  helper scripts (the code-corpus prompt builder)
tests/    kernel, rollback and end-to-end tests
texts/    prompt texts
results/  measurement output (git-ignored; results/README.md is a committed record)
```

## License

The code in this repository is MIT (see LICENSE), copyright 2026 Sevren ApS. The
Metal kernels are written against Apple's Metal Shading Language and
MetalPerformancePrimitives headers, which the macOS SDK provides at compile
time; no Apple source is vendored here. `texts/moby_dick.txt` is the Project
Gutenberg edition of Moby-Dick, public domain, and not covered by the MIT
License. The model and the MTP head are NVIDIA's, under the NVIDIA Open Model
License; the checkpoint is mlx-community's export of it.
