"""End to end, in one process (dense first: the sparse conversion is one-way):

  1. dense mode reproduces `mlx_lm`'s own greedy generation token for token (the same
     blocks, the same ops, so bit-identical)
  2. sparse mode (neuron-major down, checkpoint layout freed) matches dense for at least
     the first 10 tokens; the two may diverge later on an argmax near-tie
  3. MTP speculative decoding (k=2 drafts) produces the same text as plain sparse decode,
     again up to a near-tie

    python -m pytest tests/test_end_to_end.py -v -s
    python -m tests.test_end_to_end
"""
import json
import os

import mlx.core as mx
import pytest

from hsd.model import NemotronModel
from hsd.mtp import MTPHead
from hsd.resolve import resolve_head

MAX_TOKENS = 24
MATCH_AT_LEAST = 10
eos = {2}             # replaced inside the `loaded` fixture from the model's generation_config


def take_ids(model, prompt, n, temp=0.0, mtp=None, n_draft=0):
    """All n generated token ids, without the end-of-sequence token (mlx-lm's generate
    stops before yielding it; our loop yields it). The last token arrives with the stats,
    so the stats' own token list is the authoritative one."""
    gen = (model.generate_mtp(prompt, mtp, n_draft=n_draft, max_tokens=n) if mtp
           else model.generate(prompt, n, temp))
    stats = None
    for t, st in gen:
        if st is not None:
            stats = st
    return [t for t in stats["tokens"] if t not in eos], stats


@pytest.fixture(scope="module")
def loaded(model_dir, head_dir, wired):
    """The model, dense first; also the chat prompt and mlx-lm's own greedy tokens for it."""
    global eos
    model = NemotronModel(model_dir, mode="dense", verbose=False)
    e = json.load(open(os.path.join(model.path, "generation_config.json"))).get("eos_token_id", 2)
    eos = set(tuple(e) if isinstance(e, list) else (e,))
    text = "Write two sentences about the sea."
    prompt = list(model.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                      add_generation_prompt=True, enable_thinking=False))
    from mlx_lm.generate import stream_generate
    ref = [int(r.token) for r in stream_generate(model.model, model.tokenizer, prompt, max_tokens=MAX_TOKENS)]
    return model, prompt, ref


def test_dense_reproduces_mlx_lm_greedy(loaded):
    """(1) our dense decode loop against mlx-lm's own generate, same prompt, greedy."""
    model, prompt, ref = loaded
    out, stats = take_ids(model, prompt, MAX_TOKENS)
    n = min(len(out), len(ref))
    same = next((i for i in range(n) if out[i] != ref[i]), n)
    print(f"  dense vs mlx_lm.generate: identical for the first {same}/{n} tokens "
          f"({stats['decode_tps']:.1f} tok/s ours)")
    assert out == ref, (out, ref)


def test_sparse_matches_dense(loaded):
    """(2) sparse mode: the same tokens as dense for at least the first MATCH_AT_LEAST
    tokens (argmax near-ties can flip later)."""
    model, prompt, ref = loaded
    model.set_mode("sparse")
    out, stats = take_ids(model, prompt, MAX_TOKENS)
    n = min(len(out), len(ref))
    same = next((i for i in range(n) if out[i] != ref[i]), n)
    print(f"  sparse vs dense: identical for the first {same}/{n} tokens "
          f"({stats['decode_tps']:.1f} tok/s sparse, peak {stats['peak_gb']:.1f} GB)")
    assert same >= MATCH_AT_LEAST, (same, out, ref)
    print(f"  text: {model.tokenizer.decode(out)!r}")


def test_mtp_matches_plain_sparse(loaded):
    """(3) MTP speculative decoding (k=2) against plain sparse decode: same text, up to a
    near-tie; and the acceptance statistics come back with the run."""
    model, prompt, ref = loaded
    if model.mode != "sparse":
        model.set_mode("sparse")
    plain, _ = take_ids(model, prompt, MAX_TOKENS)
    mtp = MTPHead(model, resolve_head(download=False))
    out, stats = take_ids(model, prompt, MAX_TOKENS, mtp=mtp, n_draft=2)
    n = min(len(out), len(plain))
    same = next((i for i in range(n) if out[i] != plain[i]), n)
    sp = stats["spec"]
    print(f"  mtp(k=2) vs plain sparse: identical for the first {same}/{n} tokens | "
          f"{sp['steps']} steps, {sp['tokens_per_step']:.2f} tok/step, accepted {sp['accepted_mean']:.2f}, "
          f"hist {sp['accepted_hist']} | {stats['decode_tps']:.1f} tok/s")
    assert same >= MATCH_AT_LEAST, (same, out, plain)
    assert model.tokenizer.decode(out) == model.tokenizer.decode(plain[:same])


def main():
    import sys
    sys.exit(__import__("pytest").main([__file__, "-v", "-s"]))


if __name__ == "__main__":
    main()
