"""The multi-row verify step computes what B single decode steps compute, and the rollback
puts the caches where B single steps would have left them.

On a real prompt, with the model in sparse mode (the production configuration):

  1. verify_step on B rows  vs  B single steps: logits (argmax agreement and relative
     difference per position), the final residual stream, and the caches (Mamba states,
     conv states, KV caches)
  2. verify_step on B rows, rollback to N, then the remaining B-N rows as single steps:
     the same end state as (1)
  3. verify_step on B rows then rollback to N  vs  N single steps directly

The two paths form the same products in a different order (one multi-row GEMM and scan vs
B single-row ones), so differences sit at the bf16 level; an argmax can flip on a near-tie.

    python -m pytest tests/test_verify_rollback.py -v -s
    python -m tests.test_verify_rollback
"""
import copy

import mlx.core as mx
import pytest

from hsd.model import NemotronModel

B = 4
N = 2                       # rollback target: keep N of the B rows
PROMPT = ("Whales are not fish because they are warm-blooded mammals that breathe air "
          "through lungs, unlike fish, which")
# bf16-level bounds: the two paths share their products and differ in summation order only,
# so the differences are measured against the largest magnitude across all layers, not
# per layer (a single layer's conv state can sit entirely near zero). The conv state and
# KV cache inherit the residual stream's own difference (~1.4e-2 here); the Mamba state,
# which the rollback recomputes, stays an order below it.
RESID_BOUND = 5e-2
CACHE_BOUND = 5e-2


def clone_cache(cache):
    out = []
    for c in cache:
        if c is None:
            out.append(None)
            continue
        d = copy.copy(c)
        if hasattr(c, "cache"):                              # ArraysCache (Mamba-2)
            d.cache = [None if s is None else mx.array(s) for s in c.cache]
        else:                                                # KVCache (attention)
            d.keys = None if c.keys is None else mx.array(c.keys)
            d.values = None if c.values is None else mx.array(c.values)
        out.append(d)
    return out


def rel(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    return float((a - b).abs().max() / (b.abs().max() + 1e-12))


def run_single_steps(model, cache0, xin, n):
    """n single decode steps from a clone of cache0 -> (residual [n, D], logits [n, V], cache)."""
    c = clone_cache(cache0)
    model._cache = c
    xs, ls = [], []
    for i in range(n):
        x = model.step(xin[i:i + 1])
        xs.append(x)
        ls.append(model.logits(x).astype(mx.float32))
    mx.eval(*xs, *ls, *[s for cc in c if cc is not None for s in cc.state])
    return mx.concatenate(xs), mx.concatenate(ls), c


def compare_caches(model, ca, cb, label):
    """The caches after the same tokens by two paths. Differences are accumulated globally:
    max |a - b| over every layer, divided by max |b| over every layer."""
    diff = {"mamba_state": 0.0, "conv": 0.0, "kv": 0.0}
    mag = {"mamba_state": 1e-12, "conv": 1e-12, "kv": 1e-12}
    for l, t in enumerate(model.types):
        if t == "mamba":
            for i, k in ((0, "conv"), (1, "mamba_state")):
                a = ca[l][i].astype(mx.float32)
                b = cb[l][i].astype(mx.float32)
                diff[k] = max(diff[k], float((a - b).abs().max()))
                mag[k] = max(mag[k], float(b.abs().max()))
        elif t == "attention":
            assert ca[l].offset == cb[l].offset, (l, ca[l].offset, cb[l].offset)
            o = ca[l].offset
            for x, y in ((ca[l].keys[..., :o, :], cb[l].keys[..., :o, :]),
                         (ca[l].values[..., :o, :], cb[l].values[..., :o, :])):
                a = x.astype(mx.float32)
                b = y.astype(mx.float32)
                diff["kv"] = max(diff["kv"], float((a - b).abs().max()))
                mag["kv"] = max(mag["kv"], float(b.abs().max()))
    worst = {k: diff[k] / mag[k] for k in diff}
    print(f"  {label}: max relative diff (global)  mamba state {worst['mamba_state']:.2e}  "
          f"conv state {worst['conv']:.2e}  kv {worst['kv']:.2e}")
    for k, v in worst.items():
        assert v < CACHE_BOUND, (k, v)
    return worst


@pytest.fixture(scope="module")
def setup(model_dir, wired):
    model = NemotronModel(model_dir, mode="sparse", verbose=False)
    ids = [1] + model.tokenizer.encode(PROMPT, add_special_tokens=False)
    prompt, cont = ids[:-B], ids[-B:]
    cache0 = model.make_cache()
    h = model.prefill(mx.array(prompt, dtype=mx.int32), cache0)
    mx.eval(h)
    x0 = model.embed(mx.array(cont[:1], dtype=mx.int32))                 # the anchor row
    toks = mx.array(cont[1:], dtype=mx.int32)                           # B-1 "drafts" (the true continuation)
    xin = mx.concatenate([x0, model.embed(toks)])                      # [B, D]
    mx.eval(xin)
    return model, cache0, xin


def test_verify_equals_b_single_steps(setup):
    """(1) one verify step on B rows vs B single decode steps: same logits, same residual
    stream, same caches."""
    model, cache0, xin = setup
    resid_b, logits_b, cb = run_single_steps(model, cache0, xin, B)

    ca = clone_cache(cache0)
    xa = model.verify_step(xin, ca)
    logits_a = model.logits(xa).astype(mx.float32)
    mx.eval(xa, logits_a, *[s for c in ca if c is not None for s in c.state])

    agree = int((mx.argmax(logits_a, -1) == mx.argmax(logits_b, -1)).sum().item())
    per_pos = " ".join(f"{rel(logits_a[i], logits_b[i]):.2e}" for i in range(B))
    resid = rel(xa, resid_b)
    print(f"  B={B}: argmax agree {agree}/{B} | per-position logits rel diff {per_pos} "
          f"| final residual rel diff {resid:.2e}")
    assert agree == B, "an argmax flipped between the multi-row and single-row paths (near-tie)"
    assert resid < RESID_BOUND, resid
    compare_caches(model, ca, cb, f"caches after the {B} tokens")


def test_rollback_then_single_steps(setup):
    """(2) verify(B) -> rollback to N -> single steps for the rest: same end state as (1)."""
    model, cache0, xin = setup
    resid_b, logits_b, cb = run_single_steps(model, cache0, xin, B)
    cc = clone_cache(cache0)
    model.verify_step(xin, cc)
    model.rollback(cc, N, B)
    mx.eval(*[s for c in cc if c is not None for s in c.state])
    model._cache = cc
    for i in range(N, B):
        mx.eval(model.step(xin[i:i + 1]))
    compare_caches(model, cc, cb, f"caches after verify({B}) -> rollback to {N} -> {B - N} single steps")


def test_rollback_equals_n_single_steps(setup):
    """(3) verify(B) -> rollback to N  vs  N single steps directly."""
    model, cache0, xin = setup
    resid_d, logits_d, cd = run_single_steps(model, cache0, xin, N)
    ce = clone_cache(cache0)
    model.verify_step(xin, ce)
    model.rollback(ce, N, B)
    mx.eval(*[s for c in ce if c is not None for s in c.state])
    compare_caches(model, ce, cd, f"caches after verify({B}) -> rollback to {N}  vs  {N} single steps")


def main():
    import sys
    sys.exit(__import__("pytest").main([__file__, "-v", "-s"]))


if __name__ == "__main__":
    main()
