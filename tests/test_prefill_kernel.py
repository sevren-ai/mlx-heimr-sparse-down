"""The matrix-unit prefill kernel matches MLX's dense block on a 2048-token chunk of real
activations, at the bf16 level.

Loads the full model with BOTH down layouts resident (the production wrapper frees the
checkpoint's one; the comparison needs them), runs 2048 tokens of Moby Dick through the
model to capture a MoE layer's input residual stream, then runs the same chunk through
the stock block and through the sparse kernel.

    python -m pytest tests/test_prefill_kernel.py -v -s
    python -m tests.test_prefill_kernel
"""
import os

import mlx.core as mx
import pytest

from hsd.model import NemotronModel

NTOK = 2048
ERR_BOUND = 5e-3          # relative error at the bf16 level


def moby_ids(model, n=NTOK):
    text = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "texts", "moby_dick.txt"), encoding="utf-8", errors="replace").read()
    ids = model.tokenizer.encode(text, add_special_tokens=False)
    return [1] + list(ids[2000:2000 + n - 1])


@pytest.fixture(scope="module")
def captured(model_dir, wired):
    """The model with both layouts resident, and the residual stream entering one MoE layer
    on a real 2048-token pass."""
    layer = 10
    model = NemotronModel(model_dir, mode="sparse", verbose=False, single_layout=False)
    blk = model.blocks[layer]
    caps = {}
    orig = blk.prefill

    def prefill_capture(x):
        caps.setdefault("x", []).append(x)
        return orig(x)

    blk.prefill = prefill_capture
    ids = mx.array(moby_ids(model), dtype=mx.int32)
    model.prefill(ids, model.make_cache(), chunk=NTOK)
    blk.prefill = orig
    x = caps["x"][0]
    mx.eval(x)
    print(f"  captured layer {layer} on {x.shape[0]} real tokens")
    return model, blk, x


def test_prefill_kernel_matches_dense_block(captured):
    model, blk, x = captured
    T = x.shape[0]
    assert T == NTOK
    ref = blk.block(x[None])[0]                  # the stock block: mlx-lm's own sorted gather_qmm
    out = blk.prefill_sparse(x)                  # our kernel: tile order, union, gather, GEMM
    ref32, out32 = ref.astype(mx.float32), out.astype(mx.float32)
    mx.eval(ref32, out32)
    err = float((out32 - ref32).abs().max() / ref32.abs().max())
    live = float((blk.pack(*blk.front(x))[0] != 0).mean().item())
    print(f"  live neurons {100 * live:.1f}%  max rel err vs MLX dense block: {err:.2e} (bound {ERR_BOUND:.0e})")
    assert err < ERR_BOUND, err


def test_prefill_kernel_tt_variants(captured):
    """The 32- and 64-slot tile settings (the dense layers use 64) agree with the dense
    block too."""
    model, blk, x = captured
    ref = blk.block(x[None])[0].astype(mx.float32)
    for tt in (32, 64):
        blk.prefill_kw["tt"] = tt
        out = blk.prefill_sparse(x).astype(mx.float32)
        mx.eval(out)
        err = float((out - ref).abs().max() / ref.abs().max())
        print(f"  tt={tt}: max rel err vs MLX dense block {err:.2e}")
        assert err < ERR_BOUND, (tt, err)


def main():
    import sys
    sys.exit(__import__("pytest").main([__file__, "-v", "-s"]))


if __name__ == "__main__":
    main()
