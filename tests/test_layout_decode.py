"""The layout permutation is exact and self-inverse, and the sparse decode kernel matches
an fp32 dequantised reference on one MoE layer's real weights.

Run with pytest, or as a module:

    python -m pytest tests/test_layout_decode.py -v
    python -m tests.test_layout_decode
"""
import json
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from hsd.decode import sparse_down
from hsd.layout import nibble_transpose, to_neuron_major

GS, BITS, TOPK = 64, 4, 6
LAYER = 10          # a MoE layer of the 52


def load_moe_layer(model_dir, l=LAYER):
    """One MoE layer's quantised tensors straight from the checkpoint's safetensors."""
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    p = f"backbone.layers.{l}.mixer"
    need = [f"{p}.switch_mlp.fc1.{k}" for k in ("weight", "scales", "biases")] \
        + [f"{p}.switch_mlp.fc2.{k}" for k in ("weight", "scales", "biases")] \
        + [f"{p}.shared_experts.{n}.{k}" for n in ("up_proj", "down_proj") for k in ("weight", "scales", "biases")]
    W = {}
    for fn in sorted(set(idx[k] for k in need)):
        d = mx.load(os.path.join(model_dir, fn))
        W.update({k: d[k] for k in need if k in d})
    return W


def nibble_transpose_np(w):
    """A slow numpy reference of the nibble transpose: unpack every word to 8 nibbles, move
    them, pack them again."""
    lead = w.shape[:-2]
    R, CW = w.shape[-2], w.shape[-1]
    C = CW * 8
    a = np.asarray(w)                                      # uint32 [.., R, CW]
    nib = np.zeros((*lead, R, C), dtype=np.uint32)
    for b in range(8):
        nib[..., b::8] = (a >> np.uint32(4 * b)) & np.uint32(0xF)
    w2 = nib.transpose(*range(len(lead)), -1, -2)          # [.., C, R]  W[c, r]
    out = np.zeros((*lead, C, R // 8, 8), dtype=np.uint32)
    for b in range(8):
        out[..., b] = w2[..., b::8] << np.uint32(4 * b)
    return mx.array(out.sum(axis=-1).astype(np.uint32))


def test_nibble_transpose_exact_and_self_inverse():
    """Against the numpy reference on random words, and nibble_transpose o nibble_transpose
    = identity, over shapes that cover both the checkpoint's down layout and its transpose."""
    mx.random.seed(0)
    for shape in ((5, 24, 9), (2688, 232), (1856, 336), (3712, 336), (3, 2688, 232)):
        w = mx.random.randint(0, 2**32, shape, dtype=mx.uint32)
        t = nibble_transpose(w)
        mx.eval(t)
        assert t.shape == (*shape[:-2], shape[-1] * 8, shape[-2] // 8), t.shape
        assert mx.array_equal(t, nibble_transpose_np(w)), f"transpose wrong at {shape}"
        back = nibble_transpose(t)
        assert back.shape == w.shape
        assert mx.array_equal(back, w), f"not self-inverse at {shape}"


def test_to_neuron_major_is_the_same_bytes(model_dir):
    """The permutation moves the checkpoint's own bytes: dequantising the neuron-major
    layout must give exactly the same fp32 weights as dequantising the original."""
    W = load_moe_layer(model_dir)
    dn_w, dn_s, dn_b = (W[f"backbone.layers.{LAYER}.mixer.switch_mlp.fc2.{k}"] for k in ("weight", "scales", "biases"))
    E, D = dn_w.shape[0], dn_w.shape[1]
    pw, ps, pb = to_neuron_major(dn_w, dn_s, dn_b)
    assert pw.shape == (E, dn_w.shape[2] * 8, D // 8)
    assert ps.shape == (E, dn_w.shape[2] * 8 // GS, D)
    ref = mx.dequantize(dn_w, dn_s, dn_b, group_size=GS, bits=BITS)
    # rebuild W from the transposed words: word w of row i packs W[8w..8w+7, i]
    pw3 = nibble_transpose(pw)                            # back to [E, D, F/8]
    out = mx.dequantize(pw3, mx.swapaxes(ps, -1, -2), mx.swapaxes(pb, -1, -2), group_size=GS, bits=BITS)
    assert mx.array_equal(out.astype(mx.float32), ref.astype(mx.float32))


@pytest.fixture(scope="module")
def layer_weights(model_dir):
    return load_moe_layer(model_dir)


@pytest.fixture(scope="module")
def decode_case(model_dir, wired, layer_weights):
    """One MoE layer's real weights, with a neuron-major layout and random inputs that have
    realistic sparsity (random x through the real 4-bit up projections, then ReLU2)."""
    p = f"backbone.layers.{LAYER}.mixer"
    up_w, up_s, up_b = (layer_weights[f"{p}.switch_mlp.fc1.{k}"] for k in ("weight", "scales", "biases"))
    dn_w, dn_s, dn_b = (layer_weights[f"{p}.switch_mlp.fc2.{k}"] for k in ("weight", "scales", "biases"))
    su_w, su_s, su_b = (layer_weights[f"{p}.shared_experts.up_proj.{k}"] for k in ("weight", "scales", "biases"))
    sd_w, sd_s, sd_b = (layer_weights[f"{p}.shared_experts.down_proj.{k}"] for k in ("weight", "scales", "biases"))
    E, F, D = up_w.shape[0], up_w.shape[1], dn_w.shape[1]
    FS = su_w.shape[0]
    pw, ps, pb = to_neuron_major(dn_w, dn_s, dn_b)
    spw, sps, spb = to_neuron_major(sd_w, sd_s, sd_b)
    mx.eval(pw, ps, pb, spw, sps, spb)
    print(f"  layer {LAYER}: E={E} F={F} D={D} FS={FS}")
    return dict(up=(up_w, up_s, up_b), dn=(dn_w, dn_s, dn_b), su=(su_w, su_s, su_b), sd=(sd_w, sd_s, sd_b),
                pw=(pw, ps, pb), spw=(spw, sps, spb), E=E, F=F, D=D, FS=FS)


def make_inputs(case, B, seed=0):
    """Random x, real up projections, ReLU2: h with the sparsity the kernel exploits."""
    mx.random.seed(seed)
    x = (mx.random.normal((B, case["D"])) * 0.5).astype(mx.bfloat16)
    inds = mx.stack([mx.random.permutation(case["E"])[:TOPK] for _ in range(B)]).astype(mx.int32)
    scores = mx.random.uniform(shape=(B, TOPK)).astype(mx.float32)
    up_w, up_s, up_b = case["up"]
    su_w, su_s, su_b = case["su"]
    u = mx.gather_qmm(x[:, None, None, :], up_w, up_s, up_b, rhs_indices=inds, transpose=True,
                      group_size=GS, bits=BITS)
    h = mx.square(nn.relu(u))                                     # [B, TOPK, 1, F]
    hs = mx.square(nn.relu(mx.quantized_matmul(x, su_w, su_s, su_b, transpose=True, group_size=GS, bits=BITS)))
    hcat = mx.concatenate([h.reshape(B, TOPK * case["F"]), hs], axis=1)
    gains = mx.concatenate([scores, mx.ones((B, 1))], axis=1)
    mx.eval(hcat, inds, gains, h, hs, scores)
    return x, inds, scores, h, hs, hcat, gains


def reference(case, h, hs, inds, scores):
    """The fp32 reference: dequantise the down projections and form the products in fp32."""
    B = h.shape[0]
    F, D, FS = case["F"], case["D"], case["FS"]
    hf = h.astype(mx.float32).squeeze(-2)                         # [B, TOPK, F]
    dn_w, dn_s, dn_b = case["dn"]
    sd_w, sd_s, sd_b = case["sd"]
    Wd = mx.dequantize(dn_w, dn_s, dn_b, group_size=GS, bits=BITS).astype(mx.float32)   # [E, D, F]
    Wsd = mx.dequantize(sd_w, sd_s, sd_b, group_size=GS, bits=BITS).astype(mx.float32)
    ref = mx.zeros((B, D))
    for b in range(B):
        for k in range(TOPK):
            ref[b] = ref[b] + scores[b, k] * (Wd[inds[b, k]] @ hf[b, k])
        ref[b] = ref[b] + Wsd @ hs[b].astype(mx.float32)
    return ref


def mlx_reference(case, h, hs, inds, scores):
    """What MLX itself computes on the checkpoint layout (bf16 gather_qmm)."""
    dn_w, dn_s, dn_b = case["dn"]
    sd_w, sd_s, sd_b = case["sd"]
    y = mx.gather_qmm(h, dn_w, dn_s, dn_b, rhs_indices=inds, transpose=True, group_size=GS, bits=BITS)
    return (y.squeeze(-2) * scores[..., None].astype(y.dtype)).sum(-2) \
        + mx.quantized_matmul(hs, sd_w, sd_s, sd_b, transpose=True, group_size=GS, bits=BITS)


# slice / unroll / output-chunk settings that must all give the same answer
SWEEP = ((64, 2, 1), (64, 1, 3), (928, 2, 1), (928, 2, 3), (1856, 2, 3), (1856, 4, 2), (32, 2, 1))


@pytest.mark.parametrize("B", [1, 4])
def test_sparse_down_matches_fp32_reference(decode_case, B):
    """The kernel against the fp32 dequantised reference for several slice / unroll / chunk
    settings; also the error against MLX's own bf16 gather_qmm (reported, not asserted:
    it is the bf16 rounding of the same products)."""
    case = decode_case
    x, inds, scores, h, hs, hcat, gains = make_inputs(case, B)
    ref32 = reference(case, h, hs, inds, scores)
    ref_mlx = mlx_reference(case, h, hs, inds, scores)
    mx.eval(ref32, ref_mlx)
    live = float((hcat != 0).mean().item())
    worst32 = 0.0
    for sl, unr, nds in SWEEP:
        out = sparse_down(hcat, inds, gains, *case["pw"], *case["spw"],
                          F=case["F"], FS=case["FS"], D=case["D"], topk=TOPK,
                          sl=sl, unr=unr, nds=nds, out_dtype=mx.float32)
        mx.eval(out)
        r32 = ref32.astype(mx.float32)
        e32 = float((out - r32).abs().max() / r32.abs().max())
        rml = ref_mlx.astype(mx.float32)
        e_mlx = float((out - rml).abs().max() / rml.abs().max())
        worst32 = max(worst32, e32)
        print(f"  B={B} live {100 * live:.1f}%  sl={sl} unr={unr} nds={nds}: "
              f"max rel err vs fp32 dequant {e32:.2e}, vs MLX bf16 {e_mlx:.2e}")
        assert e32 < 2e-3, (sl, unr, nds, e32)
    # the same products in MLX's own bf16 order, for scale
    e_mlx_ref = float((mlx_reference(case, h, hs, inds, scores).astype(mx.float32) - ref32).abs().max() / ref32.abs().max())
    print(f"  B={B}: worst kernel err {worst32:.2e} (bound 2e-3); MLX bf16 gather_qmm err {e_mlx_ref:.2e}")


def main():
    import sys
    sys.exit(__import__("pytest").main([__file__, "-v", "-s"]))


if __name__ == "__main__":
    main()
