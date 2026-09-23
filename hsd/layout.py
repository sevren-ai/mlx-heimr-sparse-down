"""The neuron-major permutation of the checkpoint's quantised down projections.

mlx-lm stores a quantised down projection as W[out=D, in=F]:

    words    [.., D, F/8]  uint32   8 nibbles of one output row per word, low nibble first
    scales   [.., D, F/64] bf16     groups along the reduction axis
    biases   [.., D, F/64] bf16

In that layout a neuron (an input column of W) whose activation is exactly zero
still occupies half a byte in every one of the D rows, so a dead neuron cannot be
skipped without first gathering its bytes from D far-apart words.

Stored the other way round -- reduction-major, or neuron-major as this repository
calls it --

    pw  [.., F, D/8]  uint32   row i = neuron i, word w packs W[8w .. 8w+7, i]
    ps  [.., F/64, D] bf16
    pb  [.., F/64, D] bf16

one neuron's weights are one contiguous D/2-byte row that is never read when the
neuron is dead, and the 2 * D * 2 bytes of scales and biases of a 64-neuron group
are read only when at least one neuron of the group is live. ReLU2, the expert
activation of this model, leaves most neurons at exactly zero for any given
token, which is what makes the layout worth having.

The permutation is exact and self-inverse in its nibble part: no value is
re-quantised, only moved, so the dequantised products are the ones MLX itself
would form and a result differs from `mx.gather_qmm` only by fp32 summation order.
"""

import mlx.core as mx


def nibble_transpose(w):
    """[.., R, C/8] uint32 (8 nibbles of a row per word, low nibble first) -> [.., C, R/8]
    uint32 with the same convention: out[.., c, wo] packs W[8*wo + n, c] for n = 0..7.

    Exact and self-inverse. Pure MLX ops: every nibble is expanded to a uint32 on
    the way (that is what the shifts and the sum do; the sum of nibbles shifted
    into disjoint positions is their OR).
    """
    *lead, R, CW = w.shape
    C = CW * 8
    shifts = mx.array([4 * n for n in range(8)], dtype=mx.uint32)
    nib = (w[..., None] >> shifts) & mx.array(0xF, dtype=mx.uint32)       # [.., R, CW, 8]
    nib = nib.reshape(*lead, R, C)                                        # [.., R, C]  W[r, c]
    nib = mx.swapaxes(nib, -1, -2)                                       # [.., C, R]
    nib = nib.reshape(*lead, C, R // 8, 8)
    out = (nib << shifts).sum(axis=-1).astype(mx.uint32)
    return out


def to_neuron_major(w, s, b, chunk=16):
    """A down projection as mlx-lm stores it (w [.., D, F/8] u32, s / b [.., D, F/64])
    -> (pw [.., F, D/8] u32, ps [.., F/64, D], pb [.., F/64, D]).

    For the routed experts (a leading axis of 128) the nibble transpose is done
    `chunk` experts at a time: each intermediate step expands every nibble to a
    uint32, so the whole-expert-at-once transient would be ~4x the final size.
    """
    if w.ndim == 3 and w.shape[0] > chunk:
        parts = []
        for i in range(0, w.shape[0], chunk):
            pw = mx.contiguous(nibble_transpose(w[i:i + chunk]))
            mx.eval(pw)
            parts.append(pw)
        pw = mx.concatenate(parts, axis=0)
        mx.eval(pw)
        del parts
    else:
        pw = mx.contiguous(nibble_transpose(w))
    ps = mx.contiguous(mx.swapaxes(s, -1, -2))
    pb = mx.contiguous(mx.swapaxes(b, -1, -2))
    return pw, ps, pb
