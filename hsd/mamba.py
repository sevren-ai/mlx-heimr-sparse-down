"""Multi-row Mamba-2 steps with rollback, for speculative decoding.

A Mamba-2 layer has no KV cache to trim: its state is a recurrent (conv state, ssm state)
pair, and a verify step that processes B rows advances both by B. A rollback to the
accepted prefix cannot restore them by slicing, so the verify step remembers its inputs
-- the padded conv input and the ssm inputs of the B rows, and the state they started
from -- and the rollback recomputes the state update over the first n_keep rows only.
Attention layers, whose caches can be trimmed, are handled by the caller.

This is the standard technique any speculative implementation needs for recurrent
layers; `ssm_update` itself is mlx-lm's.
"""
import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.ssm import ssm_update


def mamba_verify(block, x3, cache, records):
    """One Mamba-2 block on x3 [1, L, D] with the cache advanced by L rows, remembering in
    `records` what a later rollback to a shorter prefix needs. Same maths as the mixer's
    own __call__, unrolled so the pieces are kept.
    """
    m = block.mixer
    h = block.norm(x3)
    proj = m.in_proj(h)
    gate, conv_in, dt = mx.split(proj, [m.intermediate_size, m.intermediate_size + m.conv_dim], axis=-1)
    nk = m.conv_kernel_size - 1
    conv_state = cache[0] if cache[0] is not None else mx.zeros((1, nk, m.conv_dim), dtype=conv_in.dtype)
    padded = mx.concatenate([conv_state, conv_in], axis=1)       # the conv window crossing the boundary
    cache[0] = padded[:, -nk:]
    co = nn.silu(m.conv1d(padded))
    hs, Bm, Cm = mx.split(co, [m.intermediate_size, m.intermediate_size + m.n_groups * m.ssm_state_size], axis=-1)
    L = x3.shape[1]
    hs4 = hs.reshape(1, L, m.num_heads, m.head_dim)
    B4 = Bm.reshape(1, L, m.n_groups, m.ssm_state_size)
    C4 = Cm.reshape(1, L, m.n_groups, m.ssm_state_size)
    state0 = cache[1]
    y, st = ssm_update(hs4, m.A_log, B4, C4, m.D.astype(hs4.dtype), dt, m.dt_bias, state0,
                       m.time_step_limit, None)
    cache[1] = st
    records.append((cache, padded, hs4, B4, C4, dt, state0, m))
    y = m.norm(y.reshape(1, L, m.intermediate_size), gate)
    return x3 + m.out_proj(y)


def mamba_rollback(records, n_keep):
    """After mamba_verify on L rows, put every cache in `records` back at the state after
    the first n_keep rows: the conv window is a slice of the padded input, and the ssm
    state is recomputed from the recorded inputs over the kept prefix only."""
    for cache, padded, hs4, B4, C4, dt, state0, m in records:
        nk = m.conv_kernel_size - 1
        cache[0] = padded[:, n_keep:n_keep + nk]
        _, st = ssm_update(hs4[:, :n_keep], m.A_log, B4[:, :n_keep], C4[:, :n_keep], m.D.astype(hs4.dtype),
                           dt[:, :n_keep], m.dt_bias, state0, m.time_step_limit, None)
        cache[1] = st
