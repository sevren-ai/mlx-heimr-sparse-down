"""Sparse MoE down projection for decode, as a Metal kernel on the neuron-major layout.

The checkpoint's down projection is permuted once at load time (hsd.layout) so
that one neuron's weights are one contiguous row. At decode the expert activation
h is already materialised for the up projection, so a neuron whose ReLU2 output is
exactly zero is known before any down bytes are touched: the kernel compacts the
live rows with simd ballots and never reads a dead row, and reads a 64-neuron
group's scales and biases only when at least one neuron of the group is live.

One threadgroup = D/8/NDS threads (NDS chunks of the D outputs) x one slice of SL
neurons of one routed expert or of the shared expert. Each thread owns one weight
word, i.e. 8 of the D outputs, and walks the slice's live rows accumulating
    out[n] += h[i] * nibble_n(W[i, 8w+n])
per live 64-group, then applies the group's affine terms,
    out[n] += scale[n] * acc[n] + bias[n] * sum(h over the group's live rows).
SL must tile F and FS and be a multiple of 32 (1856 = 2^6 x 29 and 3712 admit
32, 64, 928, 1856). A 64-group that straddles two slices is finished by both,
each reading the group's scales once. Partial sums land in part[B, NS, D] (fp32)
and a small reduce kernel sums the slices -- few, long slices keep that partial
small, which is why the default slice is 928 neurons, not 64.

With the defaults (sl=928, nds=3) a token costs 16 slices x 3 = 48 threadgroups:
6 routed experts x 2 slices x 3, plus the shared expert's 4 slices x 3.

`loadsc=0` skips the scale and bias loads; it exists only so a benchmark can
measure what the group metadata traffic costs, and must not be used for real
inference.
"""
import mlx.core as mx

HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
"""

KERNEL_SRC = """
    // h [B, TOPK*F + FS] (T), sel [B, TOPK] int32, gains [B, TOPK+1] f32
    // pw [E, F, D/8] u32, ps / pb [E, F/GS, D] (ST, neuron-major), spw [FS, D/8] u32,
    // sps / spb [FS/GS, D] -> part [B, NS, D] f32
    constexpr int GS   = 64;
    constexpr int NW   = D / 8;                          // weight words per neuron row
    constexpr int MAXG = SL / 32 + 1;                    // groups a slice can touch (partial at both ends)
    constexpr int SPE  = F / SL;                         // slices per routed expert
    constexpr int NSR  = TOPK * SPE;                     // routed slices, then the shared expert's
    constexpr int NSS  = FS / SL;
    constexpr int NS   = NSR + NSS;
    constexpr int NR   = TOPK * F + FS;                  // columns of h
    constexpr int TGW  = NW / NDS;                       // threads per threadgroup
    static_assert(F % SL == 0 && FS % SL == 0 && SL % 32 == 0 && NW % NDS == 0,
                  "slice must tile the experts, 32 rows at a time");
    const int w    = (int)thread_position_in_grid.x;     // word index over the whole D
    const int tid  = (int)thread_position_in_threadgroup.x;
    const int s    = (int)threadgroup_position_in_grid.y;
    const int bi   = (int)threadgroup_position_in_grid.z;
    const int lane = (int)thread_index_in_simdgroup;
    const int sgi  = (int)simdgroup_index_in_threadgroup;
    if (s >= NS || w >= NW) return;

    const device uint32_t* wb; const device ST* sb; const device ST* bb; const device T* hsrc;
    float gain; int r0;
    if (s < NSR) {                                      // a slice of routed expert slot `ei`
        const int ei = s / SPE; r0 = (s % SPE) * SL;
        const int e = sel[bi * TOPK + ei];
        wb   = pw + ((size_t)e * F + r0) * NW;
        sb   = ps + (size_t)e * (F / GS) * D + 8 * w;    // groups are addressed by their global index (grow)
        bb   = pb + (size_t)e * (F / GS) * D + 8 * w;
        hsrc = h + (size_t)bi * NR + ei * F + r0;
        gain = gains[bi * (TOPK + 1) + ei];
    } else {                                            // a slice of the shared expert
        r0 = (s - NSR) * SL;
        wb   = spw + (size_t)r0 * NW;
        sb   = sps + 8 * w;
        bb   = spb + 8 * w;
        hsrc = h + (size_t)bi * NR + TOPK * F + r0;
        gain = gains[bi * (TOPK + 1) + TOPK];
    }

    threadgroup float hs[SL];                           // the slice's slice of h
    threadgroup short live[SL];                         // its live rows, compacted
    threadgroup short gstart[MAXG + 1];                 // live-row prefix at each group boundary
    threadgroup short grow[MAXG];                       // the group's global index
    threadgroup short ngrp_tg;
    for (int i = tid; i < SL; i += TGW) hs[i] = float(hsrc[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi == 0) {                                     // one simdgroup compacts 32 rows at a time
        int base = 0, ng = 0;
        for (int k = 0; k < SL; k += 32) {
            if (k == 0 || ((r0 + k) % GS) == 0) {       // a 64-group starts here (or the slice starts inside one)
                if (lane == 0) { gstart[ng] = (short)base; grow[ng] = (short)((r0 + k) / GS); }
                ++ng;
            }
            const int i = k + lane;
            const bool nz = hs[i] != 0.0f;
            const uint m = (uint)((simd_vote::vote_t)simd_ballot(nz));
            const int prefix = popcount(m & ((1u << (uint)lane) - 1u));
            if (nz) live[base + prefix] = (short)i;
            base += popcount(m);
        }
        if (lane == 0) { gstart[ng] = (short)base; ngrp_tg = (short)ng; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int ngrp = ngrp_tg;

    float out[8];                                       // this thread's 8 outputs
    for (int n = 0; n < 8; ++n) out[n] = 0.0f;
    for (int g = 0; g < ngrp; ++g) {
        const int j0 = gstart[g], j1 = gstart[g + 1];
        if (j0 == j1) continue;                         // whole 64-neuron group dead: no scales, no biases
        float acc[8], hsum = 0.0f;
        for (int n = 0; n < 8; ++n) acc[n] = 0.0f;
        for (int j = j0; j < j1; j += UNR) {             // UNR live rows in flight per iteration
            uint32_t q[UNR]; float hv[UNR];
            for (int u = 0; u < UNR; ++u) {
                const bool valid = j + u < j1;
                const int i = valid ? (int)live[j + u] : (int)live[j];
                q[u]  = wb[(size_t)i * NW + w];          // the live row's word: contiguous, one load
                hv[u] = valid ? hs[i] : 0.0f;
            }
            for (int u = 0; u < UNR; ++u) {
                hsum += hv[u];                           // for the biases: b * sum(h) over the group
                for (int n = 0; n < 8; ++n) acc[n] += hv[u] * float((q[u] >> (4 * n)) & 0xFu);
            }
        }
        if (LOADSC) {                                   // the group's affine terms, 8 outputs each
            const device ST* sg = sb + (size_t)grow[g] * D;
            const device ST* bg = bb + (size_t)grow[g] * D;
            for (int n = 0; n < 8; ++n) out[n] += float(sg[n]) * acc[n] + float(bg[n]) * hsum;
        } else {                                        // benchmarks only: isolate the nibble traffic
            for (int n = 0; n < 8; ++n) out[n] += acc[n] + hsum;
        }
    }
    device float* outp = part + ((size_t)bi * NS + s) * D + 8 * w;
    for (int n = 0; n < 8; ++n) outp[n] = gain * out[n];
"""

REDUCE_SRC = """
    const int o = (int)thread_position_in_grid.x, bi = (int)thread_position_in_grid.y;
    if (o >= D) return;
    const device float* p = part + (size_t)bi * NS * D + o;
    float t = 0.0f;
    for (int s = 0; s < NS; ++s) t += p[(size_t)s * D];
    y[(size_t)bi * D + o] = T(t);
"""

_k_part = mx.fast.metal_kernel(
    name="hsd_sparse_down_decode",
    input_names=["h", "sel", "gains", "pw", "ps", "pb", "spw", "sps", "spb"],
    output_names=["part"],
    source=KERNEL_SRC,
    header=HEADER,
    ensure_row_contiguous=True,
)
_k_reduce = mx.fast.metal_kernel(
    name="hsd_sparse_down_decode_reduce",
    input_names=["part"],
    output_names=["y"],
    source=REDUCE_SRC,
    header=HEADER,
    ensure_row_contiguous=True,
)


def sparse_down(h, sel, gains, pw, ps, pb, spw, sps, spb, *, F, FS, D, topk,
                sl=928, unr=2, nds=3, out_dtype=None, loadsc=1):
    """Down projection over the live rows only.

    h [B, topk*F + FS] (bf16/f16/f32), sel [B, topk] int32 (the routed experts),
    gains [B, topk+1] f32 (routing scores, then 1 for the shared expert) -> y [B, D].

    sl: neurons per slice (a multiple of 32 that tiles F and FS; 928 measured best).
    unr: live rows in flight per loop iteration. nds: how many chunks the D outputs
    are split into (threadgroups per slice).
    """
    B = h.shape[0]
    NS = topk * (F // sl) + FS // sl
    tmpl = [("T", h.dtype), ("ST", ps.dtype), ("D", D), ("F", F), ("FS", FS),
            ("TOPK", topk), ("SL", sl), ("UNR", unr), ("NDS", nds), ("LOADSC", 1 if loadsc else 0)]
    part, = _k_part(inputs=[h, sel, gains, pw, ps, pb, spw, sps, spb], template=tmpl,
                    grid=(D // 8, NS, B), threadgroup=(D // 8 // nds, 1, 1),
                    output_shapes=[(B, NS, D)], output_dtypes=[mx.float32])
    od = out_dtype or h.dtype
    y, = _k_reduce(inputs=[part], template=[("T", od), ("D", D), ("NS", NS)],
                   grid=(D, B, 1), threadgroup=(min(D, 256), 1, 1),
                   output_shapes=[(B, D)], output_dtypes=[od])
    return y
