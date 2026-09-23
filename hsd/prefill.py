"""Sparse MoE down projection for prefill, on the matrix units, on the SAME neuron-major
layout the decode kernel uses (hsd.layout / hsd.decode).

The decode kernel is one threadgroup per (token, slice): each live weight row is loaded
once per token, which is the right trade at batch 1 but loses to a dense GEMM at prefill
lengths, where MLX's dense quantised GEMM runs on the GPU matrix units. But the neuron-major
layout is transposed relative to what that dense GEMM consumes, so without this kernel the
model would either have to keep both layouts resident (+8.4 GB) or transpose the weights for
every prompt.

So prefill runs its own GEMM on the matrix units and spends the sparsity the other way round:
the (token, expert-slot) pairs are sorted by expert and cut into tiles of TT slots; per tile
the UNION of the neurons live in any of its slots is the reduction dimension of a small
dense GEMM, h[TT x U] @ W[U x D], computed with Metal's mpp::tensor_ops::matmul2d over
K-chunks of KC rows. Rows dead for the whole tile are never read or multiplied; the FLOPs
are the union fraction of the dense GEMM.

Three small kernels around the GEMM, all on the GPU (no host round trip in the middle of a
prefill step):

    build_tiles   the tile order as MLX ops with a FIXED tile count, so nothing has to be
                  read back; tiles the routing does not fill carry expert -1 and are skipped
    _k_union      per tile, the union of live neurons, compacted with ballots and a scan
    _k_gather     each slot's h at its tile's union rows, as half, the A tile of the GEMM

The B tile is staged per K-chunk: a live row's 8 nibbles dequantised with its 64-group's
bf16 scale and bias (one uint4 each, contiguous in the neuron-major layout) into half.
The loads of chunk k+1 are issued into registers before the matrix-unit call for chunk k
(software pipeline); a barrier before restaging makes that safe, because the cooperative
matmul has every simdgroup still reading the previous tiles when an individual thread
returns from it.

The epilogue writes one row per (token, partial) with the routing gain applied, token-major,
so the reduction over a token's topk+1 partials is a contiguous sum.
"""

import mlx.core as mx

NA_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

UNION_SRC = """
    // h [T, NR] (T), tile_e [NT] int32 (>= E: shared expert, -1: empty tile),
    // tile_slot [NT*TT] int32 (-1 pad) -> ulist [NT*UMAX] int32 (-1 pad, ascending), ucount [NT] int32.
    // One threadgroup per tile: each thread owns columns c = tid, tid+NTH, ..., ORs the tile's
    // TT slots at c, then a ballot plus a prefix scan compacts the live columns.
    constexpr int NR = TOPK * F + FS;
    const int tid = (int)thread_position_in_threadgroup.x, tile = (int)threadgroup_position_in_grid.y;
    const int lane = (int)thread_index_in_simdgroup, sgi = (int)simdgroup_index_in_threadgroup;
    constexpr int NSGU = NTH / 32;
    const int e = tile_e[tile];
    if (e < 0) {                                        // empty tile of the fixed-size order
        for (int i = tid; i < UMAX; i += NTH) ulist[(size_t)tile * UMAX + i] = -1;
        if (tid == 0) ucount[tile] = 0;
        return;
    }
    const bool shared = e >= E;
    const int FF = shared ? FS : F;
    threadgroup int hoff[TT];                           // row offset of slot m's h into h[], -1 pad
    threadgroup int sgsum[NSGU];
    threadgroup int base;
    if (tid < TT) {
        const int s = tile_slot[tile * TT + tid];
        hoff[tid] = (s < 0) ? -1 : (shared ? s * NR + TOPK * F : (s / TOPK) * NR + (s % TOPK) * F);
    }
    if (tid == 0) base = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device int* out = ulist + (size_t)tile * UMAX;
    for (int c0 = 0; c0 < UMAX; c0 += NTH) {
        const int c = c0 + tid;
        bool live = false;
        if (c < FF) for (int m = 0; m < TT; ++m) live = live || (hoff[m] >= 0 && h[hoff[m] + c] != T(0));
        const int v = live ? 1 : 0;
        const int ex = simd_prefix_exclusive_sum(v);
        const int tot = simd_sum(v);
        if (lane == 0) sgsum[sgi] = tot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int before = 0;
        for (int i = 0; i < sgi; ++i) before += sgsum[i];
        if (live) out[base + before + ex] = c;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) { int t = 0; for (int i = 0; i < NSGU; ++i) t += sgsum[i]; base += t; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = base + tid; i < UMAX; i += NTH) out[i] = -1;
    if (tid == 0) ucount[tile] = base;
"""

GATHER_SRC = """
    // h [T, NR] (T), tile_e [NT], tile_slot [NT*TT], ulist [NT*UMAX], ucount [NT]
    //   -> hc [NT*TT, UMAX] half: slot (tile, m)'s h at its tile's union rows.
    // One thread over 8 union rows of one slot; rows beyond the tile's live union are left
    // untouched (the GEMM never reads them).
    constexpr int NR = TOPK * F + FS;
    const int g = (int)thread_position_in_grid.x;           // over NT*TT*(UMAX/8)
    constexpr int G8 = UMAX / 8;
    const int u8 = (g % G8) * 8, row = g / G8;              // row = tile*TT + m
    if (row >= NT * TT) return;
    const int tile = row / TT, m = row % TT;
    if (u8 >= ((ucount[tile] + 7) / 8) * 8) return;
    const int s = tile_slot[row], e = tile_e[tile];
    const bool shared = e >= E;
    const int hoff = (s < 0) ? -1 : (shared ? s * NR + TOPK * F : (s / TOPK) * NR + (s % TOPK) * F);
    half v[8];
    for (int j = 0; j < 8; ++j) {
        const int r = ulist[(size_t)tile * UMAX + u8 + j];
        v[j] = (hoff >= 0 && r >= 0) ? (half)h[hoff + r] : half(0);
    }
    device half* dst = hc + (size_t)row * UMAX + u8;
    for (int j = 0; j < 8; ++j) dst[j] = v[j];
"""

GEMM_SRC = """
    // hc [NT*TT, UMAX] half, tile_e [NT] int32 (>= E: shared, -1: empty), tile_slot [NT*TT] int32 (-1 pad),
    // ulist [NT*UMAX] int32 (union rows, -1 pad, ascending), ucount [NT] int32, gains [T, TOPK+1] f32,
    // pw [E, F, D/8] u32, ps / pb [E, F/64, D] (ST, neuron-major), spw [FS, D/8] u32, sps / spb [FS/64, D]
    //   -> part [T*(TOPK+1), D] PT
    constexpr int GS  = 64;
    constexpr int NW  = D / 8;
    constexpr int WPB = NB / 8;                            // weight words per row inside this column block
    static_assert(D % NB == 0 && NB % 8 == 0 && TT % 8 == 0 && KC % 8 == 0 && UMAX % 8 == 0, "");
    const int tid  = (int)thread_position_in_threadgroup.x;
    // grid: y = column block, z = tile, so a tile's D/NB blocks are adjacent threadgroups and
    // share its staged h and weight rows through the cache
    const int tile = (int)threadgroup_position_in_grid.z;
    const int n0   = (int)threadgroup_position_in_grid.y * NB;
    const int e = tile_e[tile];
    if (e < 0) return;                                     // empty tile of the fixed-size order
    const bool shared = e >= E;
    const device uint32_t* wb = shared ? spw : pw + (size_t)max(e, 0) * F * NW;
    const device ST* sb = shared ? sps : ps + (size_t)e * (F / GS) * D;
    const device ST* bb = shared ? spb : pb + (size_t)e * (F / GS) * D;
    threadgroup half As[TT * KC];                          // A tile: the slots' h at the chunk's union rows
    threadgroup half Bs[KC * NB];                           // B tile: the rows' dequantised weights
    threadgroup int  rows[KC];                              // the chunk's union rows (-1 pad)
    constexpr auto desc = matmul2d_descriptor(TT, NB, KC, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(As, dextents<int32_t, 2>(KC, TT));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Bs, dextents<int32_t, 2>(NB, KC));
    auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
    const int U = ucount[tile];
    const device int* ul = ulist + (size_t)tile * UMAX;
    // Software pipeline: the loads of chunk k+1 are issued into registers BEFORE the matrix-unit
    // call for chunk k, so their latency overlaps the multiply. Each thread owns WPT weight
    // words and its share of the A rows per chunk.
    constexpr int WPT = (KC * WPB + NTH - 1) / NTH;
    uint  qreg[WPT]; uint4 sreg[WPT]; uint4 breg[WPT];
    uint4 areg[(TT * (KC / 8) + NTH - 1) / NTH];
    auto bf = [](uint bits) -> half { return half(as_type<float>(bits << 16)); };   // bf16 bits -> half
    auto load_chunk = [&](int u0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < KC) rows[tid] = (u0 + tid < U) ? ul[u0 + tid] : -1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int t = 0; t < WPT; ++t) {
            const int w = tid + t * NTH;
            qreg[t] = 0u; sreg[t] = uint4(0u); breg[t] = uint4(0u);
            if (w < KC * WPB) {
                const int k = w / WPB, c8 = (w % WPB) * 8, r = rows[k];
                if (r >= 0) {
                    qreg[t] = wb[(size_t)r * NW + (n0 + c8) / 8];                // 8 outputs of live row r
                    sreg[t] = *(const device uint4*)(sb + (size_t)(r / GS) * D + n0 + c8);   // its group's 8 scales
                    breg[t] = *(const device uint4*)(bb + (size_t)(r / GS) * D + n0 + c8);
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < (TT * (KC / 8) + NTH - 1) / NTH; ++t) {
            const int i = tid + t * NTH;
            if (i < TT * (KC / 8)) { const int m = i / (KC / 8), k8 = (i % (KC / 8)) * 8; areg[t] = *(const device uint4*)(hc + (size_t)(tile * TT + m) * UMAX + u0 + k8); }
        }
    };
    if (U > 0) load_chunk(0);
    for (int u0 = 0; u0 < U; u0 += KC) {
        // Restaging overwrites As/Bs, which every simdgroup is still reading when an
        // individual thread returns from the matmul: this barrier is what makes the
        // software pipeline safe.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int t = 0; t < (TT * (KC / 8) + NTH - 1) / NTH; ++t) {
            const int i = tid + t * NTH;
            if (i < TT * (KC / 8)) { const int m = i / (KC / 8), k8 = (i % (KC / 8)) * 8; *(threadgroup uint4*)(As + m * KC + k8) = areg[t]; }
        }
        #pragma unroll
        for (int t = 0; t < WPT; ++t) {
            const int w = tid + t * NTH;
            if (w < KC * WPB) {
                const int k = w / WPB, c8 = (w % WPB) * 8;
                const uint q = qreg[t]; const uint4 s4 = sreg[t], b4 = breg[t];
                const half4 q0 = half4(half(q & 0xFu), half((q >> 4) & 0xFu), half((q >> 8) & 0xFu), half((q >> 12) & 0xFu));
                const half4 q1 = half4(half((q >> 16) & 0xFu), half((q >> 20) & 0xFu), half((q >> 24) & 0xFu), half(q >> 28));
                const half4 s0 = half4(bf(s4.x & 0xFFFFu), bf(s4.x >> 16), bf(s4.y & 0xFFFFu), bf(s4.y >> 16));
                const half4 s1 = half4(bf(s4.z & 0xFFFFu), bf(s4.z >> 16), bf(s4.w & 0xFFFFu), bf(s4.w >> 16));
                const half4 b0 = half4(bf(b4.x & 0xFFFFu), bf(b4.x >> 16), bf(b4.y & 0xFFFFu), bf(b4.y >> 16));
                const half4 b1 = half4(bf(b4.z & 0xFFFFu), bf(b4.z >> 16), bf(b4.w & 0xFFFFu), bf(b4.w >> 16));
                *(threadgroup half4*)(Bs + k * NB + c8)     = q0 * s0 + b0;   // a dead row: q = s = b = 0 -> zeros
                *(threadgroup half4*)(Bs + k * NB + c8 + 4) = q1 * s1 + b1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (u0 + KC < U) load_chunk(u0 + KC);               // next chunk's loads in flight during the multiply
        op.run(tA, tB, cT);
    }
    // epilogue: routing gain per slot, token-major (row = token*(TOPK+1) + k, shared at k = TOPK),
    // so the reduction over a token's TOPK+1 partials is a contiguous sum
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) {
        const auto idx = cT.get_multidimensional_index(i);
        const int n = idx[0], m = idx[1];
        const int s = tile_slot[tile * TT + m];
        if (s >= 0) {
            const int gi = shared ? s * (TOPK + 1) + TOPK : (s / TOPK) * (TOPK + 1) + (s % TOPK);
            part[(size_t)gi * D + n0 + n] = (PT)(cT[i] * gains[gi]);
        }
    }
"""

_k_union = mx.fast.metal_kernel(
    name="hsd_prefill_union",
    input_names=["h", "tile_e", "tile_slot"],
    output_names=["ulist", "ucount"],
    source=UNION_SRC,
    header=NA_HEADER,
)
_k_gather = mx.fast.metal_kernel(
    name="hsd_prefill_gather",
    input_names=["h", "tile_e", "tile_slot", "ulist", "ucount"],
    output_names=["hc"],
    source=GATHER_SRC,
    header=NA_HEADER,
)
_k_gemm = mx.fast.metal_kernel(
    name="hsd_prefill_gemm",
    input_names=["hc", "tile_e", "tile_slot", "ulist", "ucount", "gains", "pw", "ps", "pb", "spw", "sps", "spb"],
    output_names=["part"],
    source=GEMM_SRC,
    header=NA_HEADER,
)


def build_tiles(sel, E, tt):
    """The tile order: slots (token, expert-slot) sorted by expert, tiles of tt slots of one
    expert, then the shared expert's tiles of tt tokens.

    MLX ops only, with a FIXED tile count (an upper bound that never depends on the routing),
    so nothing has to be read back to the host between the up projection and the down kernel;
    tiles beyond what the routing fills carry expert -1 and every kernel skips them.

    sel [T, topk] int32 -> (tile_e [NT] int32, tile_slot [NT*tt] int32) where a slot is
    t*topk + k for a routed expert and t for the shared one, -1 padding.
    """
    T, K = sel.shape
    flat = sel.reshape(-1).astype(mx.int32)
    order = mx.argsort(flat).astype(mx.int32)
    counts = mx.zeros((E,), dtype=mx.int32).at[flat].add(1)
    ntile = (counts + tt - 1) // tt
    tile_base = mx.cumsum(ntile) - ntile
    exp_start = mx.cumsum(counts) - counts
    total = ntile.sum()
    NT_routed = T * K // tt + E                          # upper bound on the routed tile count
    t = mx.arange(NT_routed, dtype=mx.int32)
    e_of_t = (tile_base[None, :] <= t[:, None]).sum(axis=1).astype(mx.int32) - 1
    tile_e_routed = mx.where(t < total, e_of_t, -1)
    j = mx.arange(T * K, dtype=mx.int32)
    e_j = flat[order]
    rank = j - exp_start[e_j]                              # the slot's index within its expert
    dst = (tile_base[e_j] + rank // tt) * tt + rank % tt
    tile_slot_routed = mx.full((NT_routed * tt,), -1, dtype=mx.int32).at[dst].add(order + 1)
    NT_shared = (T + tt - 1) // tt
    tile_e_shared = mx.full((NT_shared,), E, dtype=mx.int32)
    tile_slot_shared = mx.where(mx.arange(NT_shared * tt) < T, mx.arange(NT_shared * tt), -1).astype(mx.int32)
    return mx.concatenate([tile_e_routed, tile_e_shared]), mx.concatenate([tile_slot_routed, tile_slot_shared])


def build_union(h, tile_e, tile_slot, *, F, FS, topk, E, tt, kc=32, nth=256):
    """The union of live neurons per tile, one compaction kernel. UMAX is FS padded up to a
    multiple of kc for every tile (routed tiles are padded with dead columns)."""
    NT = tile_e.shape[0]
    UMAX = (FS + kc - 1) // kc * kc
    tmpl = [("T", h.dtype), ("F", F), ("FS", FS), ("TOPK", topk), ("E", E), ("TT", tt),
            ("UMAX", UMAX), ("NTH", nth)]
    ulist, ucount = _k_union(inputs=[h, tile_e, tile_slot], template=tmpl,
                             grid=(nth, NT, 1), threadgroup=(nth, 1, 1),
                             output_shapes=[(NT, UMAX), (NT,)], output_dtypes=[mx.int32, mx.int32])
    return ulist, ucount


def sparse_down_prefill(h, sel, gains, pw, ps, pb, spw, sps, spb, *, F, FS, D, topk, E,
                        tt=32, nb=64, kc=32, nsg=4, tiles=None, union=None, out_dtype=None):
    """The MoE down projection over a whole prompt chunk on the neuron-major layout.

    h [T, topk*F + FS], sel [T, topk] int32, gains [T, topk+1] f32 -> y [T, D]. The tile
    order and the union lists are built on the GPU when not given. tt: slots per tile;
    nb: output columns per block; kc: union rows per matrix-unit chunk; nsg: simdgroups
    per threadgroup. All measured defaults; the dense middle of the model prefers tt=64.
    """
    T = h.shape[0]
    tile_e, tile_slot = tiles if tiles is not None else build_tiles(sel, E, tt)
    ulist, ucount = union if union is not None else build_union(h, tile_e, tile_slot, F=F, FS=FS, topk=topk, E=E, tt=tt, kc=kc)
    NT = tile_e.shape[0]
    UMAX = ulist.shape[1]
    nth = 32 * nsg
    gt = [("T", h.dtype), ("F", F), ("FS", FS), ("TOPK", topk), ("E", E), ("TT", tt),
          ("UMAX", UMAX), ("NT", NT)]
    ng = NT * tt * (UMAX // 8)
    hc, = _k_gather(inputs=[h, tile_e, tile_slot, ulist.reshape(-1), ucount], template=gt,
                    grid=((ng + 255) // 256 * 256, 1, 1), threadgroup=(256, 1, 1),
                    output_shapes=[(NT * tt, UMAX)], output_dtypes=[mx.float16])
    tmpl = [("ST", ps.dtype), ("PT", mx.float32), ("D", D), ("F", F), ("FS", FS), ("TOPK", topk),
            ("E", E), ("TT", tt), ("NB", nb), ("KC", kc), ("UMAX", UMAX), ("NSG", nsg), ("NTH", nth)]
    part, = _k_gemm(inputs=[hc, tile_e, tile_slot, ulist.reshape(-1), ucount, gains, pw, ps, pb, spw, sps, spb],
                    template=tmpl, grid=(nth, D // nb, NT), threadgroup=(nth, 1, 1),
                    output_shapes=[(T * (topk + 1), D)], output_dtypes=[mx.float32])
    y = part.reshape(T, topk + 1, D).sum(axis=1)
    return y.astype(out_dtype or h.dtype)
