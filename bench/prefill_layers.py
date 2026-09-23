"""Per-layer prefill benchmark: the matrix-unit sparse down kernel against MLX's own dense
down on the stock layout, per MoE layer, on a 2048-token chunk of real activations (Moby
Dick through the full model). The sparse timing includes everything the e2e path pays:
the tile order, the union compaction and the gather, all on the GPU.

Also, the cost of the alternatives that motivate having a prefill kernel on the same
layout as decode at all:
  (a) the time to build the neuron-major layout for all 23 layers -- what "transpose the
      weights for every prompt" would cost per prompt
  (b) the memory of keeping both layouts resident against one

With --story (default layer 10, comma list for more) each chosen layer also exports a
prefill storyboard JSON: how the router split the chunk over the 128 experts, the live
masks of one typical expert's tokens (base64-packed bits), and the union shares that drive
the kernel's tile GEMM.

  python -m bench.prefill_layers                       # all 23 MoE layers
  python -m bench.prefill_layers --layers 1,10,22,51
"""
import argparse
import json
import os
import time

import mlx.core as mx
import numpy as np

from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

from hsd.model import PREFILL_KW, NemotronModel, relu2
from hsd.prefill import build_tiles, build_union, sparse_down_prefill
from hsd.resolve import DEFAULT_MODEL_ID

from bench.common import (append_json, capture_moe_inputs, moby_ids, pack_masks_b64,
                          record_meta)

NTOK = 2048
REPS = 3          # timed queues, best of
CALLS = 3         # calls queued lazily per queue


def timeit(fn, reps=REPS, calls=CALLS, warm=2):
    for _ in range(warm):
        mx.eval(fn())
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        outs = [fn() for _ in range(calls)]
        mx.eval(*outs)
        best = min(best, (time.perf_counter() - t0) / calls)
    return 1e3 * best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_ID, help="local path or HF repo id")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--story", default="10", help="comma list of layers to export a prefill storyboard JSON for")
    ap.add_argument("--out", default="results/prefill_layers.json")
    a = ap.parse_args()
    story = {int(s) for s in a.story.split(",") if s}
    dev = mx.device_info()
    print(f"{dev.get('device_name')}, mlx {mx.__version__}", flush=True)

    # ---- (a) the layout build for all 23 layers, and (b) both layouts against one
    model = NemotronModel(a.model, mode="dense", verbose=False, single_layout=False)   # nothing converted yet
    active_one = mx.get_active_memory() / 1e9
    t0 = time.time()
    for l in model.moe_layers:
        model.blocks[l].build_layout(drop_stock=False)                         # keep the stock layout
    build_s = time.time() - t0
    active_both = mx.get_active_memory() / 1e9
    down_bytes = sum(b.nbytes for l in model.moe_layers
                     for b in (model.blocks[l].pw, model.blocks[l].ps, model.blocks[l].pb,
                               model.blocks[l].spw, model.blocks[l].sps, model.blocks[l].spb))
    print(f"one layout {active_one:.2f} GB active | neuron-major build for {len(model.moe_layers)} layers "
          f"{build_s:.1f}s | both layouts {active_both:.2f} GB active (down projections {down_bytes / 1e9:.2f} GB)",
          flush=True)

    model.set_mode("sparse")      # layouts already built; stock stays (single_layout was False)
    caps = capture_moe_inputs(model, moby_ids(model, NTOK))
    print(f"captured {len(caps)} MoE layers on a {NTOK}-token pass", flush=True)

    moe_layers = model.moe_layers if a.layers == "all" else [int(s) for s in a.layers.split(",")]
    res = {}
    print(f"\n{NTOK}-token chunk, real activations; ms per layer (best of {REPS} x {CALLS} queued):")
    print("| layer | tt | live rows | union of tiles | down dense (stock) | down sparse (matrix-unit) | ratio | "
          "rel err | block stock | block sparse |")
    print("|---|" + "---|" * 10)

    for l in moe_layers:
        blk = model.blocks[l]
        b = blk.block.mixer
        x = caps[l]
        xn = blk.block.norm(x)
        inds, scores = b.gate(xn)
        xs, idx, inv = _gather_sort(xn[:, None, None, :], inds)
        hsort = relu2(b.switch_mlp.fc1(xs, idx, sorted_indices=True))
        h = _scatter_unsort(hsort, inv, inds.shape).squeeze(-2)
        hs = relu2(b.shared_experts.up_proj(xn))
        hcat, sel, gains = blk.pack(inds, scores, h, hs)
        mx.eval(hsort, h, hs, hcat, sel, gains, idx, inv)
        live = float((hcat != 0).mean().item())
        kw = dict(blk.prefill_kw)

        def down_stock():
            y = _scatter_unsort(b.switch_mlp.fc2(hsort, idx, sorted_indices=True), inv, inds.shape).squeeze(-2)
            return (y * scores[..., None].astype(y.dtype)).sum(-2) + b.shared_experts.down_proj(hs)

        def down_sparse():
            tiles = build_tiles(sel, blk.E, kw["tt"])
            union = build_union(hcat, tiles[0], tiles[1], F=blk.F, FS=blk.FS, topk=blk.topk,
                                E=blk.E, tt=kw["tt"], kc=kw["kc"])
            return sparse_down_prefill(hcat, sel, gains, *blk.w_routed, *blk.w_shared,
                                       F=blk.F, FS=blk.FS, D=blk.D, topk=blk.topk, E=blk.E,
                                       tiles=tiles, union=union, **kw)

        tiles = build_tiles(sel, blk.E, kw["tt"])
        union = build_union(hcat, tiles[0], tiles[1], F=blk.F, FS=blk.FS, topk=blk.topk,
                            E=blk.E, tt=kw["tt"], kc=kw["kc"])
        mx.eval(*union)
        valid = tiles[0] >= 0
        width = mx.where(tiles[0] >= blk.E, blk.FS, blk.F)
        ufrac = float((union[1] * valid).sum() / (width * valid).sum())
        ref = down_stock().astype(mx.float32)
        out = down_sparse().astype(mx.float32)
        mx.eval(ref, out)
        err = float((out - ref).abs().max() / ref.abs().max())
        t_stock, t_sparse = timeit(down_stock), timeit(down_sparse)
        t_blk_stock = timeit(lambda: blk.block(x[None])[0])
        t_blk_sparse = timeit(lambda: blk.prefill_sparse(x))
        r = dict(layer=l, tile_slots=kw["tt"], live=live, union_frac=ufrac,
                 down_dense_ms=t_stock, down_sparse_ms=t_sparse, ratio=t_stock / t_sparse,
                 rel_err=err, block_stock_ms=t_blk_stock, block_sparse_ms=t_blk_sparse)
        res[str(l)] = r
        print(f"| {l} | {kw['tt']} | {100 * live:.1f}% | {100 * ufrac:.0f}% | {t_stock:.1f} | {t_sparse:.1f} | "
              f"{t_stock / t_sparse:.2f}x | {err:.1e} | {t_blk_stock:.1f} | {t_blk_sparse:.1f} |", flush=True)

        # ---- the prefill storyboard: the router split, one typical expert's live masks, the union shares
        if l in story:
            T = h.shape[0]
            h_np = np.asarray(h.astype(mx.float32))            # [T, topk, F] after ReLU2, token order
            hs_np = np.asarray(hs.astype(mx.float32))         # [T, FS]
            sel_np = np.asarray(sel).astype(np.int64)          # [T, topk]
            counts = np.bincount(sel_np.reshape(-1), minlength=blk.E)     # tokens per routed expert
            avg = T * blk.topk / blk.E
            e = int(np.argmin(np.abs(counts.astype(float) - avg)))        # the most average-loaded expert
            occ = sel_np == e
            toks = np.nonzero(occ.any(axis=1))[0]                         # its tokens, chunk order
            ks = np.argmax(occ[toks], axis=1)                              # the slot each one routed in
            masks = h_np[toks, ks] != 0                                    # [n, F] live masks, chunk order
            tt = kw["tt"]
            per_tile = [float(masks[i:i + tt].any(axis=0).mean()) for i in range(0, len(toks), tt)]
            story_out = dict(
                layer=l, chunk_tokens=T, tile_slots=tt,
                D=blk.D, F=blk.F, FS=blk.FS, topk=blk.topk, n_experts=blk.E,
                expert_token_counts=counts.tolist(),
                follow=dict(
                    expert=e, tokens=int(counts[e]), average_tokens=avg,
                    mask_rows=int(len(toks)), mask_bytes_per_row=(blk.F + 7) // 8,
                    live_mask_b64=pack_masks_b64(masks),
                    mean_live_per_token=float(masks.mean()),
                    union_over_all_tokens=float(masks.any(axis=0).mean()),
                    union_share_per_tile=per_tile),
                shared_union_over_chunk=float((hs_np != 0).any(axis=0).mean()),
                kernel_union_share=ufrac,
            )
            path = os.path.join(os.path.dirname(a.out) or ".", f"storyboard_prefill_layer{l:02d}.json")
            json.dump(story_out, open(path, "w"), indent=1)
            print(f"   storyboard -> {path}: expert {e} carries {int(counts[e])} tokens "
                  f"(average {avg:.1f}), live {100 * story_out['follow']['mean_live_per_token']:.1f}% per token, "
                  f"union {100 * story_out['follow']['union_over_all_tokens']:.1f}% over all its tokens, "
                  f"per-tile union {100 * min(per_tile):.0f}-{100 * max(per_tile):.0f}%", flush=True)

    if len(res) == len(model.moe_layers):
        keys = ("down_dense_ms", "down_sparse_ms", "block_stock_ms", "block_sparse_ms")
        S = {k: sum(r[k] for r in res.values()) for k in keys}
        S["mean_live"] = float(np.mean([r["live"] for r in res.values()]))
        S["mean_union_frac"] = float(np.mean([r["union_frac"] for r in res.values()]))
        res["sum"] = S
        print(f"\nsum over {len(model.moe_layers)} MoE layers: down dense {S['down_dense_ms']:.0f} ms, "
              f"down sparse {S['down_sparse_ms']:.0f} ms, block stock {S['block_stock_ms']:.0f} ms, "
              f"block sparse {S['block_sparse_ms']:.0f} ms per {NTOK}-token chunk")

    # (b) again, now with the stock layout dropped at the end of the comparison
    for l in model.moe_layers:
        model.blocks[l].drop_stock()
    mx.clear_cache()
    active_after_drop = mx.get_active_memory() / 1e9
    print(f"stock layout dropped after the comparison: {active_after_drop:.2f} GB active", flush=True)

    res["layout"] = dict(build_seconds=build_s, layers=len(model.moe_layers),
                         active_one_layout_gb=active_one, active_both_layouts_gb=active_both,
                         active_one_layout_after_drop_gb=active_after_drop,
                         down_projection_gb=down_bytes / 1e9)
    res["meta"] = dict(record_meta(), model=a.model, prompt_tokens=NTOK,
                       prefill_kw=dict(PREFILL_KW), time=time.strftime("%Y-%m-%d %H:%M:%S"))
    append_json(a.out, res)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
