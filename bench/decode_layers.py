"""Per-layer decode benchmark on real activations: what one decode token costs in each
piece of every one of the 23 MoE layers, and what the down-projection row skip is worth.

Protocol: Moby Dick (2048 tokens) is run through the full model once, capturing each MoE
layer's input residual stream; 32 positions per layer are taken. Per layer, compiled
functions are timed with calls queued lazily over the 32 positions and evaluated once,
best of 5 repeats -- the way the decode loop actually runs, weights cold like a real step.

Measured per layer, per token:
  router       norm + gate + top-k selection
  up           the stock sorted up projections + ReLU2 (routed and shared)
  down dense   MLX's own gather_qmm down on the checkpoint layout
  down sparse  our kernel over the live rows on the neuron-major layout
  down sparse, dense h   the CONTROL: the identical kernel and launch, fed an h with no
               zeros in it (every element nudged to a small positive constant), which
               separates "a faster kernel" from "bytes not read"
  block dense / block sparse   the whole MoE block, both modes

Also per layer: the live fraction of neurons and of 64-neuron groups after ReLU2 (routed
and shared), the byte model (dense bytes vs bytes actually read), and the relative error
of the sparse result against MLX's dense down on the same inputs.

  python -m bench.decode_layers                          # all 23 MoE layers
  python -m bench.decode_layers --layers 1,10,22,51
  python -m bench.decode_layers --story 10               # also export the storyboard JSON
"""
import argparse
import json
import os
import time

import mlx.core as mx
import numpy as np

from hsd.model import DECODE_KW, NemotronModel, relu2
from hsd.decode import sparse_down
from hsd.resolve import DEFAULT_MODEL_ID

from bench.common import (GS, GRP_B, ROW_B, append_json, byte_model, capture_moe_inputs,
                          gpu_core_count, moby_ids, record_meta)

NTOK = 2048
NPOS = 32
LOOPS = 8                    # passes over the positions per timed queue
REPS = 5
CORE_COUNT = gpu_core_count()


def rot_ms(f, args_list, loops=LOOPS, reps=REPS, warm=2):
    """GPU ms per call, calls rotating over args_list (so consecutive calls touch different
    experts' weights, like a real step), loops passes queued lazily and evaluated once,
    best of `reps`."""
    n = loops * len(args_list)
    for _ in range(warm):
        mx.eval(f(*args_list[0]))
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        outs = [f(*args_list[i % len(args_list)]) for i in range(n)]
        mx.eval(*outs)
        best = min(best, (time.perf_counter() - t0) / n)
    return 1e3 * best


def q8(v):
    """|h| quantised to 8 bits with a square-root scale, so 0 stays exactly 0."""
    v = np.asarray(v, dtype=np.float64)
    m = v.max()
    return [int(round(255 * np.sqrt(x / m))) if m > 0 and x > 0 else 0 for x in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_ID, help="local path or HF repo id")
    ap.add_argument("--layers", default="all", help="comma list of MoE layer indices, or 'all'")
    ap.add_argument("--story", default="", help="comma list of layers to export a storyboard JSON for (default: none)")
    ap.add_argument("--out", default="results/decode_layers.json")
    a = ap.parse_args()
    story = {int(s) for s in a.story.split(",") if s}

    dev = mx.device_info()
    print(f"{dev.get('device_name')}, {CORE_COUNT} cores, mlx {mx.__version__}", flush=True)
    t0 = time.time()
    model = NemotronModel(a.model, mode="sparse", verbose=False, single_layout=False)   # both layouts: the bench compares them
    print(f"model loaded, both down layouts resident ({mx.get_active_memory() / 1e9:.1f} GB active)", flush=True)
    caps = capture_moe_inputs(model, moby_ids(model, NTOK))
    print(f"captured {len(caps)} MoE layers on a {NTOK}-token pass ({time.time() - t0:.0f}s)", flush=True)
    positions = list(range(0, NTOK, NTOK // NPOS))[:NPOS]

    moe_layers = model.moe_layers if a.layers == "all" else [int(s) for s in a.layers.split(",")]
    res = {}
    print(f"\nper decode token, {NPOS} real positions, queued ({LOOPS} passes, best of {REPS}):")
    print("| layer | live routed | live shared | live 64-grp | down bytes read | router | up | down dense | down sparse | "
          "same, dense h | block dense | block sparse | err |")
    print("|---|" + "---|" * 12)

    for l in moe_layers:
        blk = model.blocks[l]
        b = blk.block.mixer
        X = caps[l]
        args_block = [(X[p:p + 1],) for p in positions]
        # the real router / up / pack outputs at the 32 positions
        fronts, packs = [], []
        for p in positions:
            f = blk.front(X[p:p + 1])
            fronts.append(f)
            packs.append(blk.pack(*f))
        mx.eval(*[f[2] for f in fronts], *[p[0] for p in packs])

        # live statistics and the byte model (means over the positions)
        stats_r = [dict(rows=0.0, groups=0.0) for _ in range(blk.topk)]
        stats_s = dict(rows=0.0, groups=0.0)
        for f in fronts:
            h, hs = f[2], f[3]                       # [1, K, 1, F] and [1, FS]
            hs_ = hs.reshape(-1, GS)
            stats_s["rows"] += float((hs != 0).sum())
            stats_s["groups"] += float(hs_.any(-1).sum())
            for k in range(blk.topk):
                hk = h[0, k, 0]
                stats_r[k]["rows"] += float((hk != 0).sum())
                stats_r[k]["groups"] += float((hk.reshape(-1, GS)).any(-1).sum())
        n = len(positions)
        rows_r = sum(s["rows"] for s in stats_r) / n
        grp_r = sum(s["groups"] for s in stats_r) / n
        rows_s = stats_s["rows"] / n
        grp_s = stats_s["groups"] / n
        read_r, dense_r = byte_model(rows_r, grp_r, blk.topk, blk.F)
        read_s, dense_s = byte_model(rows_s, grp_s, 1, blk.FS)
        sparse_mb, dense_mb = (read_r + read_s) / 1e6, (dense_r + dense_s) / 1e6

        # correctness of the sparse result against MLX's own dense down, same inputs
        err = 0.0
        for f, p in zip(fronts, packs):
            h, hs, inds, scores = f[2], f[3], f[0], f[1]
            y = b.switch_mlp.fc2(h, inds)
            ref = (y.squeeze(-2) * scores[..., None].astype(y.dtype)).sum(-2) + b.shared_experts.down_proj(hs)
            out = sparse_down(p[0], p[1], p[2], *blk.w_routed, *blk.w_shared,
                              F=blk.F, FS=blk.FS, D=blk.D, topk=blk.topk, **DECODE_KW)
            ref, out = ref.astype(mx.float32), out.astype(mx.float32)
            mx.eval(ref, out)
            err = max(err, float((out - ref).abs().max() / ref.abs().max()))

        # the timed functions (compiled; the whole block both ways, and the pieces)
        f_router = mx.compile(lambda x: b.gate(blk.block.norm(x)))
        f_up = mx.compile(lambda x, inds: (relu2(mx.gather_qmm(
            blk.block.norm(x)[:, None, None, :], b.switch_mlp.fc1.weight, b.switch_mlp.fc1.scales,
            b.switch_mlp.fc1.biases, rhs_indices=inds, transpose=True,
            group_size=b.switch_mlp.fc1.group_size, bits=b.switch_mlp.fc1.bits)),
            relu2(b.shared_experts.up_proj(blk.block.norm(x)))))
        f_down_dense = mx.compile(lambda h, hs, inds, scores: (
            b.switch_mlp.fc2(h, inds).squeeze(-2) * scores[..., None].astype(h.dtype)).sum(-2)
            + b.shared_experts.down_proj(hs))
        f_down_sparse = mx.compile(lambda hcat, sel, gains: sparse_down(
            hcat, sel, gains, *blk.w_routed, *blk.w_shared, F=blk.F, FS=blk.FS, D=blk.D,
            topk=blk.topk, **DECODE_KW))
        f_block_dense = model.blocks[l].decode_fn("dense")
        f_block_sparse = model.blocks[l].decode_fn("sparse")

        args_router = [(X[p:p + 1],) for p in positions]
        args_up = [(X[p:p + 1], fronts[i][0]) for i, p in enumerate(positions)]
        args_down = [(fronts[i][2], fronts[i][3], fronts[i][0], fronts[i][1]) for i in range(n)]
        args_pack = [(packs[i][0], packs[i][1], packs[i][2]) for i in range(n)]
        # the control: the same kernel, the same launch, an h with nothing to skip
        dense_h = []
        for p in packs:
            h = mx.maximum(p[0], mx.array(1e-3, dtype=p[0].dtype))
            dense_h.append((h, p[1], p[2]))
        mx.eval(*[t[0] for t in dense_h])

        r = dict(layer=l,
                 live_routed_frac=rows_r / (blk.topk * blk.F), live_shared_frac=rows_s / blk.FS,
                 live_group_frac_routed=grp_r / (blk.topk * blk.F / GS),
                 live_group_frac_shared=grp_s / (blk.FS / GS),
                 live_rows_per_token=dict(routed=rows_r, shared=rows_s),
                 live_groups_per_token=dict(routed=grp_r, shared=grp_s),
                 dense_down_mb=dense_mb, sparse_down_mb=sparse_mb,
                 down_bytes_pct=100 * sparse_mb / dense_mb,
                 router_ms=rot_ms(f_router, args_router),
                 up_ms=rot_ms(f_up, args_up),
                 down_dense_ms=rot_ms(f_down_dense, args_down),
                 down_sparse_ms=rot_ms(f_down_sparse, args_pack),
                 down_sparse_dense_h_ms=rot_ms(f_down_sparse, dense_h),
                 block_dense_ms=rot_ms(f_block_dense, args_block),
                 block_sparse_ms=rot_ms(f_block_sparse, args_block),
                 rel_err_vs_mlx_dense=err)
        res[str(l)] = r
        print(f"| {l} | {100 * r['live_routed_frac']:.1f}% | {100 * r['live_shared_frac']:.1f}% | "
              f"{100 * (grp_r + grp_s) / (blk.topk * blk.F / GS + blk.FS / GS):.1f}% | "
              f"{r['down_bytes_pct']:.1f}% ({sparse_mb:.1f} of {dense_mb:.1f} MB) | "
              f"{r['router_ms']:.3f} | {r['up_ms']:.3f} | {r['down_dense_ms']:.3f} | {r['down_sparse_ms']:.3f} | "
              f"{r['down_sparse_dense_h_ms']:.3f} | {r['block_dense_ms']:.3f} | {r['block_sparse_ms']:.3f} | "
              f"{err:.1e} |", flush=True)

        # ---- the storyboard: the median-live position, masks and |h| per expert
        if l in story:
            live = [float((p[0] != 0).mean().item()) for p in packs]
            ti = int(np.argsort(live)[len(live) // 2])
            hcat = np.asarray(packs[ti][0].astype(mx.float32))[0]
            hs = hcat[blk.topk * blk.F:]
            hr = hcat[:blk.topk * blk.F].reshape(blk.topk, blk.F)
            sel_i = np.asarray(packs[ti][1])[0].tolist()
            scores = np.asarray(fronts[ti][1].astype(mx.float32))[0].tolist()
            sh_rows = float((hs != 0).sum())
            sh_grp = float((hs.reshape(-1, GS) != 0).any(-1).sum())
            sl, nds = blk.decode_kw["sl"], blk.decode_kw["nds"]     # the kernel's partition at this layer
            story_out = dict(
                layer=l, token_index=ti, prompt_position=positions[ti],
                D=blk.D, F=blk.F, FS=blk.FS, topk=blk.topk, n_experts=blk.E,
                selected_experts=sel_i, routing_scores=scores,
                shared=dict(live_pct=100 * float((hs != 0).mean()),
                            live_rows=sh_rows, live_groups=sh_grp,
                            dense_mb=(blk.FS * ROW_B + blk.FS / GS * GRP_B) / 1e6,
                            sparse_mb=(sh_rows * ROW_B + sh_grp * GRP_B) / 1e6,
                            h_q8=q8(hs)),
                routed=[dict(expert=int(sel_i[k]), live_pct=100 * float((hr[k] != 0).mean()), h_q8=q8(hr[k]))
                        for k in range(blk.topk)],
                byte_model=dict(row_bytes=ROW_B, group_bytes=GRP_B, group_size=GS),
                kernel_partition=dict(slice_neurons=sl, output_chunks=nds,
                                      slices_per_routed_expert=blk.F // sl,
                                      slices_per_shared_expert=blk.FS // sl,
                                      threadgroups_per_token=(blk.topk * (blk.F // sl) + blk.FS // sl) * nds),
                gpu_core_count=CORE_COUNT,
                timings_ms={k: v for k, v in r.items() if k.endswith("_ms")},
            )
            path = os.path.join(os.path.dirname(a.out) or ".", f"storyboard_layer{l:02d}.json")
            json.dump(story_out, open(path, "w"), indent=1)
            print(f"   storyboard -> {path}: token {ti}, shared live "
                  f"{story_out['shared']['live_pct']:.1f}% ({sh_rows:.0f} rows, {sh_grp:.0f} groups), "
                  f"reads {story_out['shared']['sparse_mb']:.2f} of {story_out['shared']['dense_mb']:.2f} MB", flush=True)

    if len(res) == len(model.moe_layers):
        keys = ("router_ms", "up_ms", "down_dense_ms", "down_sparse_ms", "down_sparse_dense_h_ms",
                "block_dense_ms", "block_sparse_ms")
        S = {k: sum(r[k] for r in res.values()) for k in keys}
        S["dense_down_mb"] = sum(r["dense_down_mb"] for r in res.values())
        S["sparse_down_mb"] = sum(r["sparse_down_mb"] for r in res.values())
        S["mean_down_bytes_pct"] = float(np.mean([r["down_bytes_pct"] for r in res.values()]))
        res["sum"] = S
        print(f"\nsum over {len(model.moe_layers)} MoE layers (ms per decode token): "
              + ", ".join(f"{k} {v:.3f}" for k, v in S.items() if k.endswith("_ms")))
        print(f"down bytes actually read: {S['mean_down_bytes_pct']:.1f}% of dense "
              f"({S['sparse_down_mb']:.0f} of {S['dense_down_mb']:.0f} MB per token)")

    res["layer_types"] = list(model.types)      # per layer index: mamba / attention / moe
    res["device"] = dev.get("device_name")
    res["meta"] = dict(record_meta(), core_count=CORE_COUNT, model=a.model,
                       prompt_tokens=NTOK, positions=NPOS, loops=LOOPS, reps=REPS,
                       decode_kw=dict(DECODE_KW), time=time.strftime("%Y-%m-%d %H:%M:%S"))
    append_json(a.out, res)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
