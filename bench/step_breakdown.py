"""The decode step broken down by block type, on a real ~4k context.

Moby Dick is prefilled once; then the residual streams captured at the first MoE layer's
positions are fed, one per pass, through all blocks of one kind in sequence -- the 23
Mamba-2 blocks chained with their caches, the 6 attention blocks chained, the 23 MoE
blocks chained in dense and in sparse mode, and lm_head + argmax once -- each timed as a
chained pass over the real streams, cold and serial like the corresponding slice of a real
step. Also the full 52-block step chained the same way, and a measured generation loop.

A pass count of 24 or more is needed: short queues ride the GPU's boost clocks and read
low (the full matrix at n=6 measured ~40% below n=24 on the same work), while a real
decode step runs at the sustained clocks that a converged queue reproduces.

The per-type chains measure each type IN ISOLATION; the block types interact in the full
step, so the dense-to-sparse difference shows up in the full step chain and the measured
loop, not necessarily in the isolated MoE chain. In isolation the sparse MoE chain reads
slightly slower than the dense one; why the sign inverts outside isolation is not measured
(the likely explanation is that the two small sparse-down kernels hide under the
neighbouring blocks' larger kernels in a full step, but that is a hypothesis). Both the
per-type chains and the anchors are reported.

  python -m bench.step_breakdown [--ctx 4096] [--gen 64] [--passes 24]
"""
import argparse
import time

import mlx.core as mx

from hsd.model import NemotronModel
from hsd.resolve import DEFAULT_MODEL_ID

from bench.common import append_json, capture_moe_inputs, chain_ms, moby_ids, record_meta

NSTREAMS = 24              # real residual streams, one per chained pass


def rotating(xs):
    state = {"i": 0}

    def next_x():
        x = xs[state["i"] % len(xs)]
        state["i"] += 1
        return x
    return next_x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_ID, help="local path or HF repo id")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--passes", type=int, default=24)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--streams", type=int, default=NSTREAMS)
    ap.add_argument("--out", default="results/step_breakdown.json")
    a = ap.parse_args()
    dev = mx.device_info()
    print(f"{dev.get('device_name')}, mlx {mx.__version__}", flush=True)

    model = NemotronModel(a.model, mode="dense", verbose=False, single_layout=False)
    ids = moby_ids(model, a.ctx)
    model._cache = model.make_cache()
    caps = capture_moe_inputs(model, ids, cache=model._cache)   # the real context fills the caches
    first_moe = model.moe_layers[0]
    xs = [caps[first_moe][p:p + 1] for p in range(0, caps[first_moe].shape[0], a.ctx // a.streams)][:a.streams]
    mx.eval(*xs)
    print(f"prefilled {a.ctx} tokens, captured {len(xs)} real residual streams", flush=True)

    res = dict(record_meta(), ctx=a.ctx, passes=a.passes, reps=a.reps, streams=len(xs),
               model=a.model, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    counts = {t: sum(1 for v in model.types if v == t) for t in ("mamba", "attention", "moe")}
    res["blocks"] = counts

    for tag in ("dense", "sparse"):
        if tag == "sparse":
            model.set_mode(tag)
        r = {}
        for kind in ("mamba", "attention", "moe"):
            layers = [l for l, t in enumerate(model.types) if t == kind]
            next_x = rotating(xs)

            def one(kind=kind, layers=layers, next_x=next_x):
                h = next_x()
                for l in layers:
                    if kind == "moe":
                        h = model.moe_fn[l](h)
                    else:
                        h = model.blocks[l](h[:, None, :], mask=None, cache=model._cache[l])[:, 0]
                return h

            r[f"chain_{kind}_ms"] = chain_ms(one, a.passes, reps=a.reps)
        next_x = rotating(xs)
        r["lm_head_argmax_ms"] = chain_ms(lambda: mx.argmax(model.logits(next_x()).astype(mx.float32), axis=-1),
                                          a.passes, reps=a.reps)
        next_x = rotating(xs)
        r["step_chain_ms"] = chain_ms(lambda: model.step(next_x()), a.passes, reps=a.reps)
        r["sum_chain_types_ms"] = (r["chain_mamba_ms"] + r["chain_attention_ms"]
                                   + r["chain_moe_ms"] + r["lm_head_argmax_ms"])
        last = None
        for t, st in model.generate(ids, 8 + a.gen, 0.0, chunk=2048):
            if st is not None:
                last = st
        r["loop_ms_per_token"] = last["ms_per_token"]
        r["loop_tps"] = last["decode_tps"]
        res[tag] = r
        print(f"[{tag}] one pass of all blocks per type: mamba x{counts['mamba']} {r['chain_mamba_ms']:.2f} ms, "
              f"attention x{counts['attention']} {r['chain_attention_ms']:.2f} ms, moe x{counts['moe']} "
              f"{r['chain_moe_ms']:.2f} ms, lm_head+argmax {r['lm_head_argmax_ms']:.2f} ms | sum of types "
              f"{r['sum_chain_types_ms']:.2f} | 52-block step chained {r['step_chain_ms']:.2f} ms | "
              f"measured loop {r['loop_ms_per_token']:.2f} ms/tok ({r['loop_tps']:.1f} tok/s)", flush=True)

    append_json(a.out, res)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
