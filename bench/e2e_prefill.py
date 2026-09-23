"""End-to-end prefill: dense (stock checkpoint layout) vs sparse (single neuron-major
layout, matrix-unit kernel) on real literature, at several prompt lengths, with a few
decode tokens after. ONE configuration per process (the GPU must not run two at once).

The first prefill a process runs pays for Metal kernel compilation, so before the timed
prefill the script warms the path with a prefill of the same length on a different slice
of the text (fresh cache, discarded) and times that too: that is the cold figure. The
timed prefill then runs on its own fresh cache and is the warm figure, the benchmark. The
memory figures are the peak over the whole process, warm-up included.

  python -m bench.e2e_prefill --down sparse --prompt-tokens 2048
  bench/run_prefill_matrix.sh                 # runs every cell sequentially, with a cooldown
"""
import argparse
import time

import mlx.core as mx

from hsd.model import NemotronModel
from hsd.resolve import DEFAULT_MODEL_ID

from bench.common import append_json, moby_ids, record_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_ID, help="local path or HF repo id")
    ap.add_argument("--down", choices=("dense", "sparse"), default="sparse")
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--gen-tokens", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--skip", type=int, default=2000, help="first Moby Dick token to read")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="results/e2e_prefill.json")
    a = ap.parse_args()

    dev = mx.device_info()
    print(f"loading ({a.down} down) ...", flush=True)
    model = NemotronModel(a.model, mode=a.down, verbose=True)
    active_after_load = mx.get_active_memory() / 1e9
    ids = moby_ids(model, a.prompt_tokens, skip=a.skip)

    # The first prefill a process runs pays for Metal kernel compilation and first-launch
    # costs of every block, which is a real cost for a user running one prompt but not a
    # property of the kernels. Warm the path up with a prefill of the SAME length on a
    # different slice of the text (it starts well past the timed one, so the two share no
    # tokens), on a fresh cache, discarded. That warm-up prefill is the process's first,
    # so it is also the COLD figure. The timed prefill below then runs on its own fresh
    # cache and is the WARM figure, which is the benchmark. The memory figures are the
    # peak over the whole process, warm-up included: that is what a user experiences.
    warm_ids = moby_ids(model, a.prompt_tokens, skip=a.skip + a.prompt_tokens + 2048)
    t0 = time.perf_counter()
    model.prefill(mx.array(warm_ids, dtype=mx.int32), model.make_cache(), a.chunk)
    cold_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    cache = model.make_cache()
    h = model.prefill(mx.array(ids, dtype=mx.int32), cache, a.chunk)
    mx.eval(h)
    prefill_s = time.perf_counter() - t0
    peak_prefill = mx.get_peak_memory() / 1e9
    print(f"[{a.down} {a.prompt_tokens} tok] prefill {len(ids)} tok in {prefill_s:.2f}s "
          f"({len(ids) / prefill_s:.0f} tok/s, WARM, the benchmark) | "
          f"cold (the process's first prefill, a different slice): {len(warm_ids) / cold_s:.0f} tok/s | "
          f"active after load {active_after_load:.2f} GB | peak {peak_prefill:.2f} GB", flush=True)

    stats = None
    for t, st in model.generate(ids, a.gen_tokens, 0.0, chunk=a.chunk, cache=cache):
        if st is not None:
            stats = st
    print(f"  + {a.gen_tokens} decode tokens: {stats['decode_tps']:.1f} tok/s", flush=True)

    record = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), tag=a.tag or f"{a.down}-{a.prompt_tokens}",
                  down=a.down, prompt_tokens=len(ids),
                  cold_prefill_s=cold_s, cold_prefill_tps=len(warm_ids) / cold_s,
                  warm_prefill_s=prefill_s, warm_prefill_tps=len(ids) / prefill_s,
                  prompt_s=prefill_s, prompt_tps=len(ids) / prefill_s,      # the warm benchmark
                  warmup=dict(skip=a.skip + a.prompt_tokens + 2048, tokens=len(warm_ids)),
                  active_after_load_gb=active_after_load,
                  peak_prefill_gb=peak_prefill, peak_gb=stats["peak_gb"],
                  decode_tps=stats["decode_tps"], chunk=a.chunk,
                  model=a.model, **record_meta())
    append_json(a.out, record)
    print(f"appended to {a.out}", flush=True)


if __name__ == "__main__":
    main()
