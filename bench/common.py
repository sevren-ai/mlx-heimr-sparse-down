"""Shared helpers for the benchmark scripts: real-activation capture, lazy-queue timing,
the byte model of the down projection, and the common run-record fields."""
import base64
import json
import os
import re
import subprocess
import time

import mlx.core as mx
import numpy as np

from hsd.runtime import record_meta  # noqa: F401  (re-exported for the bench scripts)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 2688
ROW_B = D // 2               # nibble bytes of one neuron's down row (D/2)
GRP_B = 2 * 2 * D            # bf16 scales + bf16 biases of one 64-neuron group, D outputs each
GS = 64


def gpu_core_count():
    """The GPU's core count, from IOKit (mx.device_info does not report it)."""
    try:
        out = subprocess.run(["ioreg", "-r", "-c", "IOAccelerator", "-d", "1"],
                              capture_output=True, text=True, timeout=10).stdout
        m = re.search(r'"core-count"\s*=\s*(\d+)', out)
        return int(m.group(1)) if m else None
    except Exception:
        return None



def moby_ids(model, n, skip=2000):
    """Token ids for a literature continuation: bos + n-1 tokens of Moby Dick from `skip`."""
    text = open(os.path.join(ROOT, "texts", "moby_dick.txt"), encoding="utf-8", errors="replace").read()
    ids = model.tokenizer.encode(text, add_special_tokens=False)
    return [1] + list(ids[skip:skip + n - 1])


def chain_ms(fn, n, reps=5, warm=2):
    """GPU ms of one call of fn, measured the way the decode loop runs: n calls queued
    lazily and evaluated once, best of `reps` repeats after `warm` warm-up queues."""
    for _ in range(warm):
        mx.eval(fn())
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        outs = [fn() for _ in range(n)]
        mx.eval(*outs)
        best = min(best, (time.perf_counter() - t0) / n)
    return 1e3 * best


def capture_moe_inputs(model, ids, cache=None, chunk=2048):
    """Run a full prefill pass and capture the residual stream entering every MoE layer.
    -> {layer: x [T, D]}; the blocks still run normally through the model. Pass a `cache`
    to fill it with the real context as a side effect (the step-breakdown benchmark needs
    the attention / Mamba caches warm when it times decode-like passes)."""
    caps = {}
    for l in model.moe_layers:
        blk = model.blocks[l]
        orig = blk.prefill                      # the bound method, captured before shadowing

        def pf(x, orig=orig, l=l):
            caps[l] = x
            return orig(x)

        blk.prefill = pf
    model.prefill(mx.array(ids, dtype=mx.int32), cache or model.make_cache(), chunk=chunk)
    for l in model.moe_layers:
        del model.blocks[l].prefill              # back to the class method
    for x in caps.values():
        mx.eval(x)
    return caps


def byte_model(live_rows, live_groups, n_slots, width):
    """Bytes of the down projection actually read vs the dense total, for n_slots experts
    of `width` neurons given the mean live row / live 64-group counts."""
    sparse = live_rows * ROW_B + live_groups * GRP_B
    dense = n_slots * (width * ROW_B + width / GS * GRP_B)
    return sparse, dense


def append_json(path, record):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    hist = json.load(open(path)) if os.path.exists(path) else []
    hist.append(record)
    json.dump(hist, open(path, "w"), indent=1)


def live_stats(h, width):
    """h [width] after ReLU2 -> (live rows, live 64-groups, live fraction, live group fraction)."""
    rows = (h != 0)
    groups = rows.reshape(-1, GS).any(-1)
    return dict(rows=float(rows.sum()), groups=float(groups.sum()),
                rows_frac=float(rows.mean()), groups_frac=float(groups.mean()))


def pack_masks_b64(masks):
    """A boolean [n, width] live-mask -> one base64 string, ceil(width/8) bytes per row,
    bit k of byte j standing for element 8j + k of the row, low bit first."""
    packed = np.packbits(np.asarray(masks, dtype=bool), axis=-1, bitorder="little")
    return base64.b64encode(packed.tobytes()).decode("ascii")
