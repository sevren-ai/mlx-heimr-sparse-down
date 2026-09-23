"""Command-line generation with the sparse down projection and MTP speculative decoding.

Run from the repository root (results and texts resolve relative to it):

  python -m hsd --prompt "Why are whales not fish?"                  # sparse down, greedy
  python -m hsd --down dense --prompt "..."                           # stock mlx-lm blocks
  python -m hsd --mtp 2 --prompt "..."                               # sparse + MTP, 2 drafts
  python -m hsd --prompt-file texts/moby_dick.txt --prompt-tokens 2048 --max-tokens 256
  python -m hsd --raw --prompt "Call me Ishmael."                    # no chat template

One configuration per process: the GPU should never run two of these at once. Each run
appends a record (device, versions, timings, tokens, MTP acceptance) to --json.
"""
import argparse
import json
import os
import sys
import time

import mlx.core as mx

from .model import MODES, NemotronModel
from .mtp import MTPHead
from .resolve import DEFAULT_MTP_HEAD_ID, DEFAULT_MODEL_ID, resolve_head
from .runtime import record_meta

DEFAULT_PROMPT = "Write a short paragraph about why whales are not fish."
CODE_PROMPT = ("Write a Python function merge_sorted(a, b) that merges two sorted lists into one "
               "sorted list in linear time, then a few assert-based tests for it.")
# prompts longer than this also warm the prefill path in the warm-up: the first prefill a
# process runs pays for Metal kernel compilation, which would otherwise land in the timed
# prompt's prefill tok/s
WARM_PREFILL_MIN = 256


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL_ID, help="local path or HF repo id")
    ap.add_argument("--down", choices=MODES, default="sparse", help="down projection (default: sparse)")
    ap.add_argument("--mtp", type=int, default=0, help="MTP drafts per step (0 = off; 2 recommended)")
    ap.add_argument("--mtp-head", default=DEFAULT_MTP_HEAD_ID, help="the MTP head: local path or HF repo id")
    ap.add_argument("--prompt", default=None, help="the prompt text")
    ap.add_argument("--prompt-code", action="store_true", help="use the built-in code prompt instead of --prompt")
    ap.add_argument("--prompt-file", default=None, help="take the prompt from a text file")
    ap.add_argument("--prompt-tokens", type=int, default=4096, help="prompt tokens to take from --prompt-file")
    ap.add_argument("--raw", action="store_true", help="raw prompt (bos + tokens), no chat template")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0, help="sampling temperature (plain decode only; MTP is greedy)")
    ap.add_argument("--chunk", type=int, default=2048, help="prefill chunk size")
    ap.add_argument("--warmup", type=int, default=8, help="tokens generated and discarded before timing "
                                                         "(compiles the decode path; prompts longer than 256 "
                                                         "tokens also warm the prefill path on filler tokens)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--tag", default=None, help="a label for the run record (benchmarks use this)")
    ap.add_argument("--json", default=None, help="append the run record to this JSON file (e.g. results/generate.json)")
    return ap


def prompt_ids(a, model):
    """The prompt token ids: chat template by default (mlx-lm's own renderer), or raw
    bos + encoding with --raw / --prompt-file."""
    tok = model.tokenizer
    if a.prompt_file:
        text = open(a.prompt_file, encoding="utf-8", errors="replace").read()
        ids = tok.encode(text, add_special_tokens=False)[: a.prompt_tokens - 1]
        return [1] + list(ids)
    text = CODE_PROMPT if a.prompt_code else (a.prompt if a.prompt is not None else DEFAULT_PROMPT)
    if a.raw:
        return [1] + list(tok.encode(text, add_special_tokens=False))
    messages = [{"role": "user", "content": text}]
    return list(tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False))


def main(argv=None):
    a = build_parser().parse_args(argv)
    model_path = a.model
    print(f"model {model_path}\nloading ...", flush=True)
    model = NemotronModel(model_path, mode=a.down)
    cfg = json.load(open(os.path.join(model.path, "generation_config.json")))
    eos = cfg.get("eos_token_id", 2)
    a.eos = tuple(eos) if isinstance(eos, list) else (eos,)
    mtp = None
    if a.mtp:
        head_dir = resolve_head(a.mtp_head)
        print(f"MTP head {head_dir}", flush=True)
        mtp = MTPHead(model, head_dir)
        print(f"head loaded: {mtp.bytes / 1e6:.0f} MB of weights, {a.mtp} draft(s) per step", flush=True)
        if a.temp != 0.0:
            print("MTP speculative decoding is greedy; ignoring --temp", flush=True)
            a.temp = 0.0

    prompt = prompt_ids(a, model)

    def gen(prompt_ids_, n):
        """One generation of n tokens (the warm-up uses the warm-up count, not the run's)."""
        return (model.generate_mtp(prompt_ids_, mtp, n_draft=a.mtp, max_tokens=n, eos=a.eos, chunk=a.chunk)
                if a.mtp else model.generate(prompt_ids_, n, a.temp, a.eos, a.chunk))

    if a.warmup:
        t0 = time.perf_counter()
        # the short generation compiles the decode path
        for _ in gen(prompt[:8] if len(prompt) >= 8 else prompt, a.warmup):
            pass
        warm_note = ""
        if len(prompt) > WARM_PREFILL_MIN:
            # a long prompt also warms the prefill path, with a prefill of the same
            # length. The filler is repeated bos tokens, never a slice of the prompt
            # itself, so the real run's caches and graphs stay untouched by the real
            # text; every generation here gets its own fresh cache.
            for _ in gen([1] * len(prompt), a.warmup):
                pass
            warm_note = f" + a prefill of {len(prompt)} filler tokens"
        print(f"warmup {a.warmup} tokens{warm_note} in {time.perf_counter() - t0:.1f}s (compile)", flush=True)

    mx.reset_peak_memory()
    stats = None
    for t, st in gen(prompt, a.max_tokens):
        if st is None:
            if not a.quiet:
                sys.stdout.write(model.tokenizer.decode([t]))
                sys.stdout.flush()
    if not a.quiet:
        sys.stdout.write("\n")
        sys.stdout.flush()
    stats = st
    # the last token arrives with the stats, so decode from the stats' own token list
    stats["text"] = model.tokenizer.decode([t for t in stats["tokens"] if t not in a.eos])

    tag = a.tag or ("dense" if a.down == "dense" else "sparse") + (f"+mtp{a.mtp}" if a.mtp else "")
    extra = ""
    if a.mtp:
        sp = stats["spec"]
        extra = (f" | {sp['steps']} steps, {sp['ms_per_step']:.1f} ms/step, {sp['tokens_per_step']:.2f} tok/step"
                 f" (accepted drafts {sp['accepted_mean']:.2f} of {a.mtp}; hist {sp['accepted_hist']})")
    print(f"[{tag}] prompt {stats['prompt_tokens']} tok in {stats['prompt_s']:.2f}s ({stats['prompt_tps']:.0f} tok/s) | "
          f"decode {stats['gen_tokens']} tok: {stats['decode_tps']:.1f} tok/s ({stats['ms_per_token']:.2f} ms/tok) | "
          f"peak {stats['peak_gb']:.1f} GB{extra}", flush=True)

    if a.json:
        record = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), tag=tag, model=model_path, down=a.down,
                      mtp=a.mtp, raw=a.raw, prompt_file=a.prompt_file,
                      prompt_kind=("code" if a.prompt_code else
                                   (os.path.basename(a.prompt_file or "") or "chat") +
                                   (f":{a.prompt_tokens}" if a.prompt_file else "")),
                      chunk=a.chunk, max_tokens=a.max_tokens,
                      load_seconds=model.load_seconds, **record_meta(), **stats)
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        hist = json.load(open(a.json)) if os.path.exists(a.json) else []
        hist.append(record)
        json.dump(hist, open(a.json, "w"), indent=1)
        print(f"appended to {a.json}", flush=True)
    return stats


if __name__ == "__main__":
    main()
