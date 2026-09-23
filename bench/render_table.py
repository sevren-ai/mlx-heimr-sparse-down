"""Render the end-to-end decode matrix JSON (results/e2e_decode.json, written by
bench/run_e2e_matrix.sh) into a markdown table: per prompt, tok/s per cell (dense / sparse
x plain / MTP 2), MTP acceptance, prefill tok/s, peak GB.

  python -m bench.render_table results/e2e_decode.json
  python -m bench.render_table results/e2e_decode.json --out results/e2e_decode.md
"""
import argparse
import json
import sys

CELLS = [("dense", 0, "dense"), ("sparse", 0, "sparse"),
         ("dense", 2, "dense + MTP 2"), ("sparse", 2, "sparse + MTP 2")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file")
    ap.add_argument("--out", default=None, help="write the markdown here instead of stdout")
    a = ap.parse_args()
    runs = json.load(open(a.json_file))

    # group the records by prompt kind and cell; keep the latest record for each
    by_prompt = {}
    for r in runs:
        kind = r.get("prompt_kind") or r.get("tag", "?")
        # the file prompts carry their length in the kind ("code_corpus.txt:2048"); the
        # length is printed next to the header, so strip it from the label
        label = kind.rsplit(":", 1)[0] if kind[-1:].isdigit() and ":" in kind else kind
        by_prompt.setdefault((label, r.get("prompt_tokens")), {})[(r.get("down"), r.get("mtp", 0))] = r

    lines = []
    dev = next((r.get("device") for r in runs if r.get("device")), "?")
    mlx = next((r.get("mlx") for r in runs if r.get("mlx")), "?")
    lines.append(f"End-to-end decode matrix ({dev}, MLX {mlx}), {len(runs)} runs.")
    for (kind, ptoks), cells in by_prompt.items():
        lines.append("")
        lines.append(f"### {kind} ({ptoks} prompt tokens)")
        lines.append("| cell | prefill tok/s | decode tok/s | ms/token | accepted drafts | peak GB |")
        lines.append("|---|---|---|---|---|---|")
        for down, mtp, label in CELLS:
            r = cells.get((down, mtp))
            if r is None:
                lines.append(f"| {label} | - | - | - | - | - |")
                continue
            acc = (f"{r['spec']['accepted_mean']:.2f} of {mtp} (hist {r['spec']['accepted_hist']})"
                   if r.get("spec") else "-")
            lines.append(f"| {label} | {r['prompt_tps']:.0f} | {r['decode_tps']:.1f} | "
                         f"{r['ms_per_token']:.2f} | {acc} | {r['peak_gb']:.1f} |")
    text = "\n".join(lines) + "\n"
    if a.out:
        open(a.out, "w").write(text)
        print(f"wrote {a.out}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
