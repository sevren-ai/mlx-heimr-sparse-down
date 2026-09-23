#!/usr/bin/env python3
"""Build texts/code_corpus.txt, the long-context CODE prompt for the decode matrix.

The corpus is a raw concatenation of this repository's own Python files: the package
(hsd/), the benchmarks (bench/) and the tests (tests/), each file preceded by a short
comment line naming it, in one fixed sorted order (the sorted relative paths). It is fed
to the model raw, as a source-code continuation, at 2048 and 8192 prompt tokens.

The text is ours (MIT, like the repository), so it needs no third-party licence.

The full cycle of today's files is about 136 KB, about 41k tokens with this model's
tokenizer, which is already more than the 8192-token cell needs; the script still repeats
the whole cycle, in the same order, until the byte target is reached, so it keeps working
if the repository shrinks. The byte target (64 KiB) is chosen so that the corpus always
exceeds 8192 tokens: this tokenizer gives 3.34 bytes per token on the corpus, so 64 KiB
is over 19k tokens even at 4 bytes per token.

Run from the repository root:

    python scripts/make_code_corpus.py            # writes texts/code_corpus.txt
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = ("hsd", "bench", "tests")          # the package, the benchmarks, the tests
OUT = os.path.join(ROOT, "texts", "code_corpus.txt")
TARGET_BYTES = 64 * 1024                 # see the docstring


def file_list():
    """Every .py file of the three directories, one fixed sorted order."""
    paths = []
    for d in DIRS:
        full = os.path.join(ROOT, d)
        paths += [os.path.join(d, f) for f in sorted(os.listdir(full)) if f.endswith(".py")]
    return sorted(paths)


def cycle(paths):
    """One pass over every file: a comment line naming it, then its contents."""
    parts = []
    for rel in paths:
        with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as fh:
            body = fh.read().rstrip("\n")
        parts.append(f"# ---- {rel}\n{body}\n\n")
    return "".join(parts)


def main():
    paths = file_list()
    if not paths:
        sys.exit("no python files found")
    first = cycle(paths)
    whole, cycles = first, 1
    while len(whole.encode()) < TARGET_BYTES:        # repeat the cycle, same order, if short
        whole += first
        cycles += 1
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(whole)
    print(f"wrote {OUT}: {cycles} cycle(s) of {len(paths)} files, {len(whole.encode())} bytes")


if __name__ == "__main__":
    main()
