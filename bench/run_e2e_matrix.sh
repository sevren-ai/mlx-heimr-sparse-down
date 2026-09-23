#!/bin/bash
# The end-to-end decode matrix: dense / sparse down x plain / MTP 2, on six prompts
# (short prose and code over the chat template, and raw continuations of the novel and of
# this repository's own source code, each at 2048 and 8192 prompt tokens),
# 256 generated tokens on the short prompts, 128 at long context, warm-up first.
#
# ONE configuration per process, run sequentially with a cool-down pause between runs so
# the GPU never runs two benchmarks at once. Each run appends its record (with a UTC
# timestamp) to the JSON file; render it with:
#   python -m bench.render_table results/e2e_decode.json
#
#   bench/run_e2e_matrix.sh                                # the whole matrix, once
#   PROMPTS=code2048,code8192 bench/run_e2e_matrix.sh       # only the code-corpus cells
#   REPEATS=2 bench/run_e2e_matrix.sh                       # twice in sequence (thermal drift)
#   COOLDOWN=30 bench/run_e2e_matrix.sh                     # a longer pause between runs
#   MODEL=/path/to/checkpoint HEAD=/path/to/head bench/run_e2e_matrix.sh
#   PY=.venv/bin/python bench/run_e2e_matrix.sh
#
# PROMPTS selects which prompt groups run: 'all' (the default) or a comma list of
#   prose code moby2048 moby8192 code2048 code8192
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
OUT="${OUT:-results/e2e_decode.json}"
COOLDOWN="${COOLDOWN:-10}"          # seconds to sleep between runs
MAXTOK="${MAXTOK:-256}"            # generated tokens on the short prompts (128 at long context)
REPEATS="${REPEATS:-1}"            # run the whole selection this many times in sequence
PROMPTS="${PROMPTS:-all}"          # all, or a comma list of prompt groups (see above)

# a local path or HF repo id, passed through to the CLI (which resolves them)
EXTRA=()
if [ -n "${MODEL:-}" ]; then EXTRA+=(--model "$MODEL"); fi
if [ -n "${HEAD:-}" ]; then EXTRA+=(--mtp-head "$HEAD"); fi

want() {  # does the PROMPTS selection include this prompt group?
    case ",$PROMPTS," in
        *,all,*) return 0 ;;
        *,"$1",*) return 0 ;;
        *) return 1 ;;
    esac
}

run() {  # run <tag> <extra CLI args...>
    local tag="$1"; shift
    echo "=== $tag ==="
    $PY -m hsd --quiet --json "$OUT" --tag "$tag" ${EXTRA[@]+"${EXTRA[@]}"} "$@"
    sleep "$COOLDOWN"
}

matrix() {
    if want prose; then
        run dense-prose    --down dense  --max-tokens "$MAXTOK"
        run sparse-prose   --down sparse --max-tokens "$MAXTOK"
        run dense-prose-mtp2    --down dense  --mtp 2 --max-tokens "$MAXTOK"
        run sparse-prose-mtp2   --down sparse --mtp 2 --max-tokens "$MAXTOK"
    fi

    if want code; then
        run dense-code     --down dense  --prompt-code --max-tokens "$MAXTOK"
        run sparse-code    --down sparse --prompt-code --max-tokens "$MAXTOK"
        run dense-code-mtp2     --down dense  --prompt-code --mtp 2 --max-tokens "$MAXTOK"
        run sparse-code-mtp2    --down sparse --prompt-code --mtp 2 --max-tokens "$MAXTOK"
    fi

    if want moby2048; then
        run dense-moby2048    --down dense  --prompt-file texts/moby_dick.txt --prompt-tokens 2048 --raw --max-tokens 128
        run sparse-moby2048   --down sparse --prompt-file texts/moby_dick.txt --prompt-tokens 2048 --raw --max-tokens 128
        run dense-moby2048-mtp2    --down dense  --prompt-file texts/moby_dick.txt --prompt-tokens 2048 --raw --mtp 2 --max-tokens 128
        run sparse-moby2048-mtp2   --down sparse --prompt-file texts/moby_dick.txt --prompt-tokens 2048 --raw --mtp 2 --max-tokens 128
    fi

    if want code2048; then
        run dense-code2048    --down dense  --prompt-file texts/code_corpus.txt --prompt-tokens 2048 --raw --max-tokens 128
        run sparse-code2048   --down sparse --prompt-file texts/code_corpus.txt --prompt-tokens 2048 --raw --max-tokens 128
        run dense-code2048-mtp2    --down dense  --prompt-file texts/code_corpus.txt --prompt-tokens 2048 --raw --mtp 2 --max-tokens 128
        run sparse-code2048-mtp2   --down sparse --prompt-file texts/code_corpus.txt --prompt-tokens 2048 --raw --mtp 2 --max-tokens 128
    fi

    if want moby8192; then
        run dense-moby8192    --down dense  --prompt-file texts/moby_dick.txt --prompt-tokens 8192 --raw --max-tokens 128
        run sparse-moby8192   --down sparse --prompt-file texts/moby_dick.txt --prompt-tokens 8192 --raw --max-tokens 128
        run dense-moby8192-mtp2    --down dense  --prompt-file texts/moby_dick.txt --prompt-tokens 8192 --raw --mtp 2 --max-tokens 128
        run sparse-moby8192-mtp2   --down sparse --prompt-file texts/moby_dick.txt --prompt-tokens 8192 --raw --mtp 2 --max-tokens 128
    fi

    if want code8192; then
        run dense-code8192    --down dense  --prompt-file texts/code_corpus.txt --prompt-tokens 8192 --raw --max-tokens 128
        run sparse-code8192   --down sparse --prompt-file texts/code_corpus.txt --prompt-tokens 8192 --raw --max-tokens 128
        run dense-code8192-mtp2    --down dense  --prompt-file texts/code_corpus.txt --prompt-tokens 8192 --raw --mtp 2 --max-tokens 128
        run sparse-code8192-mtp2   --down sparse --prompt-file texts/code_corpus.txt --prompt-tokens 8192 --raw --mtp 2 --max-tokens 128
    fi
}

for rep in $(seq 1 "$REPEATS"); do
    if [ "$REPEATS" -gt 1 ]; then
        echo "===== pass $rep of $REPEATS ====="
    fi
    matrix
done

echo "done; render with: $PY -m bench.render_table $OUT"
