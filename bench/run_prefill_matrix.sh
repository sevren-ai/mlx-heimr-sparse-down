#!/bin/bash
# The end-to-end prefill matrix: dense (stock checkpoint layout) vs sparse (single
# neuron-major layout, matrix-unit kernel) at 512, 2048 and 8192 prompt tokens of Moby
# Dick, with a few decode tokens after. ONE configuration per process, run sequentially
# with a cool-down pause between runs.
#
#   bench/run_prefill_matrix.sh
#   REPEATS=2 bench/run_prefill_matrix.sh     # twice in sequence (thermal drift shows in the second pass)
#   COOLDOWN=30 bench/run_prefill_matrix.sh
#   MODEL=/path/to/checkpoint bench/run_prefill_matrix.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
OUT="${OUT:-results/e2e_prefill.json}"
COOLDOWN="${COOLDOWN:-10}"
REPEATS="${REPEATS:-1}"            # run the whole matrix this many times in sequence

MODEL_ARG=()
if [ -n "${MODEL:-}" ]; then MODEL_ARG+=(--model "$MODEL"); fi

for rep in $(seq 1 "$REPEATS"); do
    if [ "$REPEATS" -gt 1 ]; then
        echo "===== pass $rep of $REPEATS ====="
    fi
    for n in 512 2048 8192; do
        for down in dense sparse; do
            echo "=== $down, $n prompt tokens ==="
            $PY -m bench.e2e_prefill --down "$down" --prompt-tokens "$n" --out "$OUT" \
                ${MODEL_ARG[@]+"${MODEL_ARG[@]}"}
            sleep "$COOLDOWN"
        done
    done
done
echo "done: $OUT"
