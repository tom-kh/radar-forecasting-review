#!/usr/bin/env bash
set -euo pipefail
CACHE="${1:?Usage: run_matrix.sh CACHE OUT}"
OUT="${2:?Output directory required}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a METHODS <<< "${VARIANTS:-scratch scratch_transport reconstruction jepa_dense jepa_observed doppler_jepa doppler_jepa_dense no_doppler}"
mkdir -p "$OUT/logs"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
PIDS=()
for ((worker=0; worker<${#GPUS[@]}; worker++)); do
    (
        export CUDA_VISIBLE_DEVICES="${GPUS[worker]}"
        for ((job=worker; job<${#METHODS[@]}; job+=${#GPUS[@]})); do
            METHOD="${METHODS[job]}"
            bash scripts/run_variant.sh "$CACHE" "$OUT" "$METHOD" > "$OUT/logs/$METHOD.log" 2>&1
        done
    ) &
    PIDS+=("$!")
done
STATUS=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then STATUS=1; fi
done
if [[ "$STATUS" -ne 0 ]]; then
    echo "One or more jobs failed. Inspect $OUT/logs/*.log" >&2
    exit "$STATUS"
fi
printf 'All requested development runs completed. Final test evaluation has NOT been run.\n'
