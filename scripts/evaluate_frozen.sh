#!/usr/bin/env bash
set -euo pipefail
CACHE="${1:?Usage: evaluate_frozen.sh CACHE RUN_DIR}"
RUN="${2:?Finetuning directory containing best.pt and threshold_selection required}"
if [[ "${PROTOCOL_FROZEN:-0}" != "1" ]]; then
    echo 'Freeze methods, hyperparameters, thresholds and stress tests first; then set PROTOCOL_FROZEN=1.' >&2
    exit 2
fi
CKPT="$RUN/best.pt"
CAL="$RUN/threshold_selection/threshold.json"
COMMON=(--cache "$CACHE" --checkpoint "$CKPT" --split test --calibration "$CAL" --batch "${BATCH:-16}" --workers "${WORKERS:-4}" --device cuda)
python -m doppler_jepa.evaluate "${COMMON[@]}" --out "$RUN/test_clean"
for RATE in 0.25 0.5 0.75; do
    python -m doppler_jepa.evaluate "${COMMON[@]}" --out "$RUN/test_drop_$RATE" --corruption dropout --severity "$RATE"
done
for N in 1 2; do
    python -m doppler_jepa.evaluate "${COMMON[@]}" --out "$RUN/test_burst_$N" --corruption burst --severity "$N"
done
python -m doppler_jepa.evaluate "${COMMON[@]}" --out "$RUN/test_velocity_noise_1" --corruption velocity_noise --severity 1
python -m doppler_jepa.evaluate "${COMMON[@]}" --out "$RUN/test_shuffled_doppler" --corruption shuffle_doppler
