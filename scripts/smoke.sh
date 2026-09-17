#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
OUT="${1:-/tmp/doppler_jepa_smoke_$(date +%s)}"
mkdir -p "$OUT"
python -m pytest -q
python -m doppler_jepa.synthetic --out "$OUT/cache"
python -m doppler_jepa.audit --cache "$OUT/cache" --size 32 --inspect 4
python -m doppler_jepa.train --cache "$OUT/cache" --out "$OUT/pretrain" --stage pretrain --width 16 --size 32 --batch 2 --workers 0 --epochs 2 --max-steps 3 --device cpu --amp off
python -m doppler_jepa.train --cache "$OUT/cache" --out "$OUT/finetune" --stage finetune --init "$OUT/pretrain/last.pt" --width 16 --size 32 --batch 2 --workers 0 --epochs 2 --max-steps 3 --device cpu --amp off
python -m doppler_jepa.evaluate --cache "$OUT/cache" --checkpoint "$OUT/finetune/best.pt" --split dev --calibrate --out "$OUT/calibration" --batch 2 --workers 0 --device cpu
python -m doppler_jepa.evaluate --cache "$OUT/cache" --checkpoint "$OUT/finetune/best.pt" --split test --calibration "$OUT/calibration/threshold.json" --out "$OUT/evaluation" --batch 2 --workers 0 --device cpu
python -m doppler_jepa.visualize --cache "$OUT/cache" --checkpoint "$OUT/finetune/best.pt" --out "$OUT/plots" --count 1
printf '\nSoftware smoke test complete. Artificial data only: %s\n' "$OUT"
