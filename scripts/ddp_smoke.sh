#!/usr/bin/env bash
set -euo pipefail
CACHE="${1:?Pass a synthetic cache from smoke.sh}"
OUT="${2:?Pass a fresh output directory}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m torch.distributed.run --standalone --nproc_per_node=2 -m doppler_jepa.train --cache "$CACHE" --out "$OUT/pretrain" --stage pretrain --width 16 --size 32 --batch 2 --workers 0 --epochs 1 --max-steps 2 --device cpu --amp off
python -m torch.distributed.run --standalone --nproc_per_node=2 -m doppler_jepa.train --cache "$CACHE" --out "$OUT/finetune" --stage finetune --init "$OUT/pretrain/last.pt" --width 16 --size 32 --batch 2 --workers 0 --epochs 1 --max-steps 2 --device cpu --amp off
