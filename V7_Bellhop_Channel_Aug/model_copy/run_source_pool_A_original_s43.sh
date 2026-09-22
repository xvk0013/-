#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
resume_args=()
if [[ -f "runs/single_pool_A_original_ship5_noise1_s43/last.pt" ]]; then
  resume_args=(--resume "runs/single_pool_A_original_ship5_noise1_s43/last.pt")
fi
python train_source_pool.py "${resume_args[@]}" --phase single \
  --data-dir "/mnt/d/LSJ/Data/deepship_source_pool_ab/20260916_225548_748/A_original" \
  --noise-data-dir "/mnt/d/LSJ/Data/data_gen_v6_no_noise/20260916_225548_748/dataset" \
  --output-dir runs/single_pool_A_original_ship5_noise1_s43 \
  --device cuda --scan-backend cuda --batch-size 32 --workers 4 --seed 43 --epochs 60
python export_val_predictions.py --checkpoint runs/single_pool_A_original_ship5_noise1_s43/best.pt \
  --data-dir "/mnt/d/LSJ/Data/deepship_source_pool_ab/20260916_225548_748/A_original" --device cuda --workers 4
python summarize_source_pool.py --run-dir runs/single_pool_A_original_ship5_noise1_s43 --data-dir "/mnt/d/LSJ/Data/deepship_source_pool_ab/20260916_225548_748/A_original"
