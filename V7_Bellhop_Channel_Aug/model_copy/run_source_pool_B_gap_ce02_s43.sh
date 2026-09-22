#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
resume_args=()
if [[ -f "runs/single_pool_B_expanded_gap_ce02_ship5_noise1_s43/last.pt" ]]; then
  resume_args=(--resume "runs/single_pool_B_expanded_gap_ce02_ship5_noise1_s43/last.pt")
fi
python train_source_pool.py "${resume_args[@]}" --phase single \
  --data-dir "/mnt/d/LSJ/Data/deepship_source_pool_ab/20260916_225548_748/B_expanded" \
  --noise-data-dir "/mnt/d/LSJ/Data/data_gen_v6_no_noise/20260916_225548_748/dataset" \
  --output-dir runs/single_pool_B_expanded_gap_ce02_ship5_noise1_s43 \
  --device cuda --scan-backend cuda --batch-size 32 --workers 4 --seed 43 --epochs 60 --single-ce-weight 0.2 --head-type gap
python export_val_predictions.py --checkpoint runs/single_pool_B_expanded_gap_ce02_ship5_noise1_s43/best.pt \
  --data-dir "/mnt/d/LSJ/Data/deepship_source_pool_ab/20260916_225548_748/B_expanded" --device cuda --workers 4
python summarize_source_pool.py --run-dir runs/single_pool_B_expanded_gap_ce02_ship5_noise1_s43 --data-dir "/mnt/d/LSJ/Data/deepship_source_pool_ab/20260916_225548_748/B_expanded"

for report_dir in runs/single_pool_B_expanded_gap_ce02_ship5_noise1_s43/val_report_*; do :; done
python diagnose_ranking_threshold.py --report-dir "$report_dir" \
  --data-dir "/mnt/d/LSJ/Data/deepship_source_pool_ab/20260916_225548_748/B_expanded"
