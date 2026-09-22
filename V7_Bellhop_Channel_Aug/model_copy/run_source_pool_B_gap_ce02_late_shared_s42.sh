#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
resume_args=()
if [[ -f "runs/single_pool_B_expanded_gap_ce02_late_shared_ship5_noise1_s42/last.pt" ]]; then
  resume_args=(--resume "runs/single_pool_B_expanded_gap_ce02_late_shared_ship5_noise1_s42/last.pt")
fi
python train_source_pool.py "${resume_args[@]}" --phase single \
  --data-dir "/mnt/d/LSJ/Data/deepship_source_pool_ab/20260916_225548_748/B_expanded" \
  --noise-data-dir "/mnt/d/LSJ/Data/data_gen_v6_no_noise/20260916_225548_748/dataset" \
  --output-dir runs/single_pool_B_expanded_gap_ce02_late_shared_ship5_noise1_s42 \
  --device cuda --scan-backend cuda --batch-size 32 --workers 4 --seed 42 --epochs 60 --single-ce-weight 0.2 --head-type gap --weight-decay 0.0001 --local-contrast-bins 0 --band-fusion late_shared --spectral-aug-prob 0
python export_val_predictions.py --checkpoint runs/single_pool_B_expanded_gap_ce02_late_shared_ship5_noise1_s42/best.pt \
  --data-dir "/mnt/d/LSJ/Data/deepship_source_pool_ab/20260916_225548_748/B_expanded" --device cuda --workers 4
python summarize_source_pool.py --run-dir runs/single_pool_B_expanded_gap_ce02_late_shared_ship5_noise1_s42 --data-dir "/mnt/d/LSJ/Data/deepship_source_pool_ab/20260916_225548_748/B_expanded"

for report_dir in runs/single_pool_B_expanded_gap_ce02_late_shared_ship5_noise1_s42/val_report_*; do :; done
python diagnose_ranking_threshold.py --report-dir "$report_dir" \
  --data-dir "/mnt/d/LSJ/Data/deepship_source_pool_ab/20260916_225548_748/B_expanded"

python diagnose_train_val_gap.py \
  --checkpoint "runs/single_pool_B_expanded_gap_ce02_late_shared_ship5_noise1_s42/best.pt" \
  --data-dir "/mnt/d/LSJ/Data/deepship_source_pool_ab/20260916_225548_748/B_expanded" \
  --device cuda --workers 4
