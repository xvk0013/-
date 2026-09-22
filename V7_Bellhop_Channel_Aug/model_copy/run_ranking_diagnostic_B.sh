#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python diagnose_ranking_threshold.py \
  --report-dir runs/single_pool_B_expanded_ship5_noise1_s42/val_report_20260918_190616_292167 \
  --data-dir "/mnt/d/LSJ/Data/deepship_source_pool_ab/20260916_225548_748/B_expanded"
python diagnose_ranking_threshold.py \
  --report-dir runs/single_pool_B_expanded_ship5_noise1_s43/val_report_20260918_230540_185333 \
  --data-dir "/mnt/d/LSJ/Data/deepship_source_pool_ab/20260916_225548_748/B_expanded"
