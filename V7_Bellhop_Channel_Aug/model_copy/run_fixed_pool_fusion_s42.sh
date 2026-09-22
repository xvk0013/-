#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python diagnose_fixed_pool_fusion.py \
  --gap-report runs/single_pool_B_expanded_gap_ce02_ship5_noise1_s42/val_report_20260919_013919_869785 \
  --freq4-report runs/single_pool_B_expanded_freq4_ce02_ship5_noise1_s42/val_report_20260919_151258_730181 \
  --data-dir "/mnt/d/LSJ/Data/deepship_source_pool_ab/20260916_225548_748/B_expanded"
