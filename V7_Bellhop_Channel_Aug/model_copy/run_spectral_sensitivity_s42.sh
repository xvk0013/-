#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python diagnose_spectral_sensitivity.py \
  --baseline "runs/single_pool_B_expanded_gap_ce02_ship5_noise1_s42/best.pt" \
  --augmented "runs/single_pool_B_expanded_gap_ce02_spec3db_ship5_noise1_s42/best.pt" \
  --data-dir "/mnt/d/LSJ/Data/deepship_source_pool_ab/20260916_225548_748/B_expanded" \
  --output-dir "runs/fixed_spectral_sensitivity_s42" \
  --device cuda --workers 4
