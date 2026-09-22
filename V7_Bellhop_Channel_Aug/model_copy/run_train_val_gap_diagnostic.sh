#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
for seed in 42 43; do
  python diagnose_train_val_gap.py \
    --checkpoint "runs/single_pool_B_expanded_gap_ce02_ship5_noise1_s$seed/best.pt" \
    --data-dir "/mnt/d/LSJ/Data/deepship_source_pool_ab/20260916_225548_748/B_expanded" \
    --device cuda --workers 4
done
