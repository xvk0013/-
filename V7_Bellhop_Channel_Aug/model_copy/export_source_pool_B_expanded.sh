#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python export_val_predictions.py --checkpoint runs/single_pool_B_expanded_ship5_noise1_s42/best.pt \
  --data-dir "/mnt/d/LSJ/Data/deepship_source_pool_ab/20260916_225548_748/B_expanded" --device cuda --workers 4
python summarize_source_pool.py --run-dir runs/single_pool_B_expanded_ship5_noise1_s42 --data-dir "/mnt/d/LSJ/Data/deepship_source_pool_ab/20260916_225548_748/B_expanded"
