#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python train.py \
  --phase single \
  --data-dir "/mnt/d/LSJ/Data/deepship_5s_no_bellhop/20260916_225548_748_paired/bellhop_reference" \
  --noise-data-dir "/mnt/d/LSJ/Data/data_gen_v6_no_noise/20260916_225548_748/dataset" \
  --output-dir runs/single_paired_bellhop_reference_cr0_ce02_rb_emr_s42 \
  --device cuda --scan-backend cuda \
  --batch-size 32 --workers 4 --seed 42 --epochs 60

python export_val_predictions.py \
  --checkpoint runs/single_paired_bellhop_reference_cr0_ce02_rb_emr_s42/best.pt \
  --data-dir "/mnt/d/LSJ/Data/deepship_5s_no_bellhop/20260916_225548_748_paired/bellhop_reference" \
  --device cuda --workers 4
