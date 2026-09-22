#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec python -u run_channel_aug.py "$@"
