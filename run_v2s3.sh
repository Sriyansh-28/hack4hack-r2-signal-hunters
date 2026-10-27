#!/usr/bin/env bash
# CNN v2, third seed (~18 min), then refresh the final stackers (run after run_final.sh)
set -uo pipefail
cd "$(dirname "$0")"
t=$SECONDS; python cnn2.py --panel --tab --epochs 30 --seeds 3 --tag _v2s3; echo "=== cnn v2 seed3 took $(((SECONDS - t) / 60)) min"
bash run_stack_pl.sh
