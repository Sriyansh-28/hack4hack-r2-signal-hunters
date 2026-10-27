#!/usr/bin/env bash
# CNN v2, second seed (for the stacker)
set -euo pipefail
cd "$(dirname "$0")"
t=$SECONDS; python cnn2.py --panel --tab --epochs 30 --seeds 1 --tag _v2s1; echo "=== cnn v2 seed1 took $(((SECONDS - t) / 60)) min"
