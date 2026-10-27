#!/usr/bin/env bash
# CNN v2 without the physics features (more different, for the stacker)
set -euo pipefail
cd "$(dirname "$0")"
t=$SECONDS; python cnn2.py --panel --epochs 30 --seeds 0 --tag _v2notab; echo "=== cnn v2 notab took $(((SECONDS - t) / 60)) min"
