#!/usr/bin/env bash
# CPU-only smoke test of cnn4.py (cpuq, ~5 min): Z_offref build, 3 training batches, 256 val/test predictions, nothing saved.
set -uo pipefail
cd "$(dirname "$0")"
t=$SECONDS; python cnn4.py --work ../work --epochs 1 --bs 32 --width 16 --smoke --pseudo ../work/pl_blend16_probs.npz || exit 1
echo "=== cnn4 smoke took $((SECONDS - t))s"
