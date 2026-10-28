#!/usr/bin/env bash
# Phase 4 GPU job (~85 min): second seed of cnn4 in both worlds (variance reduction for the stacker),
# work_sim first (cnn4 gave the larger gain there: sim5 0.8596 -> sim6 0.8674), then work.
set -uo pipefail
cd "$(dirname "$0")"
PL=../work/pl_blend28_probs.npz; [ -f $PL ] || PL=../work/pl_blend16_probs.npz
t=$SECONDS; python cnn4.py --work ../work_sim --width 48 --epochs 28 --seeds 1 --feats5 --pseudo $PL --pseudo_thr 0.90 --tag _v4pls1 || exit 1
echo "=== sim cnn4 seed 1 took $(((SECONDS - t) / 60))m $(((SECONDS - t) % 60))s"
t=$SECONDS; python cnn4.py --work ../work --width 48 --epochs 28 --seeds 1 --feats5 --pseudo $PL --pseudo_thr 0.90 --tag _v4pls1 || exit 1
echo "=== clean cnn4 seed 1 took $(((SECONDS - t) / 60))m $(((SECONDS - t) % 60))s"
