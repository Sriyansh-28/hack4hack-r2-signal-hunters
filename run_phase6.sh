#!/usr/bin/env bash
# Phase 6 GPU job (~78 min): cnn4 seed 3 with refreshed pseudo-labels from blend33 (public 0.8804; blend28 before),
# work_sim first (largest blend weight), then work. Output cnn_v4pl33.npz in each world -> laptop blend35.
set -uo pipefail
cd "$(dirname "$0")"
PL=../work/pl_blend33_probs.npz
t=$SECONDS; python cnn4.py --work ../work_sim --width 48 --epochs 28 --seeds 3 --feats5 --pseudo $PL --pseudo_thr 0.90 --tag _v4pl33 || exit 1
echo "=== sim cnn4 pl33 took $(((SECONDS - t) / 60))m $(((SECONDS - t) % 60))s"
t=$SECONDS; python cnn4.py --work ../work --width 48 --epochs 28 --seeds 3 --feats5 --pseudo $PL --pseudo_thr 0.90 --tag _v4pl33 || exit 1
echo "=== clean cnn4 pl33 took $(((SECONDS - t) / 60))m $(((SECONDS - t) % 60))s"
