#!/usr/bin/env bash
# CNN v2 (stacked-panel branch + physics features, 30 epochs) then the stacker over all models
# -> submissions/sub_r2_stack.csv.  Needs run_feats.sh done first (feats_*.parquet, preds/gbm.npz).
set -euo pipefail
cd "$(dirname "$0")"
python cnn2.py --panel --tab --epochs 1 --max_folds 1 --tag _smoke     # ~1 min crash check
t=$SECONDS; python cnn2.py --panel --tab --epochs 30 --seeds 0 --tag _v2; echo "=== cnn v2 took $(((SECONDS - t) / 60)) min"
python stack2.py --preds cnn gbm cnn_v2 --out sub_r2_stack
