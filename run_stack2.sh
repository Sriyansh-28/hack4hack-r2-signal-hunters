#!/usr/bin/env bash
# Stacker over cnn, gbm, cnn_v2, cnn_v2s1 (CPU only) -> submissions/sub_r2_stack2.csv (local macro-F1 0.8942)
set -euo pipefail
cd "$(dirname "$0")"
python stack2.py --preds cnn gbm cnn_v2 cnn_v2s1 --out sub_r2_stack2
