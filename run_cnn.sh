#!/usr/bin/env bash
# Full run: clean the data, train the 5-fold CNN, write submissions/sub_r2_cnn.csv
set -euo pipefail
cd "$(dirname "$0")"
t=$SECONDS
python prep2.py
echo "=== prep took $((SECONDS - t))s"
t=$SECONDS
python cnn2.py --epochs 20 --seeds 0
echo "=== cnn took $(((SECONDS - t) / 60)) min"
python ens2.py --preds cnn --out sub_r2_cnn
