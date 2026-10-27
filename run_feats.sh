#!/usr/bin/env bash
# Features + LightGBM, then blend with the CNN if its predictions exist -> submissions/sub_r2_mix.csv
# Needs prep2.py output (run_cnn.sh runs it). Usage: bash submit2.sh run_feats.sh 01:30:00
set -euo pipefail
cd "$(dirname "$0")"
[ -f ../work/norm_train.npy ] || python prep2.py
t=$SECONDS; python feats2.py; echo "=== features took $(((SECONDS - t) / 60)) min"
t=$SECONDS; python gbm2.py;   echo "=== lightgbm took $(((SECONDS - t) / 60)) min"
python ens2.py --out sub_r2_mix
