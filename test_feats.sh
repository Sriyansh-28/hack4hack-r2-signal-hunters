#!/usr/bin/env bash
# Quick test of the feature + LightGBM part on 1,000 cadences (work_dbg). Cleans up after itself.
set -euo pipefail
cd "$(dirname "$0")"
W=../work_dbg2
echo "=================== TEST feats start $(date +%H:%M:%S) on ${CUDA_VISIBLE_DEVICES:-cpu}"
python prep2.py --work $W --debug 1000
python feats2.py --work $W
python gbm2.py --work $W --seeds 0
python ens2.py --work $W --out sub_dbg2
rm -rf $W ../submissions/sub_dbg2.csv
echo "=================== TEST feats OK $(date +%H:%M:%S)"
