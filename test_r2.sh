#!/usr/bin/env bash
# Quick end-to-end test on 1,000 cadences (work_dbg), ~3 min on a GPU slice. Cleans up after itself.
set -euo pipefail
cd "$(dirname "$0")"
W=../work_dbg
echo "=================== TEST start $(date +%H:%M:%S) on ${CUDA_VISIBLE_DEVICES:-cpu}"
python prep2.py --work $W --debug 1000
python cnn2.py --work $W --epochs 2 --tag _t
python ens2.py --work $W --out sub_dbg
rm -rf $W ../submissions/sub_dbg.csv
echo "=================== TEST OK $(date +%H:%M:%S)"
