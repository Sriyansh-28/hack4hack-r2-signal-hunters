#!/usr/bin/env bash
# PKS-like pipeline, GPU job (~45 min), in its own folder ../work_sim (the old ../work is untouched):
# train cadences degraded to PKS noise (x1.15-1.45, more dropped rows) and slope grids in physical drift units
# (PKS drift in pixels = 0.64x GBT), so validation scores are PKS-like.
# -> submissions/sub_r2_sim1.csv (~22 min in: LightGBM + CNN) and sub_r2_sim2.csv (~45 min: + CNN with pseudo-labels)
set -uo pipefail
cd "$(dirname "$0")"
W=../work_sim
t=$SECONDS; python prep2.py --work $W --sim || exit 1; echo "=== sim prep took $((SECONDS - t))s"
t=$SECONDS; python feats2.py --work $W --phys && python feats3.py --work $W --phys || exit 1; echo "=== sim feats took $(((SECONDS - t) / 60)) min"
t=$SECONDS; python gbm2.py --work $W --extra --tag _x || exit 1; echo "=== sim lightgbm took $(((SECONDS - t) / 60)) min"
t=$SECONDS; python cnn2.py --work $W --panel --tab --extra --epochs 30 --seeds 0 --tag _v2x || exit 1; echo "=== sim cnn took $(((SECONDS - t) / 60)) min"
python stack2.py --work $W --extra --preds gbm_x cnn_v2x --out sub_r2_sim1
[ -f $W/pl_source_probs.npz ] || cp $W/sub_r2_sim1_probs.npz $W/pl_source_probs.npz
t=$SECONDS; python cnn2.py --work $W --panel --tab --extra --epochs 30 --seeds 1 --pseudo $W/pl_source_probs.npz --pseudo_thr 0.9 --tag _v2xpl || exit 1; echo "=== sim cnn pseudo took $(((SECONDS - t) / 60)) min"
python stack2.py --work $W --extra --preds gbm_x cnn_v2x cnn_v2xpl --out sub_r2_sim2
