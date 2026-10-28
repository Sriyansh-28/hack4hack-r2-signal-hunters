#!/usr/bin/env bash
# Phase 3 GPU job (~45 min): cnn4.py in the harsh PKS-like world ../work_sim (degraded raw train cadences there),
# same pseudo-labels as Phase 2 (pl_blend28), then restack -> sub_r2_sim6 (sim side of the blends, weight 1.4).
set -uo pipefail
cd "$(dirname "$0")"
PL=../work/pl_blend28_probs.npz; [ -f $PL ] || PL=../work/pl_blend16_probs.npz
t=$SECONDS; python cnn4.py --work ../work_sim --width 48 --epochs 28 --seeds 0 --feats5 --pseudo $PL --pseudo_thr 0.90 --tag _v4pl || exit 1
echo "=== sim cnn4 took $(((SECONDS - t) / 60))m $(((SECONDS - t) % 60))s"
python stack2.py --work ../work_sim --extra --feats4 --feats5 \
  --preds gbm_x gbm_x4 gbm_x5 cnn_v2x cnn_v2xpl cnn_v3pl cnn_v4pl \
  --out sub_r2_sim6
