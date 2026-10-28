#!/usr/bin/env bash
# Phase 2 GPU job (~70 min): cnn4.py (norm + Z_offref + Z_diff channels, BiGRU sequence head, feats5 tabular branch)
# in ../work with pseudo-labels from the laptop's blend28 (= blend27 recipe, public 0.8655, + cnn_v3pl;
# falls back to pl_blend16_probs.npz), then restack -> sub_r2_stack11_pl.
set -uo pipefail
cd "$(dirname "$0")"
PL=../work/pl_blend28_probs.npz; [ -f $PL ] || PL=../work/pl_blend16_probs.npz
t=$SECONDS; python cnn4.py --work ../work --width 48 --epochs 28 --seeds 0 --feats5 --pseudo $PL --pseudo_thr 0.90 --tag _v4pl || exit 1
echo "=== cnn4 took $(((SECONDS - t) / 60))m $(((SECONDS - t) % 60))s"
python stack2.py --work ../work --extra --feats4 --feats5 \
  --preds gbm_x4 gbm_x5 cnn_v2 cnn_v2s1 cnn_v2s3 cnn_v2notab cnn_v2x cnn_v2pl cnn_v2xpl cnn_v3pl cnn_v4pl \
  --out sub_r2_stack11_pl
