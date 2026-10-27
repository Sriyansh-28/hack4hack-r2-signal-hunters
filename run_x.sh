#!/usr/bin/env bash
# GPU job (~27 min): extra features (scintillation vs OFF-panel bandpass, ETI track fit) -> LightGBM gbm_x
# -> stackers sub_r2_stack4.csv / sub_r2_stack4_pl.csv (~7 min in); then CNN v2 seed 4 that also gets the extra
# features (~18 min) -> sub_r2_stack5.csv / sub_r2_stack5_pl.csv. "_pl" = + the pseudo-label CNN.
set -uo pipefail
cd "$(dirname "$0")"
t=$SECONDS; python feats3.py || exit 1; echo "=== feats3 took $(((SECONDS - t) / 60)) min"
t=$SECONDS; python gbm2.py --extra --tag _x || exit 1; echo "=== lightgbm x took $(((SECONDS - t) / 60)) min"
M="cnn gbm gbm_x cnn_v2 cnn_v2s1 cnn_v2s3 cnn_v2notab"
python stack2.py --extra --preds $M --out sub_r2_stack4
python stack2.py --extra --preds $M cnn_v2pl --out sub_r2_stack4_pl
t=$SECONDS; python cnn2.py --panel --tab --extra --epochs 30 --seeds 4 --tag _v2x || exit 1; echo "=== cnn v2x took $(((SECONDS - t) / 60)) min"
python stack2.py --extra --preds $M cnn_v2x --out sub_r2_stack5
python stack2.py --extra --preds $M cnn_v2x cnn_v2pl --out sub_r2_stack5_pl
