#!/usr/bin/env bash
# GPU job (~26 min, after run_x.sh): CNN v2 + extra features + pseudo-labels from sub_r2_stack4 (the stack WITHOUT
# pseudo-labels, OOF 0.9083; frozen in work/pl2_source_probs.npz) -> sub_r2_stack6_pl.csv
set -uo pipefail
cd "$(dirname "$0")"
[ -f ../work/pl2_source_probs.npz ] || cp ../work/sub_r2_stack4_probs.npz ../work/pl2_source_probs.npz
t=$SECONDS; python cnn2.py --panel --tab --extra --epochs 30 --seeds 5 --pseudo ../work/pl2_source_probs.npz --pseudo_thr 0.9 --tag _v2xpl || exit 1; echo "=== cnn v2x pseudo took $(((SECONDS - t) / 60)) min"
python stack2.py --extra --preds cnn gbm gbm_x cnn_v2 cnn_v2s1 cnn_v2s3 cnn_v2notab cnn_v2x cnn_v2pl cnn_v2xpl --out sub_r2_stack6_pl
