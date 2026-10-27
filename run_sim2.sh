#!/usr/bin/env bash
# PKS-like pipeline v2, GPU job (~35 min), in ../work_sim2. Calibrated by xval.py / diag3.py:
# milder noise (x1.0-1.35: sim v1 was harsher than real PKS), +1 broadband glitch row in 30% of train cadences
# (PKS has ~2x more glitches), finer de-dispersion grid near 0 (PKS pulses are much less dispersed), physical slopes.
# -> sub_r2_simb1.csv (~20 min), sub_r2_simb2.csv (+ pseudo-label CNN, ~35 min),
#    sub_r2_blend2.csv = average of the clean-validated stack6_pl and simb2 (final candidate)
set -uo pipefail
cd "$(dirname "$0")"
W=../work_sim2
t=$SECONDS; python prep2.py --work $W --sim --noise_lo 1.0 --noise_hi 1.35 --glitch_p 0.3 || exit 1; echo "=== sim2 prep took $((SECONDS - t))s"
t=$SECONDS; python feats2.py --work $W --phys --fine_disp && python feats3.py --work $W --phys || exit 1; echo "=== sim2 feats took $(((SECONDS - t) / 60)) min"
t=$SECONDS; python gbm2.py --work $W --extra --tag _x || exit 1; echo "=== sim2 lightgbm took $(((SECONDS - t) / 60)) min"
t=$SECONDS; python cnn2.py --work $W --panel --tab --extra --epochs 30 --seeds 0 --tag _v2x || exit 1; echo "=== sim2 cnn took $(((SECONDS - t) / 60)) min"
python stack2.py --work $W --extra --preds gbm_x cnn_v2x --out sub_r2_simb1
[ -f $W/pl_source_probs.npz ] || cp $W/sub_r2_simb1_probs.npz $W/pl_source_probs.npz
t=$SECONDS; python cnn2.py --work $W --panel --tab --extra --epochs 30 --seeds 1 --pseudo $W/pl_source_probs.npz --pseudo_thr 0.9 --tag _v2xpl || exit 1; echo "=== sim2 cnn pseudo took $(((SECONDS - t) / 60)) min"
python stack2.py --work $W --extra --preds gbm_x cnn_v2x cnn_v2xpl --out sub_r2_simb2
python blend.py --probs ../work/sub_r2_stack6_pl_probs.npz $W/sub_r2_simb2_probs.npz --out sub_r2_blend2
