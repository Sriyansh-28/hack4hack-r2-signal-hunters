#!/usr/bin/env bash
# Last-hour GPU job (~35 min) in ../work_sim3: the public board rewarded the HARSH simulation (sim v1, noise x1.15-1.45:
# blend 0.8625) and punished the milder v2 (x1.0-1.35 + glitches + fine de-dispersion: blend3 0.8591), so go harsher:
# same recipe as run_sim.sh with noise x1.35-1.75. Blends are written after each stage so a file exists early.
#   ~20 min: sub_r2_sim3a  -> blend9  = stack6_pl + sim2 + sim3a,  blend10 = stack6_pl + sim3a
#   ~35 min: sub_r2_sim3b (+ pseudo-label CNN) -> blend11 = stack6_pl + sim2 + sim3b, blend12 = stack6_pl + sim3b
set -uo pipefail
cd "$(dirname "$0")"
W=../work_sim3
C6=../work/sub_r2_stack6_pl_probs.npz; S2=../work_sim/sub_r2_sim2_probs.npz
t=$SECONDS; python prep2.py --work $W --sim --noise_lo 1.35 --noise_hi 1.75 || exit 1; echo "=== sim3 prep took $((SECONDS - t))s"
t=$SECONDS; python feats2.py --work $W --phys && python feats3.py --work $W --phys || exit 1; echo "=== sim3 feats took $(((SECONDS - t) / 60)) min"
t=$SECONDS; python gbm2.py --work $W --extra --tag _x || exit 1; echo "=== sim3 lightgbm took $(((SECONDS - t) / 60)) min"
t=$SECONDS; python cnn2.py --work $W --panel --tab --extra --epochs 30 --seeds 0 --tag _v2x || exit 1; echo "=== sim3 cnn took $(((SECONDS - t) / 60)) min"
python stack2.py --work $W --extra --preds gbm_x cnn_v2x --out sub_r2_sim3a
python blend.py --probs $C6 $S2 $W/sub_r2_sim3a_probs.npz --out sub_r2_blend9
python blend.py --probs $C6 $W/sub_r2_sim3a_probs.npz --out sub_r2_blend10
[ -f $W/pl_source_probs.npz ] || cp ../work/sub_r2_blend_probs.npz $W/pl_source_probs.npz      # best public file
t=$SECONDS; python cnn2.py --work $W --panel --tab --extra --epochs 30 --seeds 1 --pseudo $W/pl_source_probs.npz --pseudo_thr 0.9 --tag _v2xpl || exit 1; echo "=== sim3 cnn pseudo took $(((SECONDS - t) / 60)) min"
python stack2.py --work $W --extra --preds gbm_x cnn_v2x cnn_v2xpl --out sub_r2_sim3b
python blend.py --probs $C6 $S2 $W/sub_r2_sim3b_probs.npz --out sub_r2_blend11
python blend.py --probs $C6 $W/sub_r2_sim3b_probs.npz --out sub_r2_blend12
