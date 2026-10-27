#!/usr/bin/env bash
# GPU job (~40 min, after run_blend3.sh): pseudo-labels round 2 from the best BLEND (public 0.8625 for the 2-way blend)
# instead of single stacks. One CNN in the PKS-like world (work_sim2) and one in the clean world (work), each restacked,
# then the 3-way blend again -> submissions/sub_r2_blend4.csv (stack7_pl + sim2 + simb3).
set -uo pipefail
cd "$(dirname "$0")"
S=../work/pl3_source_probs.npz
if [ ! -f $S ]; then
  [ -f ../work/sub_r2_blend3_probs.npz ] || python blend.py --probs ../work/sub_r2_stack6_pl_probs.npz ../work_sim/sub_r2_sim2_probs.npz --out sub_r2_blend
  if [ -f ../work/sub_r2_blend3_probs.npz ]; then cp ../work/sub_r2_blend3_probs.npz $S; else cp ../work/sub_r2_blend_probs.npz $S; fi
fi
W=../work_sim2
t=$SECONDS; python cnn2.py --work $W --panel --tab --extra --epochs 30 --seeds 2 --pseudo $S --pseudo_thr 0.9 --tag _v2xpl2 || exit 1; echo "=== sim2 cnn pseudo2 took $(((SECONDS - t) / 60)) min"
python stack2.py --work $W --extra --preds gbm_x cnn_v2x cnn_v2xpl cnn_v2xpl2 --out sub_r2_simb3
t=$SECONDS; python cnn2.py --panel --tab --extra --epochs 30 --seeds 6 --pseudo $S --pseudo_thr 0.9 --tag _v2xpl3 || exit 1; echo "=== clean cnn pseudo3 took $(((SECONDS - t) / 60)) min"
python stack2.py --extra --preds cnn gbm gbm_x cnn_v2 cnn_v2s1 cnn_v2s3 cnn_v2notab cnn_v2x cnn_v2pl cnn_v2xpl cnn_v2xpl3 --out sub_r2_stack7_pl
python blend.py --probs ../work/sub_r2_stack7_pl_probs.npz ../work_sim/sub_r2_sim2_probs.npz $W/sub_r2_simb3_probs.npz --out sub_r2_blend4
