#!/usr/bin/env bash
# (~1 min, CPU) last blends. Public board: clean : PKS-like = 1:1 is best (blend 0.8625; 1:2 blend6 0.8565,
# stack6_pl alone 0.8576). Keep 1:1 between the two groups, average more models inside each group:
#   blend13 = clean {stack6_pl, stack7_pl, stack_pl} : harsh {sim1, sim2}          (= blend8 + stack7_pl)
#   blend14 = clean {stack6_pl, stack7_pl, stack_pl} : harsh {sim1, sim2, sim3b}   (+ the harsher sim3 world)
set -uo pipefail
cd "$(dirname "$0")"
C6=../work/sub_r2_stack6_pl_probs.npz; C7=../work/sub_r2_stack7_pl_probs.npz; CP=../work/sub_r2_stack_pl_probs.npz
S1=../work_sim/sub_r2_sim1_probs.npz;  S2=../work_sim/sub_r2_sim2_probs.npz; S3=../work_sim3/sub_r2_sim3b_probs.npz
python blend.py --probs $C6 $C7 $CP $S1 $S2 --weights 2 2 2 3 3 --out sub_r2_blend13
python blend.py --probs $C6 $C7 $CP $S1 $S2 $S3 --out sub_r2_blend14
python blend.py --probs ../work/sub_r2_blend_probs.npz ../work/sub_r2_blend8_probs.npz ../work/sub_r2_blend13_probs.npz ../work/sub_r2_blend14_probs.npz --out sub_r2_blend_cmp2
