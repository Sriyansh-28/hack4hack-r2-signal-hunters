#!/usr/bin/env bash
# (~1 min, CPU) last-hour blend variants, guided by the public board: stack6_pl 0.8576 -> + sim2 (harsh PKS-like v1)
# 1:1 = 0.8625; + simb2 (milder v2) = 0.8591. So push further toward the harsh simulation:
#   blend6 = stack6_pl : sim2 = 1 : 2
#   blend7 = stack6_pl + sim1 + sim2            (both harsh-sim stacks, 2/3 harsh)
#   blend8 = stack6_pl + stack_pl + sim1 + sim2  (two clean + two harsh, 1:1 like the 0.8625 file, more models)
set -uo pipefail
cd "$(dirname "$0")"
C6=../work/sub_r2_stack6_pl_probs.npz; CP=../work/sub_r2_stack_pl_probs.npz
S1=../work_sim/sub_r2_sim1_probs.npz;  S2=../work_sim/sub_r2_sim2_probs.npz
python blend.py --probs $C6 $S2 --weights 1 2 --out sub_r2_blend6
python blend.py --probs $C6 $S1 $S2 --out sub_r2_blend7
python blend.py --probs $C6 $CP $S1 $S2 --out sub_r2_blend8
python blend.py --probs ../work/sub_r2_blend_probs.npz ../work/sub_r2_blend6_probs.npz ../work/sub_r2_blend7_probs.npz ../work/sub_r2_blend8_probs.npz --out sub_r2_blend_cmp
