#!/usr/bin/env bash
# (~1 min, CPU, after run_pl2.sh) same recipe as the best public file sub_r2_blend.csv (stack6_pl + sim2 = 0.8625),
# with the newer clean stack: stack7_pl + sim2 -> submissions/sub_r2_blend5.csv.
# (blend3 = + simb2 from the milder work_sim2 world scored 0.8591 public, lower than 0.8625, so work_sim2 is left out.)
set -uo pipefail
cd "$(dirname "$0")"
python blend.py --probs ../work/sub_r2_stack7_pl_probs.npz ../work_sim/sub_r2_sim2_probs.npz --out sub_r2_blend5
