#!/usr/bin/env bash
# (~1 min, CPU) blend8 (public 0.8639: stack6_pl + stack_pl + sim1 + sim2) with a per-class decision bias tuned for
# macro-F1 on the same blend's out-of-fold probabilities, shrunk x0.5 -> submissions/sub_r2_blend15.csv
set -uo pipefail
cd "$(dirname "$0")"
python bias2.py --probs ../work/sub_r2_stack6_pl_probs.npz ../work/sub_r2_stack_pl_probs.npz \
  ../work_sim/sub_r2_sim1_probs.npz ../work_sim/sub_r2_sim2_probs.npz --shrink 0.5 --out sub_r2_blend15
