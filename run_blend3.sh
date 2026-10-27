#!/usr/bin/env bash
# (~1 min, after run_sim2.sh) 3-way blend: clean-validated stack6_pl + PKS-like sim2 (v1) + simb2 (v2)
# -> submissions/sub_r2_blend3.csv. (sub_r2_blend.csv = stack6_pl + sim2 scored 0.8625 public, best so far.)
set -uo pipefail
cd "$(dirname "$0")"
python blend.py --probs ../work/sub_r2_stack6_pl_probs.npz ../work_sim/sub_r2_sim2_probs.npz ../work_sim2/sub_r2_simb2_probs.npz --out sub_r2_blend3
