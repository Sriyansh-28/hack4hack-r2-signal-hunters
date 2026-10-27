#!/usr/bin/env bash
# (~1 min) equal-weight blend of the clean-validated stack6_pl and the PKS-like sim2 -> submissions/sub_r2_blend.csv
set -uo pipefail
cd "$(dirname "$0")"
python blend.py --probs ../work/sub_r2_stack6_pl_probs.npz ../work_sim/sub_r2_sim2_probs.npz --out sub_r2_blend
