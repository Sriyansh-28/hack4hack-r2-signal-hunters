#!/usr/bin/env bash
# CNN v2 + pseudo-labels: also trains on PKS test cadences the best stack (sub_r2_stack2) is >= 90% sure about
# (test-telescope adaptation). The source probabilities are frozen once in work/pl_source_probs.npz.
set -euo pipefail
cd "$(dirname "$0")"
[ -f ../work/pl_source_probs.npz ] || cp ../work/sub_r2_stack2_probs.npz ../work/pl_source_probs.npz
t=$SECONDS; python cnn2.py --panel --tab --epochs 30 --seeds 2 --pseudo ../work/pl_source_probs.npz --pseudo_thr 0.9 --tag _v2pl; echo "=== cnn v2 pseudo took $(((SECONDS - t) / 60)) min"
