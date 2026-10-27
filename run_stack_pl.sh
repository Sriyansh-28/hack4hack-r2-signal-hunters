#!/usr/bin/env bash
# Final stackers (CPU, ~1 min each). stack2.py skips models whose preds file does not exist yet.
#   sub_r2_stack3.csv   : every model trained only on the labelled train data
#   sub_r2_stack_pl.csv : the same + the pseudo-label CNN (test-telescope adaptation; its OOF is slightly optimistic)
set -uo pipefail
cd "$(dirname "$0")"
python stack2.py --preds cnn gbm cnn_v2 cnn_v2s1 cnn_v2s3 cnn_v2notab --out sub_r2_stack3
python stack2.py --preds cnn gbm cnn_v2 cnn_v2s1 cnn_v2s3 cnn_v2notab cnn_v2pl --out sub_r2_stack_pl
