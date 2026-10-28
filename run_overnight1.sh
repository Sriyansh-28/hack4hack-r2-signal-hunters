#!/usr/bin/env bash
# Comprehensive H100 Overnight Pipeline (fits in the 60-min 38480 job slot so we don't wait in queue twice!):
#   Stage 1 (~7 min):  feats4 (161-slope fine ETI track, collinearity trap detector, 5-of-6 persistent RFI,
#                      RFI-masked scintillation) + gbm_x4 + restack in clean & PKS-sim worlds
#                      -> sub_r2_blend19.csv (linear) and sub_r2_blend20.csv (50/50 linear + log-odds)
#   Stage 2 (~16 min): cnn3 (ResNet-SE + SpecAugment + feats4, width 48) in clean world + pseudo-labels
#                      -> restack -> sub_r2_blend21.csv
#   Stage 3 (~16 min): cnn3 (ResNet-SE + SpecAugment + feats4, width 48) in harsh PKS-sim world + pseudo-labels
#                      -> restack -> sub_r2_blend22.csv
set -uo pipefail
cd "$(dirname "$0")"

echo "=== Stage 1A: Clean World feats4 + gbm_x4 + stack8_pl ==="
t=$SECONDS
python feats4.py --work ../work --phys || exit 1
python gbm2.py --work ../work --extra --feats4 --tag _x4 || exit 1
python stack2.py --work ../work --extra --feats4 \
  --preds cnn gbm gbm_x gbm_x4 cnn_v2 cnn_v2s1 cnn_v2s3 cnn_v2notab cnn_v2x cnn_v2pl cnn_v2xpl \
  --out sub_r2_stack8_pl || exit 1
echo "=== Stage 1A took $(((SECONDS - t) / 60))m $(((SECONDS - t) % 60))s ==="

echo "=== Stage 1B: Harsh PKS-Sim World feats4 + gbm_x4 + sim4 ==="
t=$SECONDS
python feats4.py --work ../work_sim --phys || exit 1
python gbm2.py --work ../work_sim --extra --feats4 --tag _x4 || exit 1
python stack2.py --work ../work_sim --extra --feats4 \
  --preds gbm_x gbm_x4 cnn_v2x cnn_v2xpl \
  --out sub_r2_sim4 || exit 1
echo "=== Stage 1B took $(((SECONDS - t) / 60))m $(((SECONDS - t) % 60))s ==="

echo "=== Stage 1C: Write sub_r2_blend19.csv and sub_r2_blend20.csv (50/50 linear + log-odds) ==="
python blend.py --probs ../work/sub_r2_stack8_pl_probs.npz ../work/sub_r2_stack6_pl_probs.npz \
  ../work_sim/sub_r2_sim4_probs.npz ../work_sim/sub_r2_sim2_probs.npz --weights 2 1 2 1 --out sub_r2_blend19
python finetune_final.py --work ../work --probs ../work/sub_r2_stack8_pl_probs.npz ../work/sub_r2_stack6_pl_probs.npz \
  ../work_sim/sub_r2_sim1_probs.npz ../work_sim/sub_r2_sim4_probs.npz --out_linG sub_r2_blend20

# Save best consensus probs (blend16 = 0.8645) as pseudo-label source for cnn3
python finetune_final.py --work ../work --probs ../work/sub_r2_stack6_pl_probs.npz ../work/sub_r2_stack_pl_probs.npz \
  ../work_sim/sub_r2_sim1_probs.npz ../work_sim/sub_r2_sim2_probs.npz --save_probs ../work/pl_blend16_probs.npz

echo "=== Stage 2: Clean World CNN v3 (ResNet-SE + SpecAugment + feats4 + pseudo-labels) ==="
t=$SECONDS
python cnn3.py --work ../work --width 48 --epochs 28 --seeds 0 --pseudo ../work/pl_blend16_probs.npz --pseudo_thr 0.90 --tag _v3pl || exit 1
python stack2.py --work ../work --extra --feats4 \
  --preds cnn gbm gbm_x gbm_x4 cnn_v2 cnn_v2s1 cnn_v2s3 cnn_v2notab cnn_v2x cnn_v2pl cnn_v2xpl cnn_v3pl \
  --out sub_r2_stack9_pl || exit 1
python finetune_final.py --work ../work --probs ../work/sub_r2_stack9_pl_probs.npz ../work/sub_r2_stack6_pl_probs.npz \
  ../work_sim/sub_r2_sim2_probs.npz ../work_sim/sub_r2_sim4_probs.npz --out_linG sub_r2_blend21
echo "=== Stage 2 took $(((SECONDS - t) / 60))m $(((SECONDS - t) % 60))s ==="

echo "=== Stage 3: Harsh PKS-Sim World CNN v3 (ResNet-SE + SpecAugment + feats4 + pseudo-labels) ==="
t=$SECONDS
python cnn3.py --work ../work_sim --width 48 --epochs 28 --seeds 0 --pseudo ../work/pl_blend16_probs.npz --pseudo_thr 0.90 --tag _v3pl || exit 1
python stack2.py --work ../work_sim --extra --feats4 \
  --preds gbm_x gbm_x4 cnn_v2x cnn_v2xpl cnn_v3pl \
  --out sub_r2_sim5 || exit 1
python finetune_final.py --work ../work --probs ../work/sub_r2_stack9_pl_probs.npz ../work/sub_r2_stack8_pl_probs.npz \
  ../work_sim/sub_r2_sim2_probs.npz ../work_sim/sub_r2_sim5_probs.npz --out_linG sub_r2_blend22
echo "=== Stage 3 took $(((SECONDS - t) / 60))m $(((SECONDS - t) % 60))s ==="
