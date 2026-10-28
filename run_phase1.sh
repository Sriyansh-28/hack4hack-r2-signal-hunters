#!/usr/bin/env bash
# Phase 1 + Stage 3 H100 Pipeline (~65 min total, runs in 01:40:00 slot with ncpus=1:ngpus=1:mem=12gb):
#   1. Stage 1D (~12 min): feats5.py (Brzycki 2023 2D-ACF + Secondary Spectrum 2D-FFT + turboSETI bilinear sub-pixel ETI track)
#      + gbm_x5 in both ../work (Clean) and ../work_sim (Harsh PKS-Sim)
#      + restack ../work -> sub_r2_stack10_pl (with cnn_v3pl + gbm_x5 + feats5)
#      + restack ../work_sim -> sub_r2_sim4b (with gbm_x5 + feats5)
#      + write sub_r2_blend23.csv (ready in ~12 minutes!)
#   2. Stage 3 (~50 min): cnn3.py (ResNet-SE + SpecAugment + feats4 + pseudo-labels) in ../work_sim
#      + restack ../work_sim -> sub_r2_sim5 (with cnn_v3pl + gbm_x5 + feats5)
#      + write sub_r2_blend24_v3.csv
set -uo pipefail
cd "$(dirname "$0")"

echo "=== Phase 1A: Clean World feats5 (Brzycki 2D-ACF + 2D-FFT + Sub-Pixel ETI) + gbm_x5 + stack10_pl ==="
t=$SECONDS
python feats5.py --work ../work --phys || exit 1
python gbm2.py --work ../work --extra --feats4 --feats5 --tag _x5 || exit 1
python stack2.py --work ../work --extra --feats4 --feats5 \
  --preds gbm_x4 gbm_x5 cnn_v2 cnn_v2s1 cnn_v2s3 cnn_v2notab cnn_v2x cnn_v2pl cnn_v2xpl cnn_v3pl \
  --out sub_r2_stack10_pl || exit 1
echo "=== Phase 1A took $(((SECONDS - t) / 60))m $(((SECONDS - t) % 60))s ==="

echo "=== Phase 1B: Harsh PKS-Sim World feats5 + gbm_x5 + sim4b ==="
t=$SECONDS
python feats5.py --work ../work_sim --phys || exit 1
python gbm2.py --work ../work_sim --extra --feats4 --feats5 --tag _x5 || exit 1
python stack2.py --work ../work_sim --extra --feats4 --feats5 \
  --preds gbm_x gbm_x4 gbm_x5 cnn_v2x cnn_v2xpl \
  --out sub_r2_sim4b || exit 1
python finetune_final.py --work ../work --probs ../work/sub_r2_stack10_pl_probs.npz ../work/sub_r2_stack9_pl_probs.npz \
  ../work_sim/sub_r2_sim2_probs.npz ../work_sim/sub_r2_sim4b_probs.npz --out_linG sub_r2_blend23
echo "=== Phase 1B took $(((SECONDS - t) / 60))m $(((SECONDS - t) % 60))s ==="

echo "=== Stage 3: Harsh PKS-Sim World CNN v3 (ResNet-SE + SpecAugment + feats4 + pseudo-labels) ==="
t=$SECONDS
python cnn3.py --work ../work_sim --width 48 --epochs 28 --seeds 0 --pseudo ../work/pl_blend16_probs.npz --pseudo_thr 0.90 --tag _v3pl || exit 1
python stack2.py --work ../work_sim --extra --feats4 --feats5 \
  --preds gbm_x gbm_x4 gbm_x5 cnn_v2x cnn_v2xpl cnn_v3pl \
  --out sub_r2_sim5 || exit 1
python finetune_final.py --work ../work --probs ../work/sub_r2_stack10_pl_probs.npz ../work/sub_r2_stack9_pl_probs.npz \
  ../work_sim/sub_r2_sim4b_probs.npz ../work_sim/sub_r2_sim5_probs.npz --out_linG sub_r2_blend24_v3
echo "=== Stage 3 took $(((SECONDS - t) / 60))m $(((SECONDS - t) % 60))s ==="
