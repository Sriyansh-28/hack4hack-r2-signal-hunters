# Signal Hunters — Hack4Hack 2026 Round 2

Classifies SETI ON/OFF cadences (6 panels × 16 time rows × 128 channels) into 9 labels; metric macro-F1.
Training data is from GBT, test data from PKS (noisier, different bandpass and label mix).

## Method
1. **Cleaning (`prep2.py`)** — dropped rows (value 0) → missing; per panel, subtract a smooth bandpass (running
   median over channels of the time-median spectrum, so narrow lines survive); per-panel robust z-score (fixes gain
   changes and PKS's higher noise); blank the GBT-only centre channel 64. Stored as int8 in 0.25σ steps.
   5 stratified folds (seed 42).
2. **Physics features (`feats2.py`, ~60)** — drift-line search per panel; a **global-time track search**
   (panel *p*, row *r* at `p × (16 + slew_gap_rows) + r`) for the ETI test (on in all ON panels, absent in OFF,
   one drifting track); steep lines (satellite/radar); de-dispersion (pulsar/FRB); ON vs OFF patchiness
   (scintillation); persistent lines (RFI). Telescope-specific metadata (`tsamp_s`, `channel_width_hz`,
   `center_freq_mhz`, `target`) is **not** used.
   **Extra features (`feats3.py`)** — scintillation measured against a bandpass taken from the OFF panels of the
   same cadence (keeps broad ON-only brightening that per-panel flattening removes): spectral excess and texture at
   5 block sizes, ON vs OFF; and an ETI track-fit check (track SNR / best local line in each ON panel, drift
   mismatch, OFF lines near the track) against the "same drift, not connected" trap.
3. **LightGBM on the features (`gbm2.py`)**, class-balanced.
4. **CNN (`cnn2.py`)** — input = the global-time image (126 × 128, ON/OFF marker channel); v2 adds a stacked-panel
   branch (6 panels as channels) and the features. Class-balanced loss (test label mix differs), label smoothing,
   PKS-like augmentation (frequency compression 0.55–1 for the ~0.64× drift slopes, flips, gain/noise jitter,
   dropped rows, frequency shifts), flip TTA. Optional **pseudo-labels**: test cadences the stack predicts with
   ≥ 0.9 probability are added to training (no hand labelling).
5. **Stacker (`stack2.py`)** — logistic regression + LightGBM on the models' out-of-fold log-probabilities
   (+10 physics features), same folds, averaged. No prior/bias tuning.

## Results (5-fold out-of-fold macro-F1 on train)
| File | Models | OOF macro-F1 |
|---|---|---|
| `sub_r2_cnn.csv` | cnn | 0.8494 |
| `sub_r2_stack.csv` | cnn, gbm, cnn_v2 | 0.8896 |
| `sub_r2_stack2.csv` | cnn, gbm, cnn_v2, cnn_v2s1 | 0.8942 |
| `sub_r2_stack3.csv` | + cnn_v2s3, cnn_v2notab | 0.8984 |
| `sub_r2_stack_pl.csv` | + cnn_v2pl (pseudo-labels) | 0.8996 (public 0.8585) |
| `sub_r2_stack4.csv` | + gbm_x (feats3 features), stacker sees feats3 | 0.9083 |
| `sub_r2_stack4_pl.csv` | stack4 + cnn_v2pl | 0.9098 |

## Reproduce
Layout: `R2/code` (this repo), `R2/data` (the competition files), `R2/work` and `R2/submissions` (created).
Python ≥ 3.10; `pip install -r requirements.txt`; one GPU (the CNNs take ~18 min each on an A100 MIG slice).
Run in this order (each script `cd`s to its own folder):

```bash
bash run_cnn.sh       # prep2.py (cleaning, folds) + CNN v1          -> submissions/sub_r2_cnn.csv
bash run_feats.sh     # feats2.py + gbm2.py + blend                  -> submissions/sub_r2_mix.csv
bash run_v2.sh        # CNN v2 seed 0 + stacker                      -> submissions/sub_r2_stack.csv
bash run_v2s1.sh      # CNN v2 seed 1
bash run_stack2.sh    # stacker                                      -> submissions/sub_r2_stack2.csv
bash run_final.sh     # CNN v2 + pseudo-labels, CNN v2 without features, stackers -> sub_r2_stack3.csv, sub_r2_stack_pl.csv
bash run_v2s3.sh      # CNN v2 seed 3, stackers again                -> sub_r2_stack3.csv, sub_r2_stack_pl.csv
bash run_x.sh         # feats3.py extra features, LightGBM, CNN v2 + extra features, stackers -> sub_r2_stack4*/5*.csv
bash run_xpl.sh       # CNN v2 + extra features + pseudo-labels from stack4, stacker -> sub_r2_stack6_pl.csv
bash run_diag.sh      # PKS vs GBT diagnostics (diag.py) -> ../diag.txt
bash run_sim.sh       # PKS-like pipeline in ../work_sim -> sub_r2_sim1.csv, sub_r2_sim2.csv
bash run_xval.sh      # calibration: clean-trained LightGBM scored on PKS-like features
bash run_blend.sh     # stack6_pl + sim2 -> sub_r2_blend.csv (public 0.8625)
bash run_diag2.sh; bash run_diag3.sh   # ON-panel check; glitches / dispersion / EM label mix
bash run_sim2.sh      # PKS-like v2 (milder noise, glitches, fine de-dispersion) in ../work_sim2 -> simb1, simb2, blend2
bash run_blend3.sh    # stack6_pl + sim2 + simb2 -> sub_r2_blend3.csv
bash run_pl2.sh       # pseudo-labels from the best blend -> simb3, stack7_pl -> sub_r2_blend4.csv
bash run_blend6.sh    # CPU, ~1 min: blends pushed toward the harsh simulation -> sub_r2_blend6/7/8.csv
bash run_sim3.sh      # harsher PKS-like world (noise x1.35-1.75) in ../work_sim3 -> sim3a/sim3b, blend9-12
bash run_blend13.sh   # CPU, ~1 min: clean : PKS-like = 1:1 with 3 clean stacks -> sub_r2_blend13/14.csv
bash run_bias.sh      # CPU, ~1 min: blend8 + per-class decision bias (bias2.py, tuned on OOF, x0.5) -> sub_r2_blend15.csv
python finetune_final.py --work ../data --probs ../work/sub_r2_stack6_pl_probs.npz ../work/sub_r2_stack_pl_probs.npz \
  ../work_sim/sub_r2_sim1_probs.npz ../work_sim/sub_r2_sim2_probs.npz   # 50/50 linear + log-odds (1.2:0.8:0.8:1.2) -> sub_r2_blend16.csv
```

**Final submission: `sub_r2_blend8.csv`** (`run_blend6.sh`, average of `stack6_pl`, `stack_pl`, `sim1` and `sim2`, i.e.
clean-validated stacks : PKS-like stacks = 1 : 1; public **0.8639**, OOF 0.9036) and its calibrated 50/50 linear + log-odds
refinement **`sub_r2_blend16.csv`** (`finetune_final.py`, weights 1.2 : 0.8 : 0.8 : 1.2; OOF **0.9049**). To reproduce, run the
steps above in order up to `run_blend6.sh` / `finetune_final.py` (uses the stacks from `run_final.sh`, `run_v2s3.sh`,
`run_x.sh`, `run_xpl.sh` and `run_sim.sh`). Earlier best: `sub_r2_blend.csv` (`run_blend.sh`, stack6_pl + sim2,
public 0.8625). Variants that did not beat it: `sub_r2_blend14.csv` 0.8634 (3 clean : 3 PKS-like stacks incl. sim3b),
`sub_r2_blend3.csv` 0.8591 (milder simulation), `sub_r2_blend6.csv` 0.8565 (stack6_pl : sim2 = 1 : 2);
`sub_r2_blend15.csv` (decision bias, 0.9044 OOF) was not uploaded; `stack7_pl` OOF 0.9105 vs `stack6_pl` 0.9110.

| Upload | Public macro-F1 |
|---|---|
| `sub_r2_stack_pl.csv` | 0.8585 |
| `sub_r2_stack6_pl.csv` | 0.8576 |
| `sub_r2_blend.csv` | 0.8625 |
| `sub_r2_blend3.csv` | 0.8591 |
| `sub_r2_blend6.csv` | 0.8565 |
| **`sub_r2_blend8.csv`** | **0.8639** |
| `sub_r2_blend14.csv` | 0.8634 |

### Domain shift (GBT → PKS) and the PKS-like pipeline
`diag.py` showed: PKS raw noise is 1.31× GBT (signals look ~0.7× as strong after normalisation; the 90th percentile of
line / track / persistent-line SNRs is 0.69–0.72× on test), and drift in pixels per row is
(16 s / 3.81 Hz) / (18.25 s / 2.79 Hz) = 0.64× (ETI track slope median 0.10 → 0.05). Models validated on GBT
gained +0.012 OOF but nothing on the public board. `run_sim.sh` therefore trains on PKS-like train cadences
(`prep2.py --sim`: noise ×1.15–1.45 in log units, extra dropped rows) and measures drift in physical units
(`feats2.py / feats3.py --phys`: slope grids × tsamp / channel width relative to GBT).
On the NITJ HPC every step goes through PBS: `bash submit2.sh <script> <walltime> [cpu|after:JOBID]`
(`job2.pbs` takes one free MIG slice). Seeds are fixed; GPU kernels can still make tiny run-to-run differences.

## Files
`common2.py` paths/classes/metric · `prep2.py` cleaning · `feats2.py`, `feats3.py` physics & OFF-bandpass features ·
`gbm2.py` LightGBM · `cnn2.py` CNN · `ens2.py` weighted blend · `stack2.py` stacker · `blend.py` cross-domain blend ·
`bias2.py`, `finetune_final.py` OOF/log-odds post-processing · `diag*.py`, `xval.py` domain diagnostics ·
`run_*.sh` step scripts · `job2.pbs`, `submit2.sh` PBS.
