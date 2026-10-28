"""Check the impact of the missing --phys flag in ../work/feats_test.parquet vs ../work_sim/feats_test.parquet,
and re-run the clean LightGBM + clean Stacker + Cross-Domain Stacker with physically-scaled (--phys) test features!"""
from pathlib import Path
import numpy as np
import pandas as pd

from common2 import CLASSES, ROOT, labels, log, macro_f1

PULL = ROOT / 'work_pull'
DATA = ROOT / 'data'


def main():
    # 1. Compare all 21 individual base model test predictions across the 4 worlds (only needs numpy!)
    print('\n=== ALL 21 BASE MODELS: OOF F1 & PKS TEST PREDICTIONS vs CHAMPION blend16 (0.8645) ===')
    b16 = pd.read_csv(ROOT / 'submissions' / 'sub_r2_blend16.csv').label.map({c: i for i, c in enumerate(CLASSES)}).values
    y = labels(DATA)
    short = [c[:6] for c in CLASSES]
    for folder_name, folder in [('work', PULL / 'work_preds'), ('sim1', PULL / 'sim1_preds'), ('sim2', PULL / 'sim2_preds'), ('sim3', PULL / 'sim3_preds')]:
        for npz in sorted(folder.glob('*.npz')):
            d = np.load(npz)
            oof_f1, pf1 = macro_f1(y, d['oof'].argmax(1))
            te_pred = d['test'].argmax(1)
            agree = (te_pred == b16).mean() * 100
            scint_n = (te_pred == CLASSES.index('scintillating_star')).sum()
            eti_n = (te_pred == CLASSES.index('eti_candidate')).sum()
            rfi_i_n = (te_pred == CLASSES.index('rfi_intermittent')).sum()
            noise_n = (te_pred == CLASSES.index('noise')).sum()
            print(f'{folder_name:4s}/{npz.stem:11s} | OOF:{oof_f1:.4f} (eti:{pf1[0]:.3f} sci:{pf1[3]:.3f}) | '
                  f'agree b16:{agree:5.1f}% | test ETI:{eti_n:4d} Sci:{scint_n:4d} RFI_i:{rfi_i_n:4d} Noise:{noise_n:4d}')

    try:
        f_te_unscaled = pd.read_parquet(PULL / 'work_preds' / 'feats_test.parquet')
    except ImportError:
        log('pyarrow not installed yet (run: pip install pyarrow to also inspect parquet features)')
        return
    f3_te_unscaled = pd.read_parquet(PULL / 'work_preds' / 'feats3_test.parquet')
    f_te_phys = pd.read_parquet(PULL / 'sim1_preds' / 'feats_test.parquet')
    f3_te_phys = pd.read_parquet(PULL / 'sim1_preds' / 'feats3_test.parquet')

    # How many test cadences have a valid clean drifting track (track_clean_drift > 3.0) with --phys vs without --phys?
    u_cd = f_te_unscaled['track_clean_drift'].values
    p_cd = f_te_phys['track_clean_drift'].values
    missed_eti = (p_cd > 3.5) & (u_cd == -99)
    log(f'Test cadences with strong ETI track (SNR > 3.5) under --phys that were set to -99 in clean work: {missed_eti.sum()}')
    log(f'Test cadences where --phys increases track_clean_drift by > 1.5 sigma: {((p_cd - u_cd) > 1.5).sum()}')
    log(f'Test cadences where --phys increases tfit_clean by > 1.5 sigma: {((f3_te_phys["tfit_clean"].values - f3_te_unscaled["tfit_clean"].values) > 1.5).sum()}')

    f_tr = pd.read_parquet(PULL / 'work_preds' / 'feats_train.parquet')
    f3_tr = pd.read_parquet(PULL / 'work_preds' / 'feats3_train.parquet')
    f_sim_tr = pd.read_parquet(PULL / 'sim1_preds' / 'feats_train.parquet')
    f3_sim_tr = pd.read_parquet(PULL / 'sim1_preds' / 'feats3_train.parquet')

    print('\n=== FEATS2 & FEATS3 SHIFT: GBT TRAIN vs SIM1 TRAIN vs PKS TEST (--phys) ===')
    cols_check = [
        ('feats2', 'track_clean_drift', f_tr, f_sim_tr, f_te_phys),
        ('feats2', 'track_on_min', f_tr, f_sim_tr, f_te_phys),
        ('feats2', 'line_on_min', f_tr, f_sim_tr, f_te_phys),
        ('feats2', 'persist_all6', f_tr, f_sim_tr, f_te_phys),
        ('feats2', 'disp_on_max', f_tr, f_sim_tr, f_te_phys),
        ('feats2', 'row_on_max', f_tr, f_sim_tr, f_te_phys),
        ('feats2', 'patch_diff', f_tr, f_sim_tr, f_te_phys),
        ('feats2', 'steep_max', f_tr, f_sim_tr, f_te_phys),
        ('feats3', 'exc_area_diff', f3_tr, f3_sim_tr, f3_te_phys),
        ('feats3', 'exc_max_on_min', f3_tr, f3_sim_tr, f3_te_phys),
        ('feats3', 'tex4x8_diff', f3_tr, f3_sim_tr, f3_te_phys),
        ('feats3', 'tex8x16_diff', f3_tr, f3_sim_tr, f3_te_phys),
        ('feats3', 'hot4x8_diff', f3_tr, f3_sim_tr, f3_te_phys),
        ('feats3', 'tfit_clean', f3_tr, f3_sim_tr, f3_te_phys),
        ('feats3', 'tfit_ratio_min', f3_tr, f3_sim_tr, f3_te_phys),
    ]
    for src, col, tr_df, sim_df, te_df in cols_check:
        q_tr = np.percentile(tr_df[col].values, [50, 90, 95])
        q_sim = np.percentile(sim_df[col].values, [50, 90, 95])
        q_te = np.percentile(te_df[col].values, [50, 90, 95])
        print(f'{src}:{col:18s} | GBT p50/90/95: {q_tr[0]:6.2f}/{q_tr[1]:6.2f}/{q_tr[2]:6.2f} | '
              f'SIM1: {q_sim[0]:6.2f}/{q_sim[1]:6.2f}/{q_sim[2]:6.2f} | PKS: {q_te[0]:6.2f}/{q_te[1]:6.2f}/{q_te[2]:6.2f}')

    # Also compare individual base model test predictions in work_preds and sim1_preds
    print('\n=== BASE MODEL AGREEMENT WITH CHAMPION blend16 (0.8645) ON PKS TEST ===')
    b16 = pd.read_csv(ROOT / 'submissions' / 'sub_r2_blend16.csv').label.map({c: i for i, c in enumerate(CLASSES)}).values
    for folder_name, folder in [('work', PULL / 'work_preds'), ('sim1', PULL / 'sim1_preds'), ('sim2', PULL / 'sim2_preds'), ('sim3', PULL / 'sim3_preds')]:
        for npz in sorted(folder.glob('*.npz')):
            d = np.load(npz)
            oof_f1 = macro_f1(labels(DATA), d['oof'].argmax(1))[0]
            te_pred = d['test'].argmax(1)
            agree = (te_pred == b16).mean() * 100
            scint_n = (te_pred == CLASSES.index('scintillating_star')).sum()
            eti_n = (te_pred == CLASSES.index('eti_candidate')).sum()
            print(f'{folder_name:4s}/{npz.stem:12s} | OOF F1: {oof_f1:.4f} | agree w/ b16: {agree:5.1f}% | pred ETI: {eti_n:4d} | pred Scint: {scint_n:4d}')


if __name__ == '__main__':
    main()
