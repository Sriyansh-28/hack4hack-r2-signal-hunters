"""Repair the missing --phys bug in the clean world (../work):
In ../work, feats2.py and feats3.py were run without --phys (scale=1.0 instead of 0.642 on PKS test),
which set track_clean_drift = -99 on 207 PKS test cadences with strong drifting tracks (SNR > 3.5) and
understated track_clean_drift / tfit_clean by > 1.5 sigma on 423 PKS test cadences!

Because norm_test.npy is identical between ../work and ../work_sim, sim1_preds/feats_test.parquet and
sim1_preds/feats3_test.parquet are the exact --phys test features for ../work.
This script:
1. Retrains gbm and gbm_x on clean train features and evaluates them on the --phys test features.
2. Re-runs the 5-fold Logistic + LightGBM stackers (stack5_phys, stack6_pl_phys, stack7_pl_phys) with the
   repaired gbm_x_phys and repaired META + META3 test features.
3. Writes sub_r2_blend27.csv (the repaired version of blend25, 0.8648 public) and sub_r2_blend28.csv.
"""
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common2 import CLASSES, ROOT, labels, log, macro_f1, report

warnings.filterwarnings('ignore')

PULL = ROOT / 'work_pull'
W_PREDS = PULL / 'work_preds'
S1_PREDS = PULL / 'sim1_preds'
DATA = ROOT / 'data'

META = ['track_clean', 'track_clean_drift', 'track_on_min', 'track_off_max', 'line_on_min', 'line_off_max',
        'persist_all6', 'disp_on_max', 'patch_diff', 'steep_max']
META3 = ['exc_area_diff', 'exc_max_on_min', 'tex4x8_diff', 'tex8x16_diff', 'hot4x8_diff', 'tfit_ratio_min',
         'tfit_dslope_max', 'tfit_off_near']


def make_folds(y):
    from sklearn.model_selection import StratifiedKFold
    folds = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y)):
        folds[va] = k
    return folds


def train_gbm(X, Xt, y, folds, seeds=(0, 1, 2), name='gbm'):
    K = len(CLASSES)
    oof, test = np.zeros((len(y), K)), np.zeros((len(Xt), K))
    for seed in seeds:
        for k in range(folds.max() + 1):
            t, v = folds != k, folds == k
            m = lgb.LGBMClassifier(objective='multiclass', n_estimators=4000, learning_rate=0.03, num_leaves=31,
                                   min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                                   reg_lambda=2.0, class_weight='balanced', random_state=seed, n_jobs=4, verbose=-1)
            m.fit(X[t], y[t], eval_set=[(X[v], y[v])], callbacks=[lgb.early_stopping(150, verbose=False)])
            oof[v] += m.predict_proba(X[v]) / len(seeds)
            test += m.predict_proba(Xt) / (len(seeds) * (folds.max() + 1))
    report(y, oof, f'{name} OOF')
    return oof.astype(np.float32), test.astype(np.float32)


def run_stack(pred_dict, model_names, ftr, fte, y, folds, name):
    X = np.log(np.clip(np.concatenate([pred_dict[m][0] for m in model_names], 1), 1e-7, 1))
    Xt = np.log(np.clip(np.concatenate([pred_dict[m][1] for m in model_names], 1), 1e-7, 1))
    meta_cols = [c for c in META + META3 if c in ftr.columns]
    Xg, Xgt = np.hstack([X, ftr[meta_cols].values]), np.hstack([Xt, fte[meta_cols].values])
    K = len(CLASSES)
    lo, lt, go, gt = (np.zeros((len(y), K)), np.zeros((len(Xt), K)), np.zeros((len(y), K)), np.zeros((len(Xt), K)))
    for k in range(folds.max() + 1):
        t, v = folds != k, folds == k
        m = LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced').fit(X[t], y[t])
        lo[v] = m.predict_proba(X[v])
        lt += m.predict_proba(Xt) / (folds.max() + 1)
        g = lgb.LGBMClassifier(objective='multiclass', n_estimators=2000, learning_rate=0.03, num_leaves=15,
                               min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                               reg_lambda=5.0, class_weight='balanced', random_state=k, n_jobs=4, verbose=-1)
        g.fit(Xg[t], y[t], eval_set=[(Xg[v], y[v])], callbacks=[lgb.early_stopping(100, verbose=False)])
        go[v] = g.predict_proba(Xg[v])
        gt += g.predict_proba(Xgt) / (folds.max() + 1)
    oof, test = (lo + go) / 2, (lt + gt) / 2
    report(y, oof, f'{name} STACK')
    return oof, test


def save_sub(pred, ref_pred, name):
    sub = pd.DataFrame({'id': pd.read_csv(DATA / 'test_meta.csv').id, 'label': [CLASSES[i] for i in pred]})
    ss = pd.read_csv(DATA / 'sample_submission.csv')
    sub = ss[['id']].merge(sub, on='id', how='left')
    out = ROOT / 'submissions' / f'{name}.csv'
    sub.to_csv(out, index=False)
    ok = (len(sub) == len(ss) and sub.id.is_unique and sub.label.notna().all() and sub.label.isin(CLASSES).all())
    log(f'wrote {out}; CHECK {name}: ' + ('OK' if ok else 'FAILED') + f' | diff vs blend25 (0.8648): {(pred != ref_pred).sum()}')


def main():
    y = labels(DATA)
    folds = make_folds(y)

    # Load clean train features (GBT, scale=1.0) and FIXED --phys test features (PKS, scale=0.642)
    ftr2 = pd.read_parquet(W_PREDS / 'feats_train.parquet')
    ftr3 = pd.read_parquet(W_PREDS / 'feats3_train.parquet')
    fte2_phys = pd.read_parquet(S1_PREDS / 'feats_test.parquet')[ftr2.columns]
    fte3_phys = pd.read_parquet(S1_PREDS / 'feats3_test.parquet')[ftr3.columns]

    ftr = pd.concat([ftr2, ftr3], axis=1)
    fte_phys = pd.concat([fte2_phys, fte3_phys], axis=1)

    # Load existing base models from work_preds
    preds = {}
    for f in sorted(W_PREDS.glob('*.npz')):
        d = np.load(f)
        preds[f.stem] = (d['oof'], d['test'])

    log('Step 1: Retraining gbm and gbm_x with --phys test features...')
    oof_gbm, test_gbm_phys = train_gbm(ftr2.values.astype(np.float32), fte2_phys.values.astype(np.float32), y, folds, name='gbm_phys')
    oof_gbmx, test_gbmx_phys = train_gbm(ftr.values.astype(np.float32), fte_phys.values.astype(np.float32), y, folds, name='gbm_x_phys')

    log(f'gbm_x test labels changed by --phys fix: {(preds["gbm_x"][1].argmax(1) != test_gbmx_phys.argmax(1)).sum()}')
    preds['gbm'] = (oof_gbm, test_gbm_phys)
    preds['gbm_x'] = (oof_gbmx, test_gbmx_phys)

    log('Step 2: Re-running clean stackers with --phys test features and repaired gbm/gbm_x...')
    base_m = ['cnn', 'gbm', 'gbm_x', 'cnn_v2', 'cnn_v2s1', 'cnn_v2s3', 'cnn_v2notab']
    s4_oof, s4_test = run_stack(preds, base_m, ftr, fte_phys, y, folds, 'stack4_phys')
    s4pl_oof, s4pl_test = run_stack(preds, base_m + ['cnn_v2pl'], ftr, fte_phys, y, folds, 'stack4_pl_phys')
    s5_oof, s5_test = run_stack(preds, base_m + ['cnn_v2x'], ftr, fte_phys, y, folds, 'stack5_phys')
    s5pl_oof, s5pl_test = run_stack(preds, base_m + ['cnn_v2x', 'cnn_v2pl'], ftr, fte_phys, y, folds, 'stack5_pl_phys')
    s6pl_oof, s6pl_test = run_stack(preds, base_m + ['cnn_v2x', 'cnn_v2pl', 'cnn_v2xpl'], ftr, fte_phys, y, folds, 'stack6_pl_phys')
    s7pl_oof, s7pl_test = run_stack(preds, base_m + ['cnn_v2x', 'cnn_v2pl', 'cnn_v2xpl', 'cnn_v2xpl3'], ftr, fte_phys, y, folds, 'stack7_pl_phys')

    # Save repaired stack probabilities so HPC or subsequent scripts can use them
    np.savez(PULL / 'sub_r2_stack6_pl_phys_probs.npz', oof=s6pl_oof, test=s6pl_test)
    np.savez(PULL / 'sub_r2_stack7_pl_phys_probs.npz', oof=s7pl_oof, test=s7pl_test)
    np.savez(PULL / 'sub_r2_stack5_phys_probs.npz', oof=s5_oof, test=s5_test)

    old_s6 = np.load(PULL / 'sub_r2_stack6_pl_probs.npz')['test']
    log(f'stack6_pl test labels changed by --phys fix: {(old_s6.argmax(1) != s6pl_test.argmax(1)).sum()}')

    # Load simulated stacks
    sim1 = np.load(PULL / 'sub_r2_sim1_probs.npz')
    sim2 = np.load(PULL / 'sub_r2_sim2_probs.npz')
    simb2 = np.load(PULL / 'sub_r2_simb2_probs.npz')
    simb3 = np.load(PULL / 'sub_r2_simb3_probs.npz')

    # Reference blend25 (0.8648)
    ref_b25 = pd.read_csv(ROOT / 'submissions' / 'sub_r2_blend25.csv').label.map({c: i for i, c in enumerate(CLASSES)}).values

    # Build sub_r2_blend27: exact blend25 recipe but using the --phys repaired clean stacks!
    spec_27 = [
        (s6pl_oof, s6pl_test, 1.6),
        (s7pl_oof, s7pl_test, 1.2),
        (s5_oof, s5_test, 0.9),
        (s4_oof, s4_test, 0.5),
        (s4pl_oof, s4pl_test, 0.4),
        (s5pl_oof, s5pl_test, 0.4),
        (sim2['oof'], sim2['test'], 2.2),
        (sim1['oof'], sim1['test'], 1.4),
        (simb3['oof'], simb3['test'], 0.9),
        (simb2['oof'], simb2['test'], 0.5),
    ]
    w27 = sum(w for _, _, w in spec_27)
    oof_lin = sum(w * o for o, _, w in spec_27) / w27
    test_lin = sum(w * t for _, t, w in spec_27) / w27
    oof_geo = np.exp(sum(w * np.log(np.clip(o, 1e-6, 1)) for o, _, w in spec_27) / w27)
    oof_geo /= oof_geo.sum(1, keepdims=True)
    test_geo = np.exp(sum(w * np.log(np.clip(t, 1e-6, 1)) for _, t, w in spec_27) / w27)
    test_geo /= test_geo.sum(1, keepdims=True)
    oof_27 = 0.5 * oof_lin + 0.5 * oof_geo
    test_27 = 0.5 * test_lin + 0.5 * test_geo

    log(f'blend27 (--phys repaired 10-stack 1:1 linear+log-odds) OOF F1: {macro_f1(y, oof_27.argmax(1))[0]:.4f}')
    save_sub(test_27.argmax(1), ref_b25, 'sub_r2_blend27')

    print('\nClass distribution comparison (blend25 0.8648 vs blend27 --phys repaired):')
    print(pd.DataFrame({
        'blend25 (0.8648)': pd.Series(ref_b25).map(dict(enumerate(CLASSES))).value_counts(),
        'blend27 (--phys fix)': pd.Series(test_27.argmax(1)).map(dict(enumerate(CLASSES))).value_counts(),
    }).fillna(0).astype(int).to_string())


if __name__ == '__main__':
    main()
