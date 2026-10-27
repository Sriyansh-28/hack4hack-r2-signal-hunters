"""Step 6 - stacker: logistic regression + LightGBM trained on the models' out-of-fold log-probabilities
(+ a few physics features for LightGBM), same 5 folds, averaged. No prior-based bias tuning (test label mix
differs). Writes ROOT/submissions/<out>.csv. Round 1 lesson: stackers beat weighted blends."""
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common2 import CLASSES, N_THREADS, ROOT, labels, load_feats, log, parser, report

warnings.filterwarnings('ignore', message='.*eval_set.*')
META = ['track_clean', 'track_clean_drift', 'track_on_min', 'track_off_max', 'line_on_min', 'line_off_max',
        'persist_all6', 'disp_on_max', 'patch_diff', 'steep_max']
META3 = ['exc_area_diff', 'exc_max_on_min', 'tex4x8_diff', 'tex8x16_diff', 'hot4x8_diff', 'tfit_ratio_min',
         'tfit_dslope_max', 'tfit_off_near']                     # feats3.py (--extra)


def main():
    p = parser(__doc__)
    p.add_argument('--preds', nargs='*', default=None)
    p.add_argument('--out', default='sub_r2_stack')
    p.add_argument('--extra', action='store_true', help='also give the LightGBM stacker the feats3.py features')
    a = p.parse_args()
    work = Path(a.work)
    files = sorted((work / 'preds').glob('*.npz'))
    if a.preds:
        files = [f for f in files if f.stem in a.preds]
    d = [np.load(f) for f in files]
    log('models: ' + ' '.join(f.stem for f in files))
    X = np.log(np.clip(np.concatenate([x['oof'] for x in d], 1), 1e-7, 1))
    Xt = np.log(np.clip(np.concatenate([x['test'] for x in d], 1), 1e-7, 1))
    y = labels(work)
    folds = np.load(work / 'folds.npy')
    ftr, fte = load_feats(work, a.extra)
    meta = [c for c in META + (META3 if a.extra else []) if c in ftr.columns]
    Xg, Xgt = np.hstack([X, ftr[meta].values]), np.hstack([Xt, fte[meta].values])
    K = len(CLASSES)
    lo, lt, go, gt = (np.zeros((len(y), K)), np.zeros((len(Xt), K)), np.zeros((len(y), K)), np.zeros((len(Xt), K)))
    for k in range(folds.max() + 1):
        t, v = folds != k, folds == k
        m = LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced').fit(X[t], y[t])
        lo[v] = m.predict_proba(X[v])
        lt += m.predict_proba(Xt) / (folds.max() + 1)
        g = lgb.LGBMClassifier(objective='multiclass', n_estimators=2000, learning_rate=0.03, num_leaves=15,
                               min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                               reg_lambda=5.0, class_weight='balanced', random_state=k, n_jobs=N_THREADS, verbose=-1)
        g.fit(Xg[t], y[t], eval_set=[(Xg[v], y[v])], callbacks=[lgb.early_stopping(100, verbose=False)])
        go[v] = g.predict_proba(Xg[v])
        gt += g.predict_proba(Xgt) / (folds.max() + 1)
    report(y, lo, 'logistic stacker')
    report(y, go, 'LightGBM stacker')
    oof, test = (lo + go) / 2, (lt + gt) / 2
    report(y, oof, 'STACK (average)')
    cm = np.bincount(y * K + oof.argmax(1), minlength=K * K).reshape(K, K)
    print('confusion (rows = true, cols = predicted):\n' + pd.DataFrame(cm, index=CLASSES, columns=[c[:5] for c in CLASSES]).to_string(), flush=True)
    meta_te = pd.read_csv(work / 'test_meta.csv')
    sub = pd.DataFrame({'id': meta_te.id, 'label': [CLASSES[i] for i in test.argmax(1)]})
    ss = pd.read_csv(work / 'sample_submission.csv')
    if len(ss) == len(sub) and set(ss.id) == set(sub.id):
        sub = ss[['id']].merge(sub, on='id', how='left')
    else:
        log(f'WARNING: {len(sub)} rows vs {len(ss)} in sample_submission (debug run?) - do not upload')
    out = ROOT / 'submissions' / f'{a.out}.csv'
    sub.to_csv(out, index=False)
    np.savez(work / f'{a.out}_probs.npz', oof=oof, test=test)
    log(f'wrote {out} ({len(sub)} rows)')
    print(sub.label.value_counts().to_string(), flush=True)
    ok = (len(sub) == len(ss) and sub.id.is_unique and set(sub.id) == set(ss.id)
          and sub.label.notna().all() and sub.label.isin(CLASSES).all())
    log(f'CHECK {a.out}: ' + ('OK, valid for upload' if ok else 'FAILED, do not upload'))
    ref = ROOT / 'submissions' / 'sub_r2_stack2.csv'
    if ok and ref.exists() and ref != out:
        log(f'labels different from sub_r2_stack2.csv: {(pd.read_csv(ref).label.values != sub.label.values).sum()}')


if __name__ == '__main__':
    main()
