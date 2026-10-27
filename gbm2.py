"""Step 4 - LightGBM on the physics features (class-balanced, 5 folds x seeds). Output: WORK/preds/gbm.npz"""
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from common2 import CLASSES, N_THREADS, labels, load_feats, log, parser, report, save_preds

warnings.filterwarnings('ignore', message='.*eval_set.*')


def main():
    p = parser(__doc__)
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--extra', action='store_true', help='also use the feats3.py features')
    p.add_argument('--tag', default='')
    a = p.parse_args()
    work = Path(a.work)
    tr, te = load_feats(work, a.extra)
    X, Xt = tr.values.astype(np.float32), te.values.astype(np.float32)
    y = labels(work)
    folds = np.load(work / 'folds.npy')
    seeds = [int(s) for s in a.seeds.split(',')]
    oof, test = np.zeros((len(y), len(CLASSES))), np.zeros((len(Xt), len(CLASSES)))
    imp = np.zeros(X.shape[1])
    for seed in seeds:
        for k in range(folds.max() + 1):
            t, v = folds != k, folds == k
            m = lgb.LGBMClassifier(objective='multiclass', n_estimators=4000, learning_rate=0.03, num_leaves=31,
                                   min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                                   reg_lambda=2.0, class_weight='balanced', random_state=seed, n_jobs=N_THREADS,
                                   verbose=-1)
            m.fit(X[t], y[t], eval_set=[(X[v], y[v])], callbacks=[lgb.early_stopping(150, verbose=False)])
            oof[v] += m.predict_proba(X[v]) / len(seeds)
            test += m.predict_proba(Xt) / (len(seeds) * (folds.max() + 1))
            imp += m.booster_.feature_importance('gain')
        log(f'seed {seed} done')
    save_preds(work, f'gbm{a.tag}', oof, test, report(y, oof, f'gbm{a.tag} FINAL OOF'))
    print(pd.Series(imp, tr.columns).sort_values(ascending=False).head(20).round(0).to_string(), flush=True)


if __name__ == '__main__':
    main()
