"""Calibration (CPU, ~4 min): is the PKS-like simulation (work_sim) harder or easier than real PKS?
LightGBM trained on CLEAN GBT features (work, folds != k) is scored on the PKS-LIKE features of the held-out cadences
(work_sim, fold k). On real PKS our clean-trained stack lost ~0.05 (OOF 0.911 -> public 0.858). If clean-trained
LightGBM loses much more here (clean OOF 0.8211), the simulation is too harsh."""
from pathlib import Path

import lightgbm as lgb
import numpy as np

from common2 import CLASSES, N_THREADS, ROOT, labels, load_feats, log, parser, report


def main():
    p = parser(__doc__)
    p.add_argument('--sim', default=str(ROOT / 'work_sim'))
    a = p.parse_args()
    work, sim = Path(a.work), Path(a.sim)
    ftr, _ = load_feats(work, True)
    fsim, _ = load_feats(sim, True)
    X, Xs = ftr.values.astype(np.float32), fsim[ftr.columns].values.astype(np.float32)
    y = labels(work)
    folds = np.load(work / 'folds.npy')
    oc, os_ = np.zeros((len(y), len(CLASSES))), np.zeros((len(y), len(CLASSES)))
    for k in range(folds.max() + 1):
        t, v = folds != k, folds == k
        m = lgb.LGBMClassifier(objective='multiclass', n_estimators=4000, learning_rate=0.03, num_leaves=31,
                               min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                               reg_lambda=2.0, class_weight='balanced', random_state=0, n_jobs=N_THREADS, verbose=-1)
        m.fit(X[t], y[t], eval_set=[(X[v], y[v])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oc[v], os_[v] = m.predict_proba(X[v]), m.predict_proba(Xs[v])
        log(f'fold {k} done')
    report(y, oc, 'XVAL clean-trained, clean validation')
    report(y, os_, 'XVAL clean-trained, PKS-like validation')


if __name__ == '__main__':
    main()
