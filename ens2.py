"""Step 5 - combine models and write the submission.

Weighted average of the models' probabilities, weights chosen on out-of-fold log-loss. No prior-based bias
tuning: the test telescope has a different label mix, and the models already use class-balanced training.
Writes ROOT/submissions/<out>.csv (checked against sample_submission).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common2 import CLASSES, ROOT, labels, log, parser, report


def main():
    p = parser(__doc__)
    p.add_argument('--preds', nargs='*', default=None)
    p.add_argument('--out', default='sub_r2_v1')
    a = p.parse_args()
    work = Path(a.work)
    files = sorted((work / 'preds').glob('*.npz'))
    if a.preds:
        files = [f for f in files if f.stem in a.preds]
    d = [np.load(f) for f in files]
    O, T = np.stack([x['oof'] for x in d]), np.stack([x['test'] for x in d])
    y = labels(work)
    for f, o in zip(files, O):
        report(y, o, f'{f.stem:12s}')
    ll = lambda z: -np.log(np.clip(np.tensordot(np.exp(z) / np.exp(z).sum(), O, 1)[np.arange(len(y)), y], 1e-7, 1)).mean()
    z = minimize(ll, np.zeros(len(O)), method='Nelder-Mead').x if len(O) > 1 else np.zeros(1)
    w = np.exp(z) / np.exp(z).sum()
    log('weights: ' + ', '.join(f'{f.stem}={x:.2f}' for f, x in zip(files, w)))
    oof, test = np.tensordot(w, O, 1), np.tensordot(w, T, 1)
    report(y, oof, 'ENSEMBLE')
    meta = pd.read_csv(work / 'test_meta.csv')
    sub = pd.DataFrame({'id': meta.id, 'label': [CLASSES[i] for i in test.argmax(1)]})
    ss = pd.read_csv(work / 'sample_submission.csv')
    if len(ss) == len(sub) and set(ss.id) == set(sub.id):
        sub = ss[['id']].merge(sub, on='id', how='left')
    else:
        log(f'WARNING: {len(sub)} rows vs {len(ss)} in sample_submission (debug run?) - do not upload')
    out = ROOT / 'submissions' / f'{a.out}.csv'
    out.parent.mkdir(exist_ok=True)
    sub.to_csv(out, index=False)
    np.savez(work / f'{a.out}_probs.npz', oof=oof, test=test)
    log(f'wrote {out} ({len(sub)} rows)')
    print(sub.label.value_counts().to_string(), flush=True)


if __name__ == '__main__':
    main()
