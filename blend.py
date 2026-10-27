"""Average the test probabilities of stacks from different pipelines (e.g. the clean-validated stack6_pl and the
PKS-like sim2) -> submissions/<out>.csv. Their OOFs live in different domains, so weights are given, not fitted."""
from pathlib import Path

import numpy as np
import pandas as pd

from common2 import CLASSES, ROOT, log, parser


def main():
    p = parser(__doc__)
    p.add_argument('--probs', nargs='+', required=True, help='*_probs.npz files written by stack2.py')
    p.add_argument('--weights', nargs='*', type=float, default=None)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    w = a.weights or [1.0] * len(a.probs)
    tests = [np.load(f)['test'] for f in a.probs]
    test = sum(wi * t for wi, t in zip(w, tests)) / sum(w)
    work = Path(a.work)
    sub = pd.DataFrame({'id': pd.read_csv(work / 'test_meta.csv').id, 'label': [CLASSES[i] for i in test.argmax(1)]})
    ss = pd.read_csv(work / 'sample_submission.csv')
    sub = ss[['id']].merge(sub, on='id', how='left')
    out = ROOT / 'submissions' / f'{a.out}.csv'
    sub.to_csv(out, index=False)
    np.savez(work / f'{a.out}_probs.npz', test=test)                   # e.g. as a pseudo-label source for cnn2.py
    ok = (len(sub) == len(ss) and sub.id.is_unique and sub.label.notna().all() and sub.label.isin(CLASSES).all())
    log(f'wrote {out} ({len(sub)} rows); CHECK {a.out}: ' + ('OK, valid for upload' if ok else 'FAILED, do not upload'))
    for f, t in zip(a.probs, tests):
        log(f'labels different from {Path(f).name}: {(t.argmax(1) != test.argmax(1)).sum()}')
    print(sub.label.value_counts().to_string(), flush=True)


if __name__ == '__main__':
    main()
