"""Per-class decision bias for macro-F1: tuned on the out-of-fold probabilities of a blend of stacks (coordinate
ascent on log-probability offsets), shrunk (default x0.5, the OOF mixes the clean and PKS-like worlds), then applied to
the same blend's test probabilities -> ROOT/submissions/<out>.csv. Needs *_probs.npz files from stack2.py (oof + test);
blend.py outputs have no oof and cannot be used here."""
from pathlib import Path

import numpy as np
import pandas as pd

from common2 import CLASSES, ROOT, labels, log, macro_f1, parser


def tune(lo, y, grid, passes=3):
    b = np.zeros(lo.shape[1])
    best = macro_f1(y, lo.argmax(1))[0]
    for _ in range(passes):
        for k in range(len(b)):
            for g in grid:
                c = b.copy()
                c[k] = g
                s = macro_f1(y, (lo + c).argmax(1))[0]
                if s > best + 1e-6:
                    best, b = s, c
    return b - b.mean(), best


def main():
    p = parser(__doc__)
    p.add_argument('--probs', nargs='+', required=True, help='*_probs.npz files written by stack2.py')
    p.add_argument('--weights', nargs='*', type=float, default=None)
    p.add_argument('--shrink', type=float, default=0.5)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    w = a.weights or [1.0] * len(a.probs)
    work = Path(a.work)
    y = labels(work)
    d = [np.load(f) for f in a.probs]
    ok = True
    for f, x in zip(a.probs, d):                         # sanity: rows aligned with work/train_meta.csv?
        s = macro_f1(y, x['oof'].argmax(1))[0]
        log(f'{Path(f).name}: OOF macro-F1 {s:.4f}')
        ok &= s > 0.5
    oof = sum(wi * x['oof'] for wi, x in zip(w, d)) / sum(w)
    test = sum(wi * x['test'] for wi, x in zip(w, d)) / sum(w)
    lo, lt = np.log(np.clip(oof, 1e-7, 1)), np.log(np.clip(test, 1e-7, 1))
    s0, per0 = macro_f1(y, lo.argmax(1))
    b, s1 = tune(lo, y, np.linspace(-1.5, 1.5, 61))
    bs = a.shrink * b
    s2, per2 = macro_f1(y, (lo + bs).argmax(1))
    log(f'blend OOF macro-F1 {s0:.4f} -> full bias {s1:.4f} -> bias x{a.shrink} {s2:.4f}')
    print('bias   ' + '  '.join(f'{c[:8]}={v:+.2f}' for c, v in zip(CLASSES, bs)), flush=True)
    print('F1 old ' + '  '.join(f'{c[:8]}={v:.3f}' for c, v in zip(CLASSES, per0)), flush=True)
    print('F1 new ' + '  '.join(f'{c[:8]}={v:.3f}' for c, v in zip(CLASSES, per2)), flush=True)
    pred, base = (lt + bs).argmax(1), lt.argmax(1)
    sub = pd.DataFrame({'id': pd.read_csv(work / 'test_meta.csv').id, 'label': [CLASSES[i] for i in pred]})
    ss = pd.read_csv(work / 'sample_submission.csv')
    sub = ss[['id']].merge(sub, on='id', how='left')
    out = ROOT / 'submissions' / f'{a.out}.csv'
    sub.to_csv(out, index=False)
    pb = np.exp(lt + bs)
    np.savez(work / f'{a.out}_probs.npz', test=pb / pb.sum(1, keepdims=True))
    ok &= (len(sub) == len(ss) and sub.id.is_unique and sub.label.notna().all() and sub.label.isin(CLASSES).all())
    log(f'wrote {out} ({len(sub)} rows); CHECK {a.out}: ' + ('OK, valid for upload' if ok else 'FAILED, do not upload'))
    log(f'labels different from the unbiased blend: {(pred != base).sum()}')
    print(pd.DataFrame({'unbiased': pd.Series(base).map(dict(enumerate(CLASSES))).value_counts(),
                        'biased': sub.label.value_counts()}).fillna(0).astype(int).to_string(), flush=True)


if __name__ == '__main__':
    main()
