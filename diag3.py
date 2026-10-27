"""Diagnostics 3 (~1 min): 50% of PKS test cadences have a strong ON broadband sweep (disp_on_max > 8) vs ~25% of GBT
train, but confident-noise cadences look identical on both telescopes. Which cadences are they, what do we predict
for them, and what shape do they have? Plus an EM estimate of the test label mix from our probabilities
(Saerens et al. 2002; our stackers are class-balanced, so their probabilities assume equal class priors)."""
from pathlib import Path

import numpy as np
import pandas as pd

from common2 import CLASSES, labels, load_feats, parser

SHAPE = ['disp_on_max', 'disp_off_max', 'disp_on_panels', 'disp_on_pulses', 'disp_curve', 'row_on_max',
         'patch_diff', 'tex4x8_diff', 'exc_area_diff', 'steep_max', 'line_on_max']


def em_prior(p, iters=100):
    pi = np.full(p.shape[1], 1 / p.shape[1])
    for _ in range(iters):
        q = p * pi
        q /= q.sum(1, keepdims=True)
        pi = q.mean(0)
    return pi


def main():
    p = parser(__doc__)
    p.add_argument('--probs', default='sub_r2_stack6_pl_probs.npz')
    a = p.parse_args()
    work = Path(a.work)
    ftr, fte = load_feats(work, True)
    y, pt = labels(work), np.load(work / a.probs)['test']
    yt = pt.argmax(1)
    short = [c[:8] for c in CLASSES]
    rows = {}
    for name, f, lab in (('train', ftr, y), ('test', fte, yt)):
        hi = f.disp_on_max.values > 8
        rows[f'{name}: % of all'] = np.bincount(lab, minlength=9) / len(lab) * 100
        rows[f'{name}: % of hi'] = np.bincount(lab[hi], minlength=9) / hi.sum() * 100
        rows[f'{name}: % hi in cls'] = [hi[lab == k].mean() * 100 if (lab == k).any() else np.nan for k in range(9)]
        print(f'{name}: {hi.mean() * 100:.1f}% of cadences have disp_on_max > 8')
    rows['test: EM prior %'] = em_prior(pt) * 100
    print(pd.DataFrame(rows, index=short).T.round(1).to_string())
    for name, f, lab in (('train (true)', ftr, y), ('test (pred)', fte, yt)):
        hi = f.disp_on_max.values > 8
        med = pd.DataFrame({c: f.loc[hi & (lab == k), SHAPE].median() for k, c in enumerate(short) if (hi & (lab == k)).sum() >= 20})
        print(f'== {name}, cadences with disp_on_max > 8: feature medians per class')
        print(med.round(2).to_string())


if __name__ == '__main__':
    main()
