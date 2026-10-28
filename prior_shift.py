"""Label-shift correction of a finished blend (Saerens, Latinne & Decaestecker 2002, EM prior adaptation).

PKS test has a different label mix than GBT train. EM re-estimates the test class priors from the blend's own
test probabilities, then p_test(c|x) ∝ p(c|x) * (pi_test / pi_train)^lam. lam = 0.5 is a damped half step,
because the blend's probabilities are not perfectly calibrated under the telescope shift.
Laptop only, < 50 MB RAM:  python prior_shift.py sub_r2_blend32 [lam ...]
"""
import sys

import numpy as np
import pandas as pd

from build_aligned_stack import DATA, PULL, gate, write_sub
from common2 import CLASSES, labels, log


def em_priors(p, pi_tr, iters=200, tol=1e-8):
    pi = pi_tr.copy()
    for _ in range(iters):
        q = p * (pi / pi_tr)
        q /= q.sum(1, keepdims=True)
        new = q.mean(0)
        if np.abs(new - pi).max() < tol:
            break
        pi = new
    return pi


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'sub_r2_blend32'
    lams = [float(a) for a in sys.argv[2:]] or [0.5]
    p = np.load(PULL / f'{name}_probs.npz')['test'].astype(np.float64)
    p /= p.sum(1, keepdims=True)
    pi_tr = np.bincount(labels(DATA), minlength=len(CLASSES)) / len(labels(DATA))
    pi_te = em_priors(p, pi_tr)
    base = p.argmax(1)
    log(f'{name}: class  train-prior  EM test-prior  argmax count')
    for c, cl in enumerate(CLASSES):
        log(f'  {cl:20s} {pi_tr[c]:.4f}  {pi_te[c]:.4f}  {(base == c).sum():5d}')
    for lam in lams:
        q = p * (pi_te / pi_tr) ** lam
        q /= q.sum(1, keepdims=True)
        out = f'{name}_em{int(round(lam * 100)):02d}'
        cnt = pd.Series(q.argmax(1)).value_counts()
        log(f'{out}: {(q.argmax(1) != base).sum()} labels changed | '
            + ' '.join(f'{CLASSES[c][:6]}:{cnt.get(c, 0)}' for c in range(len(CLASSES))))
        gate(out, q)
        write_sub(out, q)


if __name__ == '__main__':
    main()
