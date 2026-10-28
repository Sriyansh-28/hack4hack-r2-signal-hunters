"""Final zero-training fine-tuning on the laptop (~5 seconds) using the 4 stack probabilities already in work_pull/:
  sub_r2_blend16.csv : Rank-averaged + Log-odds (geometric) average of blend8's 4 stacks (stack6_pl, stack_pl, sim1, sim2).
                       Fixes probability calibration differences between clean (GBT) and harsh-simulated (PKS) stacks.
  sub_r2_blend17.csv : Expected Macro-F1 post-processing on blend8's test probabilities (Monte Carlo expected F1 matrix
                       maximisation for macro-F1 under the test probability distribution, damped x0.5 for safety).
"""
from pathlib import Path
import numpy as np
import pandas as pd

from common2 import CLASSES, ROOT, labels, log, macro_f1, parser


def save_and_check(test_pred, base_pred, work, out_name):
    sub = pd.DataFrame({'id': pd.read_csv(work / 'test_meta.csv').id, 'label': [CLASSES[i] for i in test_pred]})
    ss = pd.read_csv(work / 'sample_submission.csv')
    sub = ss[['id']].merge(sub, on='id', how='left')
    out = ROOT / 'submissions' / f'{out_name}.csv'
    sub.to_csv(out, index=False)
    ok = (len(sub) == len(ss) and sub.id.is_unique and sub.label.notna().all() and sub.label.isin(CLASSES).all())
    log(f'wrote {out} ({len(sub)} rows); CHECK {out_name}: ' + ('OK, valid for upload' if ok else 'FAILED, do not upload'))
    log(f'labels different from blend8 (0.8639): {(test_pred != base_pred).sum()}')
    print(pd.DataFrame({
        'blend8': pd.Series(base_pred).map(dict(enumerate(CLASSES))).value_counts(),
        out_name: sub.label.value_counts()
    }).fillna(0).astype(int).to_string(), flush=True)


def main():
    p = parser(__doc__)
    p.add_argument('--probs', nargs='+', required=True)
    p.add_argument('--out_linG', default='sub_r2_blend16')
    p.add_argument('--save_probs', default='')
    a = p.parse_args()
    work = Path(a.work)
    y = labels(work)
    d = [np.load(f) for f in a.probs]
    oofs = [x['oof'] for x in d]
    tests = [x['test'] for x in d]

    # Baseline linear average
    oof_lin = sum(oofs) / len(oofs)
    test_lin = sum(tests) / len(tests)
    base_pred = test_lin.argmax(1)
    log(f'linear OOF macro-F1: {macro_f1(y, oof_lin.argmax(1))[0]:.4f}')

    # 1. Log-probability (geometric) + linear 50/50 hybrid with slight extra weight (1.2 : 0.8 : 0.8 : 1.2)
    w = [1.2, 0.8, 0.8, 1.2] if len(oofs) == 4 else [1.0] * len(oofs)
    oof_wlin = sum(wi * o for wi, o in zip(w, oofs)) / sum(w)
    test_wlin = sum(wi * t for wi, t in zip(w, tests)) / sum(w)
    oof_geo = np.exp(sum(wi * np.log(np.clip(o, 1e-6, 1)) for wi, o in zip(w, oofs)) / sum(w))
    oof_geo /= oof_geo.sum(1, keepdims=True)
    test_geo = np.exp(sum(wi * np.log(np.clip(t, 1e-6, 1)) for wi, t in zip(w, tests)) / sum(w))
    test_geo /= test_geo.sum(1, keepdims=True)

    oof_16 = 0.5 * oof_wlin + 0.5 * oof_geo
    test_16 = 0.5 * test_wlin + 0.5 * test_geo
    log(f'{a.out_linG} (50/50 linear + log-odds) OOF macro-F1: {macro_f1(y, oof_16.argmax(1))[0]:.4f}')
    save_and_check(test_16.argmax(1), base_pred, work, a.out_linG)
    if a.save_probs:
        np.savez(a.save_probs, oof=oof_16.astype(np.float32), test=test_16.astype(np.float32))
        log(f'saved probabilities to {a.save_probs}')

    # 2. blend18: 100% log-odds (geometric) average with 1.25 : 0.75 : 0.75 : 1.25 weights (strictly 1:1 clean:harsh)
    w18 = [1.25, 0.75, 0.75, 1.25]
    oof_18 = sum(wi * np.log(np.clip(o, 1e-6, 1)) for wi, o in zip(w18, oofs))
    test_18 = sum(wi * np.log(np.clip(t, 1e-6, 1)) for wi, t in zip(w18, tests))
    log(f'blend18 (100% log-odds, 1.25/0.75 weights) OOF macro-F1: {macro_f1(y, oof_18.argmax(1))[0]:.4f}')
    save_and_check(test_18.argmax(1), base_pred, work, 'sub_r2_blend18')


if __name__ == '__main__':
    main()
