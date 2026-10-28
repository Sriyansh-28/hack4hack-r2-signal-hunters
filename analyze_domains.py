"""Analyze all pulled stacks across the 4 domains (work, work_sim, work_sim2, work_sim3):
1. Compare OOF Macro-F1, per-class F1, and PKS test-set confidence/entropy of every stack.
2. Test temperature/logit-std normalization across clean vs PKS-sim stacks so clean stacks don't overpower
   simulated stacks due to sharper calibration.
3. Build multi-world consensus blends (sub_r2_blend23.csv, sub_r2_blend24.csv) and compare against blend16 (0.8645).
"""
from pathlib import Path
import numpy as np
import pandas as pd

from common2 import CLASSES, ROOT, labels, log, macro_f1

PULL = ROOT / 'work_pull'
DATA = ROOT / 'data'

STACKS = [
    # Clean world (work)
    'sub_r2_stack2_probs.npz',
    'sub_r2_stack3_probs.npz',
    'sub_r2_stack_pl_probs.npz',
    'sub_r2_stack4_probs.npz',
    'sub_r2_stack4_pl_probs.npz',
    'sub_r2_stack5_probs.npz',
    'sub_r2_stack5_pl_probs.npz',
    'sub_r2_stack6_pl_probs.npz',
    'sub_r2_stack7_pl_probs.npz',
    # Harsh PKS-sim v1 (work_sim: noise 1.15-1.45x, --phys)
    'sub_r2_sim1_probs.npz',
    'sub_r2_sim2_probs.npz',
    # Milder + Glitches + Fine-Dispersion PKS-sim v2 (work_sim2)
    'sub_r2_simb1_probs.npz',
    'sub_r2_simb2_probs.npz',
    'sub_r2_simb3_probs.npz',
    # Ultra-Harsh PKS-sim v3 (work_sim3: noise 1.35-1.75x, --phys)
    'sub_r2_sim3a_probs.npz',
    'sub_r2_sim3b_probs.npz',
]


def entropy(p):
    p = np.clip(p, 1e-7, 1.0)
    return float(-(p * np.log(p)).sum(1).mean())


def temp_scale_to_target(p, target_ent):
    """Scale probabilities p^(1/T) so mean entropy on test equals target_ent."""
    lo_t, hi_t = 0.3, 3.0
    lp = np.log(np.clip(p, 1e-7, 1.0))
    for _ in range(25):
        mid = 0.5 * (lo_t + hi_t)
        q = np.exp(lp / mid)
        q /= q.sum(1, keepdims=True)
        if entropy(q) < target_ent:
            lo_t = mid
        else:
            hi_t = mid
    T = 0.5 * (lo_t + hi_t)
    q = np.exp(lp / T)
    return q / q.sum(1, keepdims=True), T


def quantile_match(src, ref):
    """Map each column of src (N, K) to the exact empirical quantile distribution of ref (N, K)."""
    out = np.empty_like(src)
    for k in range(src.shape[1]):
        order = np.argsort(src[:, k], kind='mergesort')
        out[order, k] = np.sort(ref[:, k])
    out /= np.maximum(out.sum(1, keepdims=True), 1e-7)
    return out


def save_sub(pred, ref_pred, name):
    sub = pd.DataFrame({'id': pd.read_csv(DATA / 'test_meta.csv').id, 'label': [CLASSES[i] for i in pred]})
    ss = pd.read_csv(DATA / 'sample_submission.csv')
    sub = ss[['id']].merge(sub, on='id', how='left')
    out = ROOT / 'submissions' / f'{name}.csv'
    sub.to_csv(out, index=False)
    ok = (len(sub) == len(ss) and sub.id.is_unique and sub.label.notna().all() and sub.label.isin(CLASSES).all())
    log(f'wrote {out}; CHECK {name}: ' + ('OK' if ok else 'FAILED') + f' | diff vs blend16 (0.8645): {(pred != ref_pred).sum()}')


def main():
    y = labels(DATA)
    loaded = {}
    rows = []
    short = [c[:6] for c in CLASSES]

    for fname in STACKS:
        fpath = PULL / fname
        if not fpath.exists():
            continue
        d = np.load(fpath)
        oof, test = d['oof'], d['test']
        loaded[fname.replace('sub_r2_', '').replace('_probs.npz', '')] = (oof, test)
        mf1, pf1 = macro_f1(y, oof.argmax(1))
        row = {'model': fname.replace('sub_r2_', '').replace('_probs.npz', ''),
               'OOF_F1': round(mf1, 4),
               'oof_ent': round(entropy(oof), 3),
               'test_ent': round(entropy(test), 3),
               'test_maxP': round(float(test.max(1).mean()), 3)}
        for c, v in zip(short, pf1):
            row[c] = round(v, 3)
        rows.append(row)

    df = pd.DataFrame(rows)
    print('\n=== ALL PULLED STACKS: OOF PER-CLASS F1 & TEST ENTROPY ===')
    print(df.to_string(index=False))

    # Reconstruct blend16 (0.8645 champion) as exact reference
    m16 = ['stack6_pl', 'stack_pl', 'sim1', 'sim2']
    w16 = [1.2, 0.8, 0.8, 1.2]
    oof_wlin = sum(w * loaded[k][0] for w, k in zip(w16, m16)) / sum(w16)
    test_wlin = sum(w * loaded[k][1] for w, k in zip(w16, m16)) / sum(w16)
    oof_geo = np.exp(sum(w * np.log(np.clip(loaded[k][0], 1e-6, 1)) for w, k in zip(w16, m16)) / sum(w16))
    oof_geo /= oof_geo.sum(1, keepdims=True)
    test_geo = np.exp(sum(w * np.log(np.clip(loaded[k][1], 1e-6, 1)) for w, k in zip(w16, m16)) / sum(w16))
    test_geo /= test_geo.sum(1, keepdims=True)
    oof_b16 = 0.5 * oof_wlin + 0.5 * oof_geo
    test_b16 = 0.5 * test_wlin + 0.5 * test_geo
    pred_b16 = test_b16.argmax(1)
    log(f'Reference blend16 (0.8645 public) OOF F1: {macro_f1(y, oof_b16.argmax(1))[0]:.4f}')

    # Candidate A (sub_r2_blend23): Richer 1:1 Clean vs PKS-Sim Grand Consensus (50/50 linear + log-odds)
    # Clean side (total weight 5.0): stack6_pl (1.6), stack7_pl (1.2), stack5 (1.0, un-pseudo-labeled!), stack4_pl (0.6), stack_pl (0.6)
    # Sim side   (total weight 5.0): sim2 (2.0), sim1 (1.4), simb3 (0.8, glitches+fine_disp), sim3b (0.8, ultra-harsh)
    spec_23 = [
        ('stack6_pl', 1.6), ('stack7_pl', 1.2), ('stack5', 1.0), ('stack4_pl', 0.6), ('stack_pl', 0.6),
        ('sim2', 2.0), ('sim1', 1.4), ('simb3', 0.8), ('sim3b', 0.8)
    ]
    w23 = sum(w for _, w in spec_23)
    oof_lin23 = sum(w * loaded[k][0] for k, w in spec_23) / w23
    test_lin23 = sum(w * loaded[k][1] for k, w in spec_23) / w23
    oof_geo23 = np.exp(sum(w * np.log(np.clip(loaded[k][0], 1e-6, 1)) for k, w in spec_23) / w23)
    oof_geo23 /= oof_geo23.sum(1, keepdims=True)
    test_geo23 = np.exp(sum(w * np.log(np.clip(loaded[k][1], 1e-6, 1)) for k, w in spec_23) / w23)
    test_geo23 /= test_geo23.sum(1, keepdims=True)
    oof_23 = 0.5 * oof_lin23 + 0.5 * oof_geo23
    test_23 = 0.5 * test_lin23 + 0.5 * test_geo23
    log(f'blend23 (9-stack 4-world 1:1 linear+log-odds) OOF F1: {macro_f1(y, oof_23.argmax(1))[0]:.4f}')
    save_sub(test_23.argmax(1), pred_b16, 'sub_r2_blend23')

    # Candidate B (sub_r2_blend24): Temperature-equalized blend16 + stack5 + simb3
    # Equalize test entropy of each stack to 0.55 so clean stacks and sim stacks have identical sharpness on PKS test!
    spec_24 = [
        ('stack6_pl', 1.5), ('stack5', 0.8), ('stack_pl', 0.7),
        ('sim2', 1.5), ('sim1', 1.0), ('simb3', 0.5)
    ]
    target_ent = 0.55
    w24 = sum(w for _, w in spec_24)
    test_t_list, oof_t_list = [], []
    for k, w in spec_24:
        qt, T = temp_scale_to_target(loaded[k][1], target_ent)
        lo = np.log(np.clip(loaded[k][0], 1e-7, 1.0)) / T
        qo = np.exp(lo)
        qo /= qo.sum(1, keepdims=True)
        test_t_list.append((w, qt))
        oof_t_list.append((w, qo))
    oof_lin24 = sum(w * q for w, q in oof_t_list) / w24
    test_lin24 = sum(w * q for w, q in test_t_list) / w24
    oof_geo24 = np.exp(sum(w * np.log(np.clip(q, 1e-6, 1)) for w, q in oof_t_list) / w24)
    oof_geo24 /= oof_geo24.sum(1, keepdims=True)
    test_geo24 = np.exp(sum(w * np.log(np.clip(q, 1e-6, 1)) for w, q in test_t_list) / w24)
    test_geo24 /= test_geo24.sum(1, keepdims=True)
    oof_24 = 0.5 * oof_lin24 + 0.5 * oof_geo24
    test_24 = 0.5 * test_lin24 + 0.5 * test_geo24
    log(f'blend24 (Temperature-equalized 6-stack 1:1 linear+log-odds) OOF F1: {macro_f1(y, oof_24.argmax(1))[0]:.4f}')
    save_sub(test_24.argmax(1), pred_b16, 'sub_r2_blend24')

    # Candidate C (sub_r2_blend25): Pure feats3-generation 1:1 ensemble (drops pre-feats3 stack_pl drag completely!)
    # Clean side (5.0): stack6_pl (1.6), stack7_pl (1.2), stack5 (0.9, no-PL), stack4 (0.5, no-PL), stack4_pl (0.4), stack5_pl (0.4)
    # Sim side   (5.0): sim2 (2.2), sim1 (1.4, no-PL), simb3 (0.9), simb2 (0.5)
    spec_25 = [
        ('stack6_pl', 1.6), ('stack7_pl', 1.2), ('stack5', 0.9), ('stack4', 0.5), ('stack4_pl', 0.4), ('stack5_pl', 0.4),
        ('sim2', 2.2), ('sim1', 1.4), ('simb3', 0.9), ('simb2', 0.5)
    ]
    w25 = sum(w for _, w in spec_25)
    oof_lin25 = sum(w * loaded[k][0] for k, w in spec_25) / w25
    test_lin25 = sum(w * loaded[k][1] for k, w in spec_25) / w25
    oof_geo25 = np.exp(sum(w * np.log(np.clip(loaded[k][0], 1e-6, 1)) for k, w in spec_25) / w25)
    oof_geo25 /= oof_geo25.sum(1, keepdims=True)
    test_geo25 = np.exp(sum(w * np.log(np.clip(loaded[k][1], 1e-6, 1)) for k, w in spec_25) / w25)
    test_geo25 /= test_geo25.sum(1, keepdims=True)
    oof_25 = 0.5 * oof_lin25 + 0.5 * oof_geo25
    test_25 = 0.5 * test_lin25 + 0.5 * test_geo25
    log(f'blend25 (10-stack pure-feats3 1:1 linear+log-odds, no stack_pl drag) OOF F1: {macro_f1(y, oof_25.argmax(1))[0]:.4f}')
    save_sub(test_25.argmax(1), pred_b16, 'sub_r2_blend25')
    log(f'diff between blend23 and blend25: {(test_23.argmax(1) != test_25.argmax(1)).sum()}')

    # Candidate D (sub_r2_blend26): 11-stack (adds sim3b) with Per-Class Quantile-Matched Rank Consensus + Linear/Log-odds
    spec_26 = [
        ('stack6_pl', 1.6), ('stack7_pl', 1.2), ('stack5', 0.9), ('stack4', 0.5), ('stack4_pl', 0.4), ('stack5_pl', 0.4),
        ('sim2', 1.9), ('sim1', 1.3), ('simb3', 0.8), ('simb2', 0.4), ('sim3b', 0.6)
    ]
    w26 = sum(w for _, w in spec_26)
    oof_lin26 = sum(w * loaded[k][0] for k, w in spec_26) / w26
    test_lin26 = sum(w * loaded[k][1] for k, w in spec_26) / w26
    oof_geo26 = np.exp(sum(w * np.log(np.clip(loaded[k][0], 1e-6, 1)) for k, w in spec_26) / w26)
    oof_geo26 /= oof_geo26.sum(1, keepdims=True)
    test_geo26 = np.exp(sum(w * np.log(np.clip(loaded[k][1], 1e-6, 1)) for k, w in spec_26) / w26)
    test_geo26 /= test_geo26.sum(1, keepdims=True)
    ref_oof, ref_test = 0.5 * oof_lin26 + 0.5 * oof_geo26, 0.5 * test_lin26 + 0.5 * test_geo26

    oof_qm = sum(w * quantile_match(loaded[k][0], ref_oof) for k, w in spec_26) / w26
    test_qm = sum(w * quantile_match(loaded[k][1], ref_test) for k, w in spec_26) / w26
    oof_26 = 0.6 * ref_oof + 0.4 * oof_qm
    test_26 = 0.6 * ref_test + 0.4 * test_qm
    log(f'blend26 (11-stack 4-world Quantile-Matched + Linear + Log-Odds) OOF F1: {macro_f1(y, oof_26.argmax(1))[0]:.4f}')
    save_sub(test_26.argmax(1), pred_b16, 'sub_r2_blend26')
    log(f'diff between blend25 and blend26: {(test_25.argmax(1) != test_26.argmax(1)).sum()}')

    print('\nClass distribution comparison:')
    print(pd.DataFrame({
        'blend16 (0.8645)': pd.Series(pred_b16).map(dict(enumerate(CLASSES))).value_counts(),
        'blend23 (9-stack)': pd.Series(test_23.argmax(1)).map(dict(enumerate(CLASSES))).value_counts(),
        'blend25 (no-old-pl)': pd.Series(test_25.argmax(1)).map(dict(enumerate(CLASSES))).value_counts(),
        'blend26 (quantile)': pd.Series(test_26.argmax(1)).map(dict(enumerate(CLASSES))).value_counts(),
    }).fillna(0).astype(int).to_string())


if __name__ == '__main__':
    main()
