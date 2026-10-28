"""Build Covariate-Shift-Aligned Stacks (stack8_pl_aligned & sim4_aligned) + sub_r2_blend26.csv.

Why Stage 1A's raw stack8_pl had 692 Scint and 1266 rfi_persistent on PKS test:
1. Stage 1A ran on HPC at 00:42 IST BEFORE work/feats_test.parquet and work/feats3_test.parquet were
   replaced with --phys (so 92% of PKS test had track_clean = -99 in gbm_x4 and stack8_pl!).
2. Stage 1A also included legacy pre---phys models (work/cnn, work/gbm, work/gbm_x) where work/gbm
   predicted 1,613 scintillating_star and work/cnn predicted 520 eti_candidate on PKS test.
3. Raw PKS test tabular features had disp_on_max p50 = 8.95 (vs 2.96 on GBT) due to 1-row glitches,
   and 0.70x SNR compression on track_on_min / line_on_min / persist_all6 / persist_5of6.

This script (uses < 150 MB RAM, no spectrograms loaded):
1. Validates damped_quantile_align on our Cross-Telescope Benchmark (Clean GBT -> Sim PKS held-out folds).
2. Re-runs gbm_x4 (clean & sim1) with --phys test features + damped_quantile_align(alpha=0.85).
3. Re-runs stack8_pl_aligned (clean, excluding broken pre---phys cnn/gbm/gbm_x) and sim4_aligned.
4. Builds sub_r2_blend26.csv (50/50 Linear + Log-Odds consensus of stack8_pl_aligned, stack6_pl,
   stack_pl, sim4_aligned, sim2, sim1) and compares class counts & agreement with blend16 (0.8645).
"""
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from common2 import CLASSES, N_THREADS, ROOT, damped_quantile_align, labels, log, macro_f1, report
from stack2 import META, META3, META4, META5

warnings.filterwarnings('ignore')
PULL = ROOT / 'work_pull'
DATA = ROOT / 'data'
# Pre-upload gate: PKS test class counts of the proven public winners (blend16 0.8645, blend27 0.8655)
GATE = {0: (280, 315), 3: (575, 605), 6: (1860, 1905), 7: (1345, 1385), 8: (2440, 2500)}


def get_folds(y):
    folds = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y)):
        folds[va] = k
    return folds


def has_feats5(folder):
    return (folder / 'feats5_train.parquet').exists() and (folder / 'feats5_test.parquet').exists()


def load_domain_feats(folder_tr, folder_te_phys, align_alpha=0.85, feats5=False):
    """Always use folder_te_phys (sim1_preds) for test features so --phys is 100% guaranteed.
    feats5=True adds feats5.py (test from folder_te_phys if pulled there, else from folder_tr: same PKS test cadences)."""
    tr_parts = [pd.read_parquet(folder_tr / f'{n}_train.parquet') for n in ('feats', 'feats3', 'feats4')]
    te_parts = [pd.read_parquet(folder_te_phys / f'{n}_test.parquet') for n in ('feats', 'feats3', 'feats4')]
    if feats5:
        tr_parts.append(pd.read_parquet(folder_tr / 'feats5_train.parquet'))
        te_src = folder_te_phys if has_feats5(folder_te_phys) else folder_tr
        te_parts.append(pd.read_parquet(te_src / 'feats5_test.parquet'))
    tr = pd.concat(tr_parts, axis=1)
    te = pd.concat(te_parts, axis=1)[tr.columns]
    if align_alpha > 0:
        te = damped_quantile_align(te, tr, alpha=align_alpha)
    return tr, te


def run_xval_check(y, folds):
    """Cross-telescope validation: train stacker on Clean GBT (work_preds), evaluate on Harsh Sim PKS (sim1_preds)
    with vs without damped_quantile_align on the tabular meta features."""
    log('=== STEP 1: Cross-Telescope Validation (Train on GBT -> Test on Simulated PKS Folds) ===')
    tr_clean, _ = load_domain_feats(PULL / 'work_preds', PULL / 'sim1_preds', align_alpha=0.0)
    tr_sim, _ = load_domain_feats(PULL / 'sim1_preds', PULL / 'sim1_preds', align_alpha=0.0)
    tr_sim_aligned = damped_quantile_align(tr_sim, tr_clean, alpha=0.85)

    meta = [c for c in META + META3 + META4 if c in tr_clean.columns]
    # Compare Clean CNN + GBM stack evaluated on Sim1 folds
    clean_models = ['cnn_v2x', 'cnn_v2xpl', 'gbm_x4']
    sim_models = ['cnn_v2x', 'cnn_v2xpl', 'gbm_x4']
    Xc = np.log(np.clip(np.concatenate([np.load(PULL / 'work_preds' / f'{m}.npz')['oof'] for m in clean_models], 1), 1e-7, 1))
    Xs = np.log(np.clip(np.concatenate([np.load(PULL / 'sim1_preds' / f'{m}.npz')['oof'] for m in sim_models], 1), 1e-7, 1))

    Xgc = np.hstack([Xc, tr_clean[meta].values])
    Xgs_raw = np.hstack([Xs, tr_sim[meta].values])
    Xgs_ali = np.hstack([Xs, tr_sim_aligned[meta].values])

    K = len(CLASSES)
    go_raw, go_ali = np.zeros((len(y), K)), np.zeros((len(y), K))
    for k in range(5):
        t, v = folds != k, folds == k
        g = lgb.LGBMClassifier(objective='multiclass', n_estimators=2000, learning_rate=0.03, num_leaves=15,
                               min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                               reg_lambda=5.0, class_weight='balanced', random_state=k, n_jobs=N_THREADS, verbose=-1)
        g.fit(Xgc[t], y[t], eval_set=[(Xgc[v], y[v])], callbacks=[lgb.early_stopping(100, verbose=False)])
        go_raw[v] = g.predict_proba(Xgs_raw[v])
        go_ali[v] = g.predict_proba(Xgs_ali[v])
    f1_raw, pf_raw = macro_f1(y, go_raw.argmax(1))
    f1_ali, pf_ali = macro_f1(y, go_ali.argmax(1))
    log(f'Cross-Telescope LightGBM Stacker (Raw Sim Features):     Macro-F1 = {f1_raw:.4f} (eti={pf_raw[0]:.3f} sci={pf_raw[3]:.3f} rfi_p={pf_raw[7]:.3f})')
    log(f'Cross-Telescope LightGBM Stacker (Aligned Sim Features): Macro-F1 = {f1_ali:.4f} (eti={pf_ali[0]:.3f} sci={pf_ali[3]:.3f} rfi_p={pf_ali[7]:.3f})')


def train_aligned_gbm(folder_tr, y, folds, name):
    """Keep the exact OOF from HPC's gbm_x4.npz, or re-train 5 folds with aligned --phys test features."""
    tr, te = load_domain_feats(folder_tr, PULL / 'sim1_preds', align_alpha=0.85)
    X, Xt = tr.values, te.values
    K = len(CLASSES)
    oof = np.zeros((len(y), K), dtype=np.float32)
    test = np.zeros((len(Xt), K), dtype=np.float32)
    for k in range(5):
        t, v = folds != k, folds == k
        m = lgb.LGBMClassifier(
            objective='multiclass', num_class=K, n_estimators=2500, learning_rate=0.03,
            num_leaves=31, min_child_samples=25, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0, class_weight='balanced',
            random_state=42 + k, n_jobs=N_THREADS, verbose=-1
        )
        m.fit(X[t], y[t], eval_set=[(X[v], y[v])], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[v] = m.predict_proba(X[v])
        test += m.predict_proba(Xt) / 5.0
    s = report(y, oof, f'{name} (aligned --phys)')
    out_p = folder_tr / 'gbm_x4_aligned.npz'
    np.savez(out_p, oof=oof, test=test, score=s)
    return oof, test


def align_model_test_logits(d_list, model_names, ref_names=('cnn_v2notab', 'cnn_v2pl', 'cnn_v2xpl')):
    """Eliminate per-model covariate bias on PKS test (e.g. gbm_x4 predicting 896 Scint and cnn_v2 predicting 722 Scint)
    by aligning each base model's test log-probability mean & std per class to the PKS-adapted reference models,
    adjusted by their OOF mean/std ratio."""
    K = len(CLASSES)
    oof_logs = [np.log(np.clip(x['oof'], 1e-7, 1.0)) for x in d_list]
    te_logs = [np.log(np.clip(x['test'], 1e-7, 1.0)) for x in d_list]

    ref_idx = [i for i, m in enumerate(model_names) if m in ref_names]
    if not ref_idx:
        ref_idx = [i for i, m in enumerate(model_names) if 'pl' in m]
    if not ref_idx:
        return np.concatenate(oof_logs, axis=1), np.concatenate(te_logs, axis=1)

    # Reference shift from OOF to Test on the clean/PKS-adapted CNN models
    ref_oof_mu = np.mean([oof_logs[i].mean(axis=0) for i in ref_idx], axis=0)
    ref_te_mu = np.mean([te_logs[i].mean(axis=0) for i in ref_idx], axis=0)
    ref_oof_sd = np.mean([oof_logs[i].std(axis=0) for i in ref_idx], axis=0)
    ref_te_sd = np.mean([te_logs[i].std(axis=0) for i in ref_idx], axis=0)

    aligned_te_logs = []
    for i, m in enumerate(model_names):
        Xo, Xt = oof_logs[i], te_logs[i]
        if m in ref_names:
            aligned_te_logs.append(Xt)
            continue
        # Target test mean & std for model i based on its own OOF scale + reference OOF->Test shift
        target_mu = Xo.mean(axis=0) + (ref_te_mu - ref_oof_mu)
        target_sd = Xo.std(axis=0) * (ref_te_sd / np.maximum(ref_oof_sd, 1e-6))
        cur_mu = Xt.mean(axis=0)
        cur_sd = np.maximum(Xt.std(axis=0), 1e-6)
        Xt_cal = target_mu + (Xt - cur_mu) * (target_sd / cur_sd)
        # Re-normalize in log-prob space so exp(Xt_cal) sums to 1 per row
        Xt_cal -= np.log(np.exp(Xt_cal).sum(axis=1, keepdims=True))
        aligned_te_logs.append(Xt_cal.astype(np.float32))

    return np.concatenate(oof_logs, axis=1), np.concatenate(aligned_te_logs, axis=1)


def run_aligned_stack(folder_tr, model_names, y, folds, out_name, feats5=False):
    tr, te = load_domain_feats(folder_tr, PULL / 'sim1_preds', align_alpha=0.85, feats5=feats5)
    meta = [c for c in META + META3 + META4 + (META5 if feats5 else []) if c in tr.columns]
    d = [np.load(folder_tr / f'{m}.npz') for m in model_names]
    log(f'{out_name} models: ' + ' '.join(model_names))
    X, Xt = align_model_test_logits(d, model_names)
    Xg, Xgt = np.hstack([X, tr[meta].values]), np.hstack([Xt, te[meta].values])
    K = len(CLASSES)
    lo, lt, go, gt = (np.zeros((len(y), K)), np.zeros((len(Xt), K)), np.zeros((len(y), K)), np.zeros((len(Xt), K)))
    for k in range(5):
        t, v = folds != k, folds == k
        m = LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced').fit(X[t], y[t])
        lo[v] = m.predict_proba(X[v])
        lt += m.predict_proba(Xt) / 5.0
        g = lgb.LGBMClassifier(objective='multiclass', n_estimators=2000, learning_rate=0.03, num_leaves=15,
                               min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                               reg_lambda=5.0, class_weight='balanced', random_state=k, n_jobs=N_THREADS, verbose=-1)
        g.fit(Xg[t], y[t], eval_set=[(Xg[v], y[v])], callbacks=[lgb.early_stopping(100, verbose=False)])
        go[v] = g.predict_proba(Xg[v])
        gt += g.predict_proba(Xgt) / 5.0
    report(y, lo, f'{out_name} logistic stacker')
    report(y, go, f'{out_name} LightGBM stacker (aligned)')
    oof, test = (lo + go) / 2.0, (lt + gt) / 2.0
    report(y, oof, f'{out_name} STACK (average)')
    np.savez(PULL / f'{out_name}_probs.npz', oof=oof.astype(np.float32), test=test.astype(np.float32))
    # Also save pure logistic stacker (100% immune to any tree split shift)
    np.savez(PULL / f'{out_name}_log_probs.npz', oof=lo.astype(np.float32), test=lt.astype(np.float32))
    return oof, test, lo, lt


def blend_lin_geom(stacks, weights):
    w = np.array(weights, dtype=np.float64)
    w /= w.sum()
    oof_lin = sum(wi * s['oof'] for wi, s in zip(w, stacks))
    test_lin = sum(wi * s['test'] for wi, s in zip(w, stacks))

    oof_log = sum(wi * np.log(np.clip(s['oof'], 1e-6, 1.0)) for wi, s in zip(w, stacks))
    oof_geom = np.exp(oof_log)
    oof_geom /= oof_geom.sum(axis=1, keepdims=True)

    test_log = sum(wi * np.log(np.clip(s['test'], 1e-6, 1.0)) for wi, s in zip(w, stacks))
    test_geom = np.exp(test_log)
    test_geom /= test_geom.sum(axis=1, keepdims=True)

    return 0.5 * oof_lin + 0.5 * oof_geom, 0.5 * test_lin + 0.5 * test_geom


def inspect_sub(name, oof, test, y, b16):
    oof_f1, pf = macro_f1(y, oof.argmax(1))
    te_p = test.argmax(1)
    agree = (te_p == b16).mean() * 100
    diff = (te_p != b16).sum()
    cnt = pd.Series(te_p).value_counts()
    print(f'{name:22s} | OOF:{oof_f1:.4f} (eti:{pf[0]:.3f} sci:{pf[3]:.3f}) | agree b16:{agree:5.1f}% ({diff:3d} diff) | '
          f'ETI:{cnt.get(0,0):4d} Sci:{cnt.get(3,0):4d} RFI_i:{cnt.get(6,0):4d} RFI_p:{cnt.get(7,0):4d} Noise:{cnt.get(8,0):4d}')


def gate(name, test):
    """Pre-upload gate: PKS test class counts must stay in the band of the proven public winners."""
    cnt = pd.Series(test.argmax(1)).value_counts()
    bad = [f'{CLASSES[c][:8]}={cnt.get(c, 0)} not in {lo}-{hi}' for c, (lo, hi) in GATE.items()
           if not lo <= cnt.get(c, 0) <= hi]
    log(f'GATE {name}: ' + ('PASS' if not bad else 'FAIL (' + ', '.join(bad) + ')'))
    return not bad


def write_sub(name, test):
    meta_te = pd.read_csv(DATA / 'test_meta.csv')
    ss = pd.read_csv(DATA / 'sample_submission.csv')
    sub = pd.DataFrame({'id': meta_te.id, 'label': [CLASSES[i] for i in test.argmax(1)]})
    sub = ss[['id']].merge(sub, on='id', how='left')
    out = ROOT / 'submissions' / f'{name}.csv'
    sub.to_csv(out, index=False)
    np.savez(PULL / f'{name}_probs.npz', test=test.astype(np.float32))           # pseudo-label source for cnn3/cnn4
    ok = (len(sub) == len(ss) and sub.id.is_unique and set(sub.id) == set(ss.id)
          and sub.label.notna().all() and sub.label.isin(CLASSES).all())
    log(f'wrote {out} ({len(sub)} rows) -> CHECK: {"OK" if ok else "FAILED"}')


def main():
    y = labels(DATA)
    folds = get_folds(y)
    b16 = pd.read_csv(ROOT / 'submissions' / 'sub_r2_blend16.csv').label.map({c: i for i, c in enumerate(CLASSES)}).values

    # 2. Load or train aligned gbm_x4 in both Clean and Sim1 worlds
    if not (PULL / 'work_preds' / 'gbm_x4_aligned.npz').exists():
        log('=== STEP 2: Re-computing gbm_x4 with --phys + Damped Quantile Alignment ===')
        train_aligned_gbm(PULL / 'work_preds', y, folds, 'work/gbm_x4_aligned')
        train_aligned_gbm(PULL / 'sim1_preds', y, folds, 'sim1/gbm_x4_aligned')
    else:
        log('=== Re-using cached gbm_x4_aligned.npz ===')

    # 3. Build aligned stacks (automatically including cnn_v3pl if pulled from Stage 2/3!)
    log('=== STEP 3: Building stack8_pl_aligned and sim4_aligned ===')
    clean_models = ['gbm_x4_aligned', 'cnn_v2', 'cnn_v2s1', 'cnn_v2s3', 'cnn_v2notab', 'cnn_v2x', 'cnn_v2pl', 'cnn_v2xpl', 'cnn_v2xpl3']
    if (PULL / 'work_preds' / 'cnn_v3pl.npz').exists():
        clean_models.append('cnn_v3pl')
    sim_models = ['gbm_x', 'gbm_x4_aligned', 'cnn_v2x', 'cnn_v2xpl']    # fixed: keeps blend26/28 reproducible
    sim_v3 = (PULL / 'sim1_preds' / 'cnn_v3pl.npz').exists()             # Stage 3 (work_sim cnn3) -> blend30 only
    s8_oof, s8_te, s8_lo, s8_lt = run_aligned_stack(PULL / 'work_preds', clean_models, y, folds, 'sub_r2_stack8_pl_aligned')
    s4_oof, s4_te, s4_lo, s4_lt = run_aligned_stack(PULL / 'sim1_preds', sim_models, y, folds, 'sub_r2_sim4_aligned')

    # 4. Compare all candidates vs blend16 (0.8645) and blend20 (from HPC)
    print('\n=== CANDIDATE COMPARISON vs CHAMPION blend16 (0.8645) ===')
    s6 = np.load(PULL / 'sub_r2_stack6_pl_probs.npz')
    spl = np.load(PULL / 'sub_r2_stack_pl_probs.npz')
    sim1 = np.load(PULL / 'sub_r2_sim1_probs.npz')
    sim2 = np.load(PULL / 'sub_r2_sim2_probs.npz')
    s8_raw = np.load(PULL / 'sub_r2_stack8_pl_probs.npz')
    s4_raw = np.load(PULL / 'sub_r2_sim4_probs.npz')

    inspect_sub('stack8_pl (raw HPC)', s8_raw['oof'], s8_raw['test'], y, b16)
    inspect_sub('stack8_pl_aligned', s8_oof, s8_te, y, b16)
    inspect_sub('stack8_pl_logistic', s8_lo, s8_lt, y, b16)
    inspect_sub('sim4 (raw HPC)', s4_raw['oof'], s4_raw['test'], y, b16)
    inspect_sub('sim4_aligned', s4_oof, s4_te, y, b16)

    # blend20 from HPC (raw stack8_pl + stack6_pl + sim1 + sim4)
    b20_oof, b20_te = blend_lin_geom([s8_raw, s6, sim1, s4_raw], [1.2, 0.8, 0.8, 1.2])
    inspect_sub('sub_r2_blend20 (HPC)', b20_oof, b20_te, y, b16)

    # blend26: 50/50 Linear + Log-Odds of (stack8_pl_aligned, stack6_pl, stack_pl, sim4_aligned, sim2, sim1)
    s8_al = {'oof': s8_oof, 'test': s8_te}
    s8_lg = {'oof': s8_lo, 'test': s8_lt}
    s4_al = {'oof': s4_oof, 'test': s4_te}
    b26_oof, b26_te = blend_lin_geom([s8_al, s6, spl, s4_al, sim2, sim1], [1.5, 0.8, 0.8, 1.5, 0.8, 0.6])
    inspect_sub('sub_r2_blend26 (ALIGN)', b26_oof, b26_te, y, b16)
    write_sub('sub_r2_blend26', b26_te)

    # blend27 / blend28 (if cnn_v3pl is present, write sub_r2_blend28 so blend27=0.8655 is preserved!)
    out_tag = 'sub_r2_blend28' if (PULL / 'work_preds' / 'cnn_v3pl.npz').exists() else 'sub_r2_blend27'
    b27_oof, b27_te = blend_lin_geom([s8_al, s8_lg, s6, spl, s4_al, sim2, sim1], [1.0, 1.0, 0.7, 0.9, 1.4, 0.8, 0.6])
    inspect_sub(f'{out_tag} (SAFE)', b27_oof, b27_te, y, b16)
    gate(out_tag, b27_te)
    write_sub(out_tag, b27_te)

    # blend29 (Phase 1): same recipe as blend27/28, with gbm_x5 + feats5 meta features in both worlds.
    # Needs work_preds/{gbm_x5.npz, feats5_*.parquet}; sim1_preds/{gbm_x5.npz, feats5_*.parquet} is used if pulled.
    if (PULL / 'work_preds' / 'gbm_x5.npz').exists() and has_feats5(PULL / 'work_preds'):
        v4 = (PULL / 'work_preds' / 'cnn_v4pl.npz').exists()                  # Phase 2 cnn4 -> blend31
        sim_v4 = v4 and sim_v3 and (PULL / 'sim1_preds' / 'cnn_v4pl.npz').exists()   # Phase 3 sim cnn4 -> blend32
        s1 = sim_v4 and all((PULL / f / 'cnn_v4pls1.npz').exists() for f in ('work_preds', 'sim1_preds'))  # Phase 4 seed 1
        s2 = s1 and all((PULL / f / 'cnn_v4pls2.npz').exists() for f in ('work_preds', 'sim1_preds'))  # Phase 5 seed 2
        p33w, p33s = [s1 and (PULL / f / 'cnn_v4pl33.npz').exists() for f in ('work_preds', 'sim1_preds')]  # Phase 6
        if p33w or p33s:
            s2 = False      # blend34 (with the weaker seed 2) < blend33 on OOF -> blend35 = blend33 models + cnn_v4pl33
        b29 = ('sub_r2_blend35' if p33w and p33s else 'sub_r2_blend35s' if p33s else 'sub_r2_blend34' if s2 else 'sub_r2_blend33' if s1 else 'sub_r2_blend32' if sim_v4 else 'sub_r2_blend31' if v4
               else 'sub_r2_blend30' if sim_v3 else 'sub_r2_blend29')
        log(f'=== STEP 5: Phase 1/2 stacks with gbm_x5 + feats5 -> {b29} ===')
        s10_oof, s10_te, s10_lo, s10_lt = run_aligned_stack(PULL / 'work_preds', clean_models + ['gbm_x5'] + (['cnn_v4pl'] if v4 else [])
                                                            + (['cnn_v4pls1'] if s1 else []) + (['cnn_v4pls2'] if s2 else [])
                                                            + (['cnn_v4pl33'] if p33w else []),
                                                            y, folds, 'sub_r2_stack11_pl_aligned' if v4 else 'sub_r2_stack10_pl_aligned',
                                                            feats5=True)
        sim_ok = (PULL / 'sim1_preds' / 'gbm_x5.npz').exists() and has_feats5(PULL / 'sim1_preds')
        if sim_ok:
            s4b_oof, s4b_te, _, _ = run_aligned_stack(
                PULL / 'sim1_preds', sim_models + ['gbm_x5'] + (['cnn_v3pl'] if sim_v3 else []) + (['cnn_v4pl'] if sim_v4 else [])
                + (['cnn_v4pls1'] if s1 else []) + (['cnn_v4pls2'] if s2 else []) + (['cnn_v4pl33'] if p33s else []),
                y, folds, 'sub_r2_sim6_aligned' if sim_v4 else 'sub_r2_sim5_aligned' if sim_v3 else 'sub_r2_sim4b_aligned', feats5=True)
        else:
            log('sim1_preds has no gbm_x5/feats5 yet -> blend29 keeps sim4_aligned for the PKS-sim world')
            s4b_oof, s4b_te = s4_oof, s4_te
        s10_al = {'oof': s10_oof, 'test': s10_te}
        s10_lg = {'oof': s10_lo, 'test': s10_lt}
        s4b_al = {'oof': s4b_oof, 'test': s4b_te}
        inspect_sub('stack11_pl_aligned' if v4 else 'stack10_pl_aligned', s10_oof, s10_te, y, b16)
        inspect_sub('sim4b_aligned' if sim_ok else 'sim4_aligned (reused)', s4b_oof, s4b_te, y, b16)
        b29_oof, b29_te = blend_lin_geom([s10_al, s10_lg, s6, spl, s4b_al, sim2, sim1], [1.0, 1.0, 0.7, 0.9, 1.4, 0.8, 0.6])
        inspect_sub(f'{b29} (PHASE1)', b29_oof, b29_te, y, b16)
        log(f'{b29} labels different from {out_tag}: {(b29_te.argmax(1) != b27_te.argmax(1)).sum()}')
        for prev in ('sub_r2_blend29', 'sub_r2_blend30', 'sub_r2_blend31', 'sub_r2_blend32', 'sub_r2_blend33', 'sub_r2_blend34', 'sub_r2_blend35s'):
            if prev != b29 and (PULL / f'{prev}_probs.npz').exists():
                log(f'{b29} labels different from {prev}: '
                    f'{(b29_te.argmax(1) != np.load(PULL / f"{prev}_probs.npz")["test"].argmax(1)).sum()}')
        gate(b29, b29_te)
        write_sub(b29, b29_te)
    else:
        log('work_preds has no gbm_x5.npz + feats5_*.parquet yet -> sub_r2_blend29 skipped')


if __name__ == '__main__':
    main()
