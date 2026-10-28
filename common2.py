"""Shared settings for the Round 2 (Signal Hunters) pipeline."""
import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=RuntimeWarning)   # all-NaN slices from dropped rows

ROOT = Path(__file__).resolve().parent.parent                 # the R2 folder
# label list in the official priority order (index = class id)
CLASSES = ['eti_candidate', 'frb', 'pulsar', 'scintillating_star', 'satellite_pass', 'radar_chirp',
           'rfi_intermittent', 'rfi_persistent', 'noise']
CLS2ID = {c: i for i, c in enumerate(CLASSES)}
N_THREADS = int(os.environ.get('N_THREADS', '8'))
NP, NT, NF = 6, 16, 128                                       # panels, time rows, channels
_T0 = time.time()


def log(*a):
    print(f'[{time.time() - _T0:7.1f}s]', *a, flush=True)


def parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument('--data', default=str(ROOT / 'data'))
    p.add_argument('--work', default=str(ROOT / 'work'))
    return p


def labels(work):
    return pd.read_csv(Path(work) / 'train_meta.csv').label.map(CLS2ID).values


def slope_scale(work, split):
    """Drift in pixels per row of this split's telescope relative to GBT, from tsamp_s / channel_width_hz
    (PKS: (16 / 3.81) / (18.25 / 2.79) = 0.64). Slope grids are multiplied by it, so features measure the same
    physical drift rates on both telescopes."""
    m = pd.read_csv(Path(work) / f'{split}_meta.csv', usecols=['tsamp_s', 'channel_width_hz'])
    r = m.tsamp_s / m.channel_width_hz
    log(f'{split}: slope scale {r.min() / (18.25 / 2.79):.3f} .. {r.max() / (18.25 / 2.79):.3f}')
    return float(r.median() / (18.25 / 2.79))


def damped_quantile_align(src_df, ref_df, alpha=0.85):
    """Align each feature column of src_df (PKS test) toward ref_df (train) empirical quantiles.
    Removes telescope receiver-noise scaling (~0.70x SNR) and 1-row broadband glitch offset
    (disp_on_max median 8.95 -> 2.96) while leaving (1 - alpha) slack for label-mix differences."""
    out = pd.DataFrame(index=src_df.index, columns=src_df.columns, dtype=np.float32)
    ranks = np.linspace(0.0, 100.0, len(src_df))
    for col in src_df.columns:
        s = np.nan_to_num(src_df[col].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        r = np.nan_to_num(ref_df[col].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(s, kind='mergesort')
        mapped = np.percentile(r, ranks).astype(np.float32)
        aligned = np.empty_like(s)
        aligned[order] = alpha * mapped + (1.0 - alpha) * s[order]
        out[col] = aligned
    return out


def load_feats(work, extra=False, feats4=False, feats5=False, align_test=True):
    """Features for train and test: feats2 + feats3 (if extra) + feats4 (if feats4) + feats5 (if feats5)."""
    work = Path(work)
    tr, te = pd.read_parquet(work / 'feats_train.parquet'), pd.read_parquet(work / 'feats_test.parquet')
    if extra:
        tr = pd.concat([tr, pd.read_parquet(work / 'feats3_train.parquet')], axis=1)
        te = pd.concat([te, pd.read_parquet(work / 'feats3_test.parquet')], axis=1)
    if feats4 and (work / 'feats4_train.parquet').exists():
        tr = pd.concat([tr, pd.read_parquet(work / 'feats4_train.parquet')], axis=1)
        te = pd.concat([te, pd.read_parquet(work / 'feats4_test.parquet')], axis=1)
    if feats5 and (work / 'feats5_train.parquet').exists():
        tr = pd.concat([tr, pd.read_parquet(work / 'feats5_train.parquet')], axis=1)
        te = pd.concat([te, pd.read_parquet(work / 'feats5_test.parquet')], axis=1)
    te = te[tr.columns]
    if align_test:
        te = damped_quantile_align(te, tr, alpha=0.85)
    return tr, te


def macro_f1(y, p, n=len(CLASSES)):
    cm = np.bincount(y * n + p, minlength=n * n).reshape(n, n)
    tp = np.diag(cm).astype(float)
    pr, rc = tp / np.maximum(cm.sum(0), 1), tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return f1.mean(), f1


def report(y, prob, title):
    s, per = macro_f1(y, prob.argmax(1))
    ll = -np.log(np.clip(prob[np.arange(len(y)), y], 1e-7, 1)).mean()
    log(f'{title} macro-F1 {s:.4f} logloss {ll:.4f}')
    print('   ' + '  '.join(f'{c[:8]}={f:.3f}' for c, f in zip(CLASSES, per)), flush=True)
    return s


def save_preds(work, name, oof, test, score):
    d = Path(work) / 'preds'
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f'{name}.npz', oof=oof.astype(np.float32), test=test.astype(np.float32), score=score)
    log(f'saved {d / (name + ".npz")}')
