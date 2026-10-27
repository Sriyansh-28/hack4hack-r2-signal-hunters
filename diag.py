"""Diagnostics (CPU, ~1 min): how different does the PKS test set look from GBT train?
Prints the label mixes, the raw noise level, cleaned-data artefacts (channel stripes, band edges, tails, dropped rows)
and, per class, key feature medians for train vs confidently predicted test cadences
(ratio < 1 = signals look weaker on PKS; confident test cadences are biased towards bright ones)."""
from pathlib import Path

import numpy as np
import pandas as pd

from common2 import CLASSES, labels, load_feats, parser

GLOBAL = ['row_on_max', 'row_off_max', 'disp_on_max', 'disp_off_max', 'line_on_max', 'line_off_max', 'line_slope_p0',
          'persist_all6', 'steep_max', 'std_on', 'max_on', 'patch_on', 'tex4x8_on_mean', 'exc_max_on_mean',
          'track_slope', 'track_on_min', 'track_clean_drift', 'tfit_ratio_min']
KEY = {'eti_candidate': ['track_on_min', 'track_clean_drift', 'line_on_min', 'track_slope'],
       'frb': ['disp_on_max', 'row_on_max'],
       'pulsar': ['disp_on_max', 'disp_on_pulses'],
       'scintillating_star': ['exc_area_diff', 'tex4x8_diff', 'patch_diff'],
       'satellite_pass': ['steep_max'],
       'radar_chirp': ['steep_max', 'steep_repeats'],
       'rfi_intermittent': ['line_on_max', 'line_off_max'],
       'rfi_persistent': ['persist_all6', 'line_on_min'],
       'noise': ['line_on_max', 'max_on', 'std_on', 'track_on_min', 'exc_max_on_mean', 'tex4x8_on_mean']}


def raw_noise(R):
    v = np.asarray(R).astype(np.float32)
    v[v == 0] = np.nan
    v -= np.nanmedian(v, axis=2, keepdims=True)                     # remove each channel's time median
    return np.nanmedian(np.abs(v), axis=(2, 3))                     # (B,6) MAD per panel, raw units


def clean_stats(Xn):
    s = np.asarray(Xn).astype(np.float32) / 4                       # sigma units
    prof = s.mean(2)                                                # (B,6,128) time-mean per channel
    return {'stripe_std': np.median(prof.std(-1)), 'edge_lo': prof[..., :3].mean(), 'edge_hi': prof[..., -3:].mean(),
            'tail>4sig_%': (np.abs(s) > 4).mean() * 100, 'zero_rows_%': (np.abs(s).max(-1) == 0).mean() * 100}


def main():
    p = parser(__doc__)
    p.add_argument('--probs', default='sub_r2_stack6_pl_probs.npz')
    a = p.parse_args()
    data, work = Path(a.data), Path(a.work)
    y = labels(work)
    pt = np.load(work / a.probs)['test']
    yt, conf = pt.argmax(1), pt.max(1) >= 0.9
    print(pd.DataFrame({'train_%': np.bincount(y, minlength=9) / len(y) * 100,
                        'test_pred_%': np.bincount(yt, minlength=9) / len(yt) * 100,
                        'test_conf_n': np.bincount(yt[conf], minlength=9)}, index=CLASSES).round(1).to_string())
    for split in ('train', 'test'):
        R = np.load(data / f'{split}_cadences.npy', mmap_mode='r')[:2000]
        Xn = np.load(work / f'norm_{split}.npy', mmap_mode='r')[:2000]
        drop = (np.asarray(R).max(-1) == 0).mean() * 100
        print(f'{split}: raw noise MAD {np.median(raw_noise(R)):.2f}  raw dropped rows {drop:.1f}%  '
              + '  '.join(f'{k} {v:.3f}' for k, v in clean_stats(Xn).items()))
    ftr, fte = load_feats(work, True)
    print('ALL cadences: feature            train p50 / p90      test p50 / p90')
    for f in GLOBAL:
        q1, q2 = np.percentile(ftr[f].values, [50, 90]), np.percentile(fte[f].values, [50, 90])
        print(f'  {f:22s} {q1[0]:8.2f} / {q1[1]:8.2f}   {q2[0]:8.2f} / {q2[1]:8.2f}')
    for k, c in enumerate(CLASSES):
        tr, te = y == k, (yt == k) & conf
        for f in KEY[c]:
            m1 = np.median(ftr[f].values[tr])
            m2 = np.median(fte[f].values[te]) if te.any() else np.nan
            print(f'{c[:14]:14s} {f:16s} train {m1:8.2f}  test {m2:8.2f}  ratio {m2 / m1 if m1 else np.nan:5.2f}  (n test {te.sum()})')


if __name__ == '__main__':
    main()
