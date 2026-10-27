"""Step 1 - clean every cadence so both telescopes look alike.

Per panel: subtract a smooth bandpass (running median over channels of the time-median spectrum, so narrow
lines survive), remove the panel level and divide by its robust noise (fixes gain changes and PKS's higher
noise), blank the GBT-only centre channel (64) and set dropped rows to 0. Stored as int8 in 0.25-sigma steps.
Writes WORK/norm_<split>.npy, <split>_meta.csv and folds.npy (5 stratified folds).
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.model_selection import StratifiedKFold

from common2 import NP, NT, log, parser


def degrade(x, rng, lo=1.15, hi=1.45, extra_drop=0.003, glitch_p=0.0):
    """Make GBT cadences look like PKS: noise x lo..hi (PKS raw noise is 1.31x GBT, so its signals look ~0.76x as
    strong after normalisation), more dropped rows (PKS 0.4% vs GBT 0.1%) and, with probability glitch_p, one extra
    broadband glitch row (+1.5..3.5 sigma in a random panel; ~45% of PKS cadences have one in an ON panel vs ~27% of
    GBT; glitches do not change the label). uint8 in, uint8 out, 0 = dropped."""
    miss = x == 0
    v = x.astype(np.float32)
    v[miss] = np.nan
    sig = 1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v, axis=2, keepdims=True)), axis=(2, 3), keepdims=True)
    f = rng.uniform(lo, hi, size=(len(x), 1, 1, 1)).astype(np.float32)
    v += rng.standard_normal(v.shape, dtype=np.float32) * sig * np.sqrt(f ** 2 - 1)
    if glitch_p > 0:
        g = np.where(rng.random(len(x)) < glitch_p)[0]
        pn, rw = rng.integers(0, NP, len(g)), rng.integers(0, NT, len(g))
        v[g, pn, rw, :] += (rng.uniform(1.5, 3.5, len(g)).astype(np.float32) * np.nan_to_num(sig[g, pn, 0, 0]))[:, None]
    out = np.clip(np.round(np.nan_to_num(v, nan=0.0)), 1, 255).astype(np.uint8)
    out[miss | np.broadcast_to(rng.random((len(x), NP, NT, 1)) < extra_drop, x.shape)] = 0
    return out


def normalize(x):
    v = x.astype(np.float32)
    v[v == 0] = np.nan                                            # dropped rows
    spec = np.nanmedian(v, axis=2)                                # (B, 6, 128) time-median spectrum
    spec = np.where(np.isnan(spec), np.nanmedian(spec, axis=-1, keepdims=True), spec)
    v -= median_filter(spec, size=(1, 1, 15), mode='nearest')[:, :, None, :]
    med = np.nanmedian(v, axis=(2, 3), keepdims=True)
    sig = 1.4826 * np.nanmedian(np.abs(v - med), axis=(2, 3), keepdims=True)
    v = (v - med) / np.maximum(sig, 1e-3)
    v[:, :, :, 64] = 0                                            # GBT-only centre-channel spike
    v = np.nan_to_num(v, nan=0.0)
    return np.clip(np.round(v * 4), -32, 127).astype(np.int8)


def main():
    p = parser(__doc__)
    p.add_argument('--debug', type=int, default=0, help='keep only N train / N/2 test cadences')
    p.add_argument('--sim', action='store_true',
                   help='make the TRAIN cadences PKS-like first (degrade); also saves them as WORK/train_cadences.npy')
    p.add_argument('--noise_lo', type=float, default=1.15)
    p.add_argument('--noise_hi', type=float, default=1.45)
    p.add_argument('--glitch_p', type=float, default=0.0)
    a = p.parse_args()
    data, work = Path(a.data), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    for split in ('train', 'test'):
        X = np.load(data / f'{split}_cadences.npy', mmap_mode='r')
        meta = pd.read_csv(data / f'{split}_meta.csv')
        n = len(meta) if not a.debug else (a.debug if split == 'train' else a.debug // 2)
        out = np.lib.format.open_memmap(work / f'norm_{split}.npy', mode='w+', dtype=np.int8, shape=(n,) + X.shape[1:])
        raw = None
        if a.sim and split == 'train':
            raw = np.lib.format.open_memmap(work / 'train_cadences.npy', mode='w+', dtype=np.uint8, shape=(n,) + X.shape[1:])
        for s in range(0, n, 2000):
            xb = np.asarray(X[s:min(s + 2000, n)])
            if raw is not None:
                xb = degrade(xb, rng, a.noise_lo, a.noise_hi, glitch_p=a.glitch_p)
                raw[s:s + len(xb)] = xb
            out[s:s + len(xb)] = normalize(xb)
        out.flush()
        if raw is not None:
            raw.flush()
            log('train cadences degraded to PKS-like noise')
        meta.iloc[:n].to_csv(work / f'{split}_meta.csv', index=False)
        log(f'{split}: {n} cadences cleaned')
    y = pd.read_csv(work / 'train_meta.csv').label.values
    folds = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y)):
        folds[va] = k
    np.save(work / 'folds.npy', folds)
    shutil.copy(data / 'sample_submission.csv', work / 'sample_submission.csv')
    log('done')


if __name__ == '__main__':
    main()
