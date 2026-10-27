"""Step 3b - extra features (CPU is fine): scintillation against an OFF-panel bandpass, and how well the ETI track fits.

S  prep2.py removes each panel's OWN smooth bandpass, which also removes broad ON-only brightening (scintillation).
   Here the bandpass comes from the OFF panels of the same cadence (same receiver, so ripple / roll-off / tilt cancel;
   the problem statement suggests flattening per cadence) and every panel is compared with it: broad spectral excess
   and texture at several block sizes, ON vs OFF. Each OFF panel is compared with the other two OFF panels.
T  track fit: on the best clean drifting global track, the track's SNR in each ON panel divided by the best local
   line near that position (~1 when the three ON lines really lie on one track, lower for the "same drift, not
   connected" trap), the drift mismatch of those local lines, and the strongest OFF line near the track.
Output: WORK/feats3_<split>.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter1d

from common2 import N_THREADS, NF, NP, NT, log, parser, slope_scale
from feats2 import OFFP, ONP, line_search, track_search

SCALES = [(2, 4), (4, 8), (8, 16), (2, 32), (16, 8)]                 # (time rows, channels) block sizes


def scint_features(raw):
    v = raw.astype(np.float32)
    v[v == 0] = np.nan                                                  # dropped rows
    v -= np.nanmedian(v, axis=(2, 3), keepdims=True)                    # panel level (gain changes)
    spec = np.nanmedian(v, axis=2)                                      # (B,6,128) time-median spectrum
    ref = np.empty_like(spec)
    ref[:, ONP] = np.nanmedian(spec[:, OFFP], axis=1)[:, None]          # ON: median of the 3 OFF panels
    for j in OFFP:
        ref[:, j] = np.nanmean(spec[:, [q for q in OFFP if q != j]], axis=1)
    z = v - ref[:, :, None, :]
    zo = z[:, OFFP].reshape(len(z), -1)
    sig = 1.4826 * np.nanmedian(np.abs(zo - np.nanmedian(zo, axis=1, keepdims=True)), axis=1)
    z = np.nan_to_num(z / np.maximum(sig, 1e-3)[:, None, None, None])  # (B,6,16,128) in OFF-noise units
    o = {}

    def agg(name, m):                                                   # m (B,6): one value per panel
        on, off = m[:, ONP], m[:, OFFP]
        o[f'{name}_on_mean'], o[f'{name}_on_min'] = on.mean(1), on.min(1)
        o[f'{name}_off_max'], o[f'{name}_diff'] = off.max(1), on.mean(1) - off.mean(1)

    e = uniform_filter1d(np.median(z, axis=2), 8, axis=-1)              # broad spectral excess (8-channel smooth)
    agg('exc_max', e.max(-1))
    agg('exc_area', np.clip(e, 0, None).mean(-1))
    B = len(z)
    for bt, bc in SCALES:
        blk = z.reshape(B, NP, NT // bt, bt, NF // bc, bc).mean((3, 5)).reshape(B, NP, -1) * np.sqrt(bt * bc)
        agg(f'tex{bt}x{bc}', blk.std(-1))                               # ~1 for pure noise
        agg(f'hot{bt}x{bc}', (blk > 2).mean(-1))
    return o


@torch.no_grad()
def track_features(x, gaps, scale=1.0):
    B, dev = x.shape[0], x.device
    sl = torch.linspace(-6, 6, 49, device=dev) * scale                  # physical drift units (see slope_scale)
    Lbest, Larg = line_search(x, sl).max(2)                             # (B,6,128) best line through each channel
    ts = torch.linspace(-1.2, 1.2, 49, device=dev) * scale
    c0 = torch.arange(-64, 192, device=dev, dtype=torch.float32)
    T = track_search(x, gaps, ts, c0)                                   # (B,6,S,C)
    clean = T[:, ONP].min(1).values - T[:, OFFP].max(1).values.clamp(min=0)
    clean = torch.where((ts.abs() >= 0.05 * scale)[None, :, None], clean, torch.full_like(clean, -99))
    best, arg = clean.flatten(1).max(1)
    si, ci = arg // len(c0), arg % len(c0)
    s, c = ts[si], c0[ci]
    rows = torch.arange(B, device=dev)
    win = torch.arange(-4, 5, device=dev)
    ratio, dslope, near = [], [], []
    for p in range(NP):
        centre = (p * (NT + gaps) + (NT - 1) / 2 - 63).float()          # global row of the panel's centre
        idx = (torch.round(c + s * centre).long()[:, None] + win[None]).clamp(0, NF - 1)   # (B,9) channels
        loc, j = Lbest[:, p].gather(1, idx).max(1)
        if p in ONP:
            ratio.append(T[rows, p, si, ci] / loc.clamp(min=1.0))
            dslope.append((sl[Larg[:, p].gather(1, idx)[rows, j]] - s).abs() / scale)
        else:
            near.append(loc)
    ratio, dslope, near = torch.stack(ratio, 1), torch.stack(dslope, 1), torch.stack(near, 1)
    return {'tfit_clean': best, 'tfit_ratio_min': ratio.min(1).values, 'tfit_ratio_mean': ratio.mean(1),
            'tfit_dslope_max': dslope.max(1).values, 'tfit_off_near': near.max(1).values}


def main():
    p = parser(__doc__)
    p.add_argument('--batch', type=int, default=500)
    p.add_argument('--phys', action='store_true', help='slope grids in physical drift units (tsamp / channel width)')
    a = p.parse_args()
    data, work = Path(a.data), Path(a.work)
    torch.set_num_threads(N_THREADS)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for split in ('train', 'test'):
        src = work / f'{split}_cadences.npy'                            # degraded copy from prep2.py --sim, if any
        R = np.load(src if src.exists() else data / f'{split}_cadences.npy', mmap_mode='r')
        X = np.load(work / f'norm_{split}.npy', mmap_mode='r')
        gaps = pd.read_csv(work / f'{split}_meta.csv').slew_gap_rows.values
        scale = slope_scale(work, split) if a.phys else 1.0
        assert len(R) == len(X) == len(gaps), (len(R), len(X), len(gaps))
        parts = []
        for s in range(0, len(X), a.batch):
            e = slice(s, s + a.batch)
            f = scint_features(np.asarray(R[e]))
            x = torch.tensor(np.asarray(X[e]), device=dev).float() / 4
            g = torch.tensor(gaps[e], device=dev, dtype=torch.long)
            f.update({k: v.float().cpu().numpy() for k, v in track_features(x, g, scale).items()})
            parts.append(pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()}))
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(work / f'feats3_{split}.parquet', index=False)
        log(f'{split}: {df.shape[1]} extra features for {len(df)} cadences ({dev})')


if __name__ == '__main__':
    main()
