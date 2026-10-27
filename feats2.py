"""Step 3 - physics features for each cadence (runs on the GPU slice, or CPU).

A  narrow-line search per panel: best straight line (start channel x drift) over slopes -6..6 px/row
B  global ON-track search: one straight track through the REAL time of the three ON panels (slew gaps
   included); for each track take the weakest ON panel (all three must show it) and the strongest OFF panel
   on the same track -> the core ETI test; also its drift (zero drift = interference)
C  steep lines (satellites, radar sweeps): slopes 6..40 px/row, and how many repeats per panel
D  broadband dispersed sweeps (pulsar, FRB): curved de-dispersion, pulses per panel, ON vs OFF
E  patchiness (scintillation): smoothed-image texture, ON vs OFF
F  persistent lines: same channel bright in all 6 panels
Output: WORK/feats_<split>.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common2 import N_THREADS, NF, NP, NT, log, parser, slope_scale

ONP, OFFP = [0, 2, 4], [1, 3, 5]
CURVES = [0, 2, 4, 6, 9, 12]                                   # de-dispersion delays (rows at the lowest channel)
CURVES_FINE = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 9, 12]               # PKS pulses are much less dispersed (pulsars ~0)


def line_search(x, slopes):
    """x (B,6,16,128) -> SNR of the best straight line per panel for each slope: (B,6,S,128)."""
    dev = x.device
    c = torch.arange(NF, device=dev)
    r = torch.arange(NT, device=dev, dtype=torch.float32) - (NT - 1) / 2
    out = []
    for s in slopes:
        idx = (c[None, :] + torch.round(s * r)[:, None]).long()              # (16, 128)
        ok = ((idx >= 0) & (idx < NF)).float()
        g = x.gather(-1, idx.clamp(0, NF - 1)[None, None].expand(x.shape[0], NP, -1, -1)) * ok
        out.append(g.sum(2) / ok.sum(0).clamp(min=1).sqrt())
    return torch.stack(out, 2)


def track_search(x, gaps, slopes, c0s):
    """Global straight tracks: channel = c0 + s * (global_row - 63). Returns per-panel segment SNR (B,6,S,C)."""
    B, dev = x.shape[0], x.device
    p = torch.arange(NP, device=dev).repeat_interleave(NT)
    r = torch.arange(NT, device=dev).repeat(NP)
    grow = (p[None] * (NT + gaps[:, None]) + r[None]).float() - 63.0          # (B, 96)
    xf = x.reshape(B, NP * NT, NF)
    res = []
    for s in slopes:
        ch = torch.round(c0s[None, None, :] + s * grow[:, :, None]).long()    # (B, 96, C)
        ok = ((ch >= 0) & (ch < NF)).float()
        v = xf.gather(-1, ch.clamp(0, NF - 1)) * ok                           # (B, 96, C)
        v, ok = v.reshape(B, NP, NT, -1), ok.reshape(B, NP, NT, -1)
        res.append(v.sum(2) / ok.sum(2).clamp(min=1).sqrt() * (ok.sum(2) >= 4))
    return torch.stack(res, 2)


def dedisperse(x, curves):
    """Sum over channels along curved sweeps (later at lower frequency). -> (B,6,A,16) row SNR."""
    dev = x.device
    c = torch.arange(NF, device=dev, dtype=torch.float32)
    out = []
    for a in curves:
        delay = torch.round(a * ((NF - 1 - c) / (NF - 1)) ** 2).long()        # (128,)
        rows = torch.arange(NT, device=dev)[:, None] + delay[None]            # (16, 128) row of each channel
        ok = (rows < NT).float()
        g = x.gather(2, rows.clamp(max=NT - 1)[None, None].expand(x.shape[0], NP, -1, -1)) * ok
        out.append(g.sum(-1) / ok.sum(-1).clamp(min=1).sqrt())
    return torch.stack(out, 2)


@torch.no_grad()
def features(x, gaps, scale=1.0, curves=CURVES):
    """scale = slope_scale() of the telescope: slope grids in physical drift units, slopes reported in GBT pixels."""
    o = {}
    B = x.shape[0]
    # A: per-panel narrow lines
    sl = torch.linspace(-6, 6, 49) * scale
    L = line_search(x, sl)                                                    # (B,6,S,128)
    best, arg = L.flatten(2).max(2)
    for pnl in range(NP):
        o[f'line_snr_p{pnl}'] = best[:, pnl]
        o[f'line_slope_p{pnl}'] = sl.to(x.device)[arg[:, pnl] // NF].abs() / scale
    on_b, off_b = best[:, ONP], best[:, OFFP]
    o.update(line_on_min=on_b.min(1).values, line_on_max=on_b.max(1).values, line_off_max=off_b.max(1).values,
             line_off_min=off_b.min(1).values, line_n_panels=(best > 7).sum(1).float())
    z0 = L[:, :, int(sl.abs().argmin())]                                         # zero-drift column profiles
    o['zero_on_min'] = z0[:, ONP].amax(-1).min(1).values
    # F: persistent line = same channel bright in all 6 panels / in all ON only
    o['persist_all6'] = z0.min(1).values.max(1).values
    o['persist_on3'] = z0[:, ONP].min(1).values.max(1).values
    o['persist_off3'] = z0[:, OFFP].min(1).values.max(1).values
    # B: global ON track
    ts = torch.linspace(-1.2, 1.2, 49, device=x.device) * scale
    c0 = torch.arange(-64, 192, device=x.device, dtype=torch.float32)
    T = track_search(x, gaps, ts, c0)                                         # (B,6,S,C)
    onmin = T[:, ONP].min(1).values                                           # weakest ON panel on each track
    offmax = T[:, OFFP].max(1).values
    score, arg = onmin.flatten(1).max(1)
    sidx, cidx = arg // len(c0), arg % len(c0)
    rows = torch.arange(B, device=x.device)
    o['track_on_min'] = score
    o['track_off_max'] = offmax[rows, sidx, cidx]
    o['track_slope'] = ts[sidx].abs() / scale
    o['track_on_mean'] = T[:, ONP].mean(1)[rows, sidx, cidx]
    clean = onmin - offmax.clamp(min=0)
    o['track_clean'] = clean.flatten(1).max(1).values                         # ON on track, OFF quiet on it
    drift_mask = (ts.abs() >= 0.05 * scale)[None, :, None]
    o['track_clean_drift'] = torch.where(drift_mask, clean, torch.full_like(clean, -99)).flatten(1).max(1).values
    o['track_vs_panels'] = score - on_b.min(1).values                         # does the track connect the lines?
    # C: steep lines (satellites, radar)
    st = torch.cat([torch.linspace(-40, -6, 18), torch.linspace(6, 40, 18)]) * scale
    S = line_search(x, st)                                                    # (B,6,36,128)
    sbest = S.flatten(2).max(2).values
    for pnl in range(NP):
        o[f'steep_snr_p{pnl}'] = sbest[:, pnl]
    o['steep_n_panels'] = (sbest > 7).sum(1).float()
    o['steep_max'] = sbest.max(1).values
    peaks = (S.amax(2) > 6).float()                                           # channels with a steep sweep
    o['steep_repeats'] = (peaks[..., 1:] - peaks[..., :-1]).clamp(min=0).sum(-1).mean(1)
    # D: dispersed broadband sweeps
    D = dedisperse(x, curves)                                                 # (B,6,A,16)
    dbest = D.amax(2)                                                          # (B,6,16)
    for pnl in range(NP):
        o[f'disp_max_p{pnl}'] = dbest[:, pnl].max(1).values
    o['disp_on_max'] = dbest[:, ONP].amax((1, 2))
    o['disp_off_max'] = dbest[:, OFFP].amax((1, 2))
    o['disp_on_panels'] = (dbest[:, ONP].amax(2) > 6).sum(1).float()
    o['disp_on_pulses'] = (dbest[:, ONP] > 6).sum((1, 2)).float()
    o['disp_curve'] = D.amax(-1)[:, ONP].amax(1).argmax(1).float()
    rowmean = x.mean(-1)                                                       # undispersed broadband rows
    o['row_on_max'] = rowmean[:, ONP].amax((1, 2)) * NF ** 0.5
    o['row_off_max'] = rowmean[:, OFFP].amax((1, 2)) * NF ** 0.5
    # E: patchiness
    sm = F.avg_pool2d(x.reshape(B * NP, 1, NT, NF), (4, 8)).reshape(B, NP, -1)
    tex = sm.std(-1)
    o['patch_on'] = tex[:, ONP].mean(1)
    o['patch_off'] = tex[:, OFFP].mean(1)
    o['patch_diff'] = o['patch_on'] - o['patch_off']
    o['patch_on_frac'] = (sm[:, ONP] > 1.0).float().mean((1, 2))
    o['patch_off_frac'] = (sm[:, OFFP] > 1.0).float().mean((1, 2))
    # panel-level basics
    ps = x.std((2, 3))
    o['std_on'], o['std_off'] = ps[:, ONP].mean(1), ps[:, OFFP].mean(1)
    pm = x.amax((2, 3))
    o['max_on'], o['max_off'] = pm[:, ONP].mean(1), pm[:, OFFP].mean(1)
    return {k: v.float().cpu().numpy() for k, v in o.items()}


def main():
    p = parser(__doc__)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--phys', action='store_true', help='slope grids in physical drift units (tsamp / channel width)')
    p.add_argument('--fine_disp', action='store_true', help='finer de-dispersion grid near 0 (CURVES_FINE)')
    a = p.parse_args()
    curves = CURVES_FINE if a.fine_disp else CURVES
    work = Path(a.work)
    torch.set_num_threads(N_THREADS)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for split in ('train', 'test'):
        X = np.load(work / f'norm_{split}.npy', mmap_mode='r')
        gaps = pd.read_csv(work / f'{split}_meta.csv').slew_gap_rows.values
        scale = slope_scale(work, split) if a.phys else 1.0
        parts = []
        for s in range(0, len(X), a.batch):
            x = torch.tensor(np.asarray(X[s:s + a.batch]), device=dev).float() / 4
            g = torch.tensor(gaps[s:s + a.batch], device=dev, dtype=torch.long)
            parts.append(pd.DataFrame(features(x, g, scale, curves)))
        df = pd.concat(parts, ignore_index=True)
        df['slew_gap_rows'] = gaps
        df.to_parquet(work / f'feats_{split}.parquet', index=False)
        log(f'{split}: {df.shape[1]} features for {len(df)} cadences ({dev})')


if __name__ == '__main__':
    main()
