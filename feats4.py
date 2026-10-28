"""Step 3c - High-impact physics features targeting the remaining error modes:
1. Fine-grid global ETI track search (201 slopes, step 0.012 px/row, 1-px and 3-px widths) so narrowband ETI tracks
   never fall between the coarse 49-slope grid points across the ~95-row cadence span.
2. Exact Slew-Gap Collinearity & Trap Detector: since ON panels 0, 2, 4 are equally spaced in time, any straight
   track satisfies c_2 = 0.5 * (c_0 + c_4) and s_global = (c_4 - c_0) / (32 + 2 * gap). Directly separates genuine
   ETI from the "same drift rate in ON panels, but not connected" rfi_intermittent trap.
3. Fade-tolerant Persistent RFI (persist_5of6, persist_4of6): catches rfi_persistent lines that fade in one panel.
4. RFI- & Glitch-Masked Scintillation: masks narrowband lines and 1-row PKS glitches before measuring multi-scale
   ON vs OFF blotchy patchiness and ON1-ON3-ON5 spectral cross-correlation.
Output: WORK/feats4_<split>.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common2 import N_THREADS, NF, NP, NT, log, parser, slope_scale
from feats2 import OFFP, ONP, line_search, track_search


def track_search_3px(x, gaps, slopes, c0s):
    """3-channel max-pooled track search: catches ETI tracks that straddle adjacent channels or have slight curvature."""
    x3 = F.max_pool1d(x.reshape(-1, 1, NF), kernel_size=3, stride=1, padding=1).reshape(x.shape)
    return track_search(x3, gaps, slopes, c0s)


@torch.no_grad()
def compute_targeted_features(norm_x, raw_x, gaps, scale=1.0):
    B, dev = norm_x.shape[0], norm_x.device
    rows = torch.arange(B, device=dev)
    o = {}

    # 1. Fine-grid global ON track search (161 slopes: 3.3x finer than feats2's 49 slopes)
    ts = torch.linspace(-1.2, 1.2, 161, device=dev) * scale
    c0 = torch.arange(-64, 192, device=dev, dtype=torch.float32)
    drift_mask = (ts.abs() >= 0.025 * scale)[None, :, None]

    T = track_search(norm_x, gaps, ts, c0)                                  # (B, 6, S, C)
    on_t, off_t = T[:, ONP], T[:, OFFP]
    onmin, offmax = on_t.min(1).values, off_t.max(1).values
    clean = onmin - offmax.clamp(min=0)
    clean_d = torch.where(drift_mask, clean, torch.full_like(clean, -99))

    best_cd, arg = clean_d.flatten(1).max(1)
    si, ci = arg // len(c0), arg % len(c0)
    on_vals = on_t[rows, :, si, ci]                                         # (B, 3)
    off_vals = off_t[rows, :, si, ci]                                       # (B, 3)

    o['ftrack_clean_drift'] = best_cd
    o['ftrack_clean'] = clean.flatten(1).max(1).values
    o['ftrack_on_min'] = on_vals.min(1).values
    o['ftrack_on_mean'] = on_vals.mean(1)
    o['ftrack_off_max'] = off_vals.max(1).values
    o['ftrack_slope'] = ts[si].abs() / scale
    o['ftrack_on_ratio'] = on_vals.min(1).values / on_vals.max(1).values.clamp(min=1e-2)
    o['ftrack_on_cv'] = on_vals.std(1) / on_vals.mean(1).clamp(min=1e-2)
    o['ftrack_contrast'] = on_vals.mean(1) - off_vals.mean(1)

    # 3-channel pooled track search (handles channel-straddling tracks on PKS)
    T3 = track_search_3px(norm_x, gaps, ts[::2], c0)
    on3, off3 = T3[:, ONP].min(1).values, T3[:, OFFP].max(1).values
    clean3 = on3 - off3.clamp(min=0)
    dmask3 = (ts[::2].abs() >= 0.025 * scale)[None, :, None]
    o['ftrack3_clean_drift'] = torch.where(dmask3, clean3, torch.full_like(clean3, -99)).flatten(1).max(1).values
    o['ftrack3_on_min'] = on3.flatten(1).max(1).values

    # 2. Exact Slew-Gap Collinearity & Trap Detector + Fade-Tolerant Persistent RFI
    sl = torch.linspace(-6, 6, 49, device=dev) * scale
    L = line_search(norm_x, sl)                                             # (B, 6, 49, 128)
    lbest, larg = L.flatten(2).max(2)                                       # (B, 6)
    ls_idx, lc_idx = larg // NF, larg % NF
    l_slope = sl[ls_idx]                                                    # (B, 6) physical-scaled slope
    l_chan = lc_idx.float()                                                 # (B, 6) center channel at panel midpoint

    c_0, c_2, c_4 = l_chan[:, 0], l_chan[:, 2], l_chan[:, 4]
    s_0, s_2, s_4 = l_slope[:, 0], l_slope[:, 2], l_slope[:, 4]
    dt_04 = (2.0 * (NT + gaps)).float()                                     # exact rows between ON0 and ON4 centers
    s_04 = (c_4 - c_0) / dt_04                                              # global slope connecting ON0 and ON4

    o['trap_collinear_err'] = (c_2 - 0.5 * (c_0 + c_4)).abs()               # 0 for true ETI line
    o['trap_slope_match'] = torch.stack([(s_0 - s_04).abs(), (s_2 - s_04).abs(), (s_4 - s_04).abs()], 1).max(1).values / scale
    o['trap_slope_spread'] = torch.stack([(s_0 - s_2).abs(), (s_2 - s_4).abs(), (s_0 - s_4).abs()], 1).max(1).values / scale
    o['trap_ftrack_vs_panel'] = o['ftrack_on_min'] - lbest[:, ONP].min(1).values

    # Persistent RFI that fades in 1 or 2 panels (sorted across the 6 panels per channel)
    z0 = L[:, :, int(sl.abs().argmin())]                                    # (B, 6, 128) zero-drift SNR
    z0_sort = z0.sort(dim=1).values                                         # (B, 6, 128) ascending over panels
    o['persist_5of6'] = z0_sort[:, 1].max(1).values                         # 2nd weakest of 6 panels
    o['persist_4of6'] = z0_sort[:, 2].max(1).values                         # 3rd weakest of 6 panels
    o['persist_off_med'] = z0[:, OFFP].median(1).values.max(1).values

    # Also allow slightly drifting persistent lines (max over small slopes |s| <= 0.5)
    small_s = (sl.abs() <= 0.5 * scale)
    z_small = L[:, :, small_s].amax(2).sort(dim=1).values                   # (B, 6, 128)
    o['persist_5of6_drift'] = z_small[:, 1].max(1).values

    # 3. RFI- & Glitch-Masked Scintillation Features
    # Mask narrowband RFI channels (high column median) and 1-row broadband glitches (high row mean)
    ch_med = norm_x.median(dim=2, keepdim=True).values                      # (B, 6, 1, 128)
    rfi_ch = (ch_med.amax(dim=1, keepdim=True) > 1.75)                      # channels with strong narrow lines
    row_mean = norm_x.mean(dim=-1, keepdim=True)                            # (B, 6, 16, 1)
    glitch_r = (row_mean.abs() > 0.45)                                      # broadband impulse rows
    clean_x = norm_x.masked_fill(rfi_ch | glitch_r, 0.0)

    for bt, bc in ((2, 8), (4, 8), (4, 16), (8, 16)):
        sm = F.avg_pool2d(clean_x.reshape(B * NP, 1, NT, NF), (bt, bc)).reshape(B, NP, -1) * ((bt * bc) ** 0.5)
        tex = sm.std(-1)
        ton, toff = tex[:, ONP].mean(1), tex[:, OFFP].mean(1)
        o[f'cpat{bt}x{bc}_diff'] = ton - toff
        o[f'cpat{bt}x{bc}_min'] = tex[:, ONP].min(1).values - tex[:, OFFP].max(1).values
        o[f'cpat{bt}x{bc}_hot'] = (sm[:, ONP] > 1.2).float().mean((1, 2)) - (sm[:, OFFP] > 1.2).float().mean((1, 2))

    # Spectral cross-correlation across ON1, ON3, ON5 after subtracting OFF bandpass and linear tilt
    raw_v = raw_x.astype(np.float32)
    raw_v[raw_v == 0] = np.nan
    raw_v -= np.nanmedian(raw_v, axis=(2, 3), keepdims=True)
    spec = np.nanmedian(raw_v, axis=2)                                      # (B, 6, 128)
    ref_off = np.nanmedian(spec[:, OFFP], axis=1, keepdims=True)            # (B, 1, 128)
    nspec = np.nan_to_num(spec - ref_off)
    # Mask narrowband spikes before smoothing across 8 channels
    nspec = np.clip(nspec, -12.0, 12.0)
    on_sp = torch.tensor(nspec[:, ONP], device=dev, dtype=torch.float32)    # (B, 3, 128)
    on_sp = F.avg_pool1d(on_sp, kernel_size=5, stride=1, padding=2)
    on_sp = on_sp - on_sp.mean(dim=-1, keepdim=True)
    on_sp = on_sp / (on_sp.norm(dim=-1, keepdim=True).clamp(min=1e-4))
    c01 = (on_sp[:, 0] * on_sp[:, 1]).sum(-1)
    c12 = (on_sp[:, 1] * on_sp[:, 2]).sum(-1)
    c02 = (on_sp[:, 0] * on_sp[:, 2]).sum(-1)
    o['scint_corr_mean'] = (c01 + c12 + c02) / 3.0
    o['scint_corr_min'] = torch.minimum(torch.minimum(c01, c12), c02)

    # 4. PKS Glitch vs Real Dispersed Burst Discriminators
    row_max_snr = norm_x.mean(-1) * (NF ** 0.5)                             # (B, 6, 16) undispersed row SNR
    o['glitch_on_rows'] = (row_max_snr[:, ONP] > 5.0).float().sum((1, 2))
    o['glitch_off_rows'] = (row_max_snr[:, OFFP] > 5.0).float().sum((1, 2))

    return {k: v.float().cpu().numpy() for k, v in o.items()}


def main():
    p = parser(__doc__)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--phys', action='store_true')
    a = p.parse_args()
    data, work = Path(a.data), Path(a.work)
    torch.set_num_threads(N_THREADS)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for split in ('train', 'test'):
        src = work / f'{split}_cadences.npy'
        R = np.load(src if src.exists() else data / f'{split}_cadences.npy', mmap_mode='r')
        X = np.load(work / f'norm_{split}.npy', mmap_mode='r')
        gaps = pd.read_csv(work / f'{split}_meta.csv').slew_gap_rows.values
        scale = slope_scale(work, split) if a.phys else 1.0

        parts = []
        for s in range(0, len(X), a.batch):
            e = slice(s, s + a.batch)
            raw_b = np.asarray(R[e])
            norm_b = torch.tensor(np.asarray(X[e]), device=dev).float() / 4.0
            gap_b = torch.tensor(gaps[e], device=dev, dtype=torch.long)
            f = compute_targeted_features(norm_b, raw_b, gap_b, scale)
            parts.append(pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()}))

        df = pd.concat(parts, ignore_index=True)
        out_path = work / f'feats4_{split}.parquet'
        df.to_parquet(out_path, index=False)
        log(f'{split}: {df.shape[1]} feats4 features written to {out_path} ({dev})')


if __name__ == '__main__':
    main()
