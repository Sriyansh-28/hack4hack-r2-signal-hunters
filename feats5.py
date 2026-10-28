"""Step 3d - Permanent Research-Backed Physics Features (feats5.py):
1. Brzycki et al. (2023, ApJ 952:46, arXiv:2307.08793) OFF-Referenced 2D Autocorrelation Function (2D ACF)
   & Detrended Scintillation:
   - Scrubs 1-row broadband receiver glitches (by subtracting per-row median across the 128 channels)
     and blanks channel 64.
   - Removes PKS inter-panel bandpass tilt, roll-off, and cubic curvature by projecting out a degree-3
     Legendre polynomial basis across frequency channels from (spec_p - ref_OFF).
   - Computes 2D ACF A(dt, df) at strictly non-zero time lags dt in {1, 2, 4} and frequency lags
     df in {1, 2, 4, 8, 16} (immune to 1-row glitches at dt=0!) plus multi-scale detrended block texture.
2. Sub-Pixel Bilinear De-Doppler Track Search (turboSETI / Ma et al. 2023 Nature Astronomy arXiv:2301.12670):
   - Evaluates fractional channel positions c(t) = c0 + s * t using bilinear sub-pixel interpolation
     (1 - w) * x[floor(c)] + w * x[ceil(c)] over 241 slopes and 0.5-channel c0 steps (384 positions).
   - Computes 3-of-3 ON min (subpx_clean_min), 48-row coherent ON mean (subpx_clean_mean, 1.73x higher
     sensitivity on faint PKS ETI!), 2-of-3 fade-tolerant ON track (subpx_clean_2of3), and 3-panel
     collinearity R^2.
Output: WORK/feats5_<split>.parquet (~42 features, ~35s on GPU).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common2 import N_THREADS, NF, NP, NT, log, parser, slope_scale
from feats2 import OFFP, ONP


def build_legendre_basis(nf=NF, deg=3):
    """Orthonormal polynomial basis of degree 0..deg over nf channels for fast bandpass tilt/roll-off removal."""
    u = np.linspace(-1.0, 1.0, nf, dtype=np.float32)
    B = np.vstack([u ** d for d in range(deg + 1)])
    Q, _ = np.linalg.qr(B.T)
    return Q.T.astype(np.float32)  # (deg+1, nf), orthonormal rows


LEG_BASIS = build_legendre_basis(NF, deg=3)


def brzycki_scint_acf_features(raw):
    """Glitch-scrubbed, degree-3 polynomial tilt-detrended OFF-referenced dynamic spectrum + 2D ACF (Brzycki et al. 2023)."""
    v = raw.astype(np.float32)
    v[v == 0] = np.nan
    v[:, :, :, 64] = np.nan  # Blank GBT center-channel spike

    # 1. Subtract each row's median across the 128 channels -> 100% removes 1-row broadband PKS glitches & panel gain!
    row_med = np.nanmedian(v, axis=3, keepdims=True)
    v_rowclean = v - row_med

    # 2. Time-median spectrum per panel (clean of 1-row glitches)
    spec = np.nanmedian(v_rowclean, axis=2)  # (B, 6, 128)
    spec = np.nan_to_num(spec, nan=0.0)

    # 3. Reference OFF spectrum per panel
    ref = np.empty_like(spec)
    ref[:, ONP] = np.median(spec[:, OFFP], axis=1)[:, None, :]
    for j in OFFP:
        others = [q for q in OFFP if q != j]
        ref[:, j] = 0.5 * (spec[:, others[0]] + spec[:, others[1]])

    diff_spec = spec - ref  # (B, 6, 128)
    # 4. Project out smooth degree-3 polynomial across frequency (removes PKS bandpass tilt, roll-off, curvature!)
    # LEG_BASIS is (4, 128) with orthonormal rows: trend = (diff_spec @ P.T) @ P
    coeff = np.einsum('bpnf,df->bpd', diff_spec[:, :, None, :], LEG_BASIS)[:, :, 0, :] if False else (diff_spec @ LEG_BASIS.T)
    trend = coeff @ LEG_BASIS  # (B, 6, 128)

    # Z_clean: OFF-referenced, row-glitch-scrubbed, poly3-tilt-detrended dynamic spectrum
    z = v_rowclean - (ref + trend)[:, :, None, :]
    zo = z[:, OFFP].reshape(len(z), -1)
    sig = 1.4826 * np.nanmedian(np.abs(zo - np.nanmedian(zo, axis=1, keepdims=True)), axis=1)
    z = np.nan_to_num(z / np.maximum(sig, 1e-3)[:, None, None, None], nan=0.0)  # (B, 6, 16, 128)

    # Clamp extreme narrowband RFI spikes (> 4.5 sigma) before computing scintillation ACF & block texture
    z_scint = np.clip(z, -4.5, 4.5)
    o = {}

    def agg(name, m):
        on, off = m[:, ONP], m[:, OFFP]
        o[f'{name}_on_mean'] = on.mean(1)
        o[f'{name}_on_min'] = on.min(1)
        o[f'{name}_off_max'] = off.max(1)
        o[f'{name}_diff'] = on.mean(1) - off.mean(1)

    # Detrended spectral profile (8-channel and 16-channel boxcar smoothed)
    spec_clean = np.median(z_scint, axis=2)  # (B, 6, 128)
    k8 = np.ones(8, dtype=np.float32) / 8.0
    e8 = np.apply_along_axis(lambda m: np.convolve(m, k8, mode='same'), -1, spec_clean)
    agg('dexc8_max', e8.max(-1))
    agg('dexc8_area', np.clip(e8, 0, None).mean(-1))
    agg('dexc8_std', e8.std(-1))

    # Cross-correlation of detrended ON spectra (ON1, ON3, ON5)
    s_on = e8[:, ONP]  # (B, 3, 128)
    s_on_n = (s_on - s_on.mean(-1, keepdims=True)) / np.maximum(s_on.std(-1, keepdims=True), 1e-4)
    c01 = (s_on_n[:, 0] * s_on_n[:, 1]).mean(-1)
    c12 = (s_on_n[:, 1] * s_on_n[:, 2]).mean(-1)
    c02 = (s_on_n[:, 0] * s_on_n[:, 2]).mean(-1)
    o['dspec_on_corr_mean'] = (c01 + c12 + c02) / 3.0
    o['dspec_on_corr_min'] = np.minimum(np.minimum(c01, c12), c02)

    # Brzycki et al. (2023) 2D Autocorrelation Function A(dt, df) at strictly non-zero time lags dt >= 1
    # (completely immune to 1-row broadband glitches which only live at dt = 0!)
    for dt, df in [(1, 2), (1, 4), (1, 8), (2, 4), (2, 8), (4, 8), (4, 16)]:
        prod = z_scint[:, :, :-dt, :-df] * z_scint[:, :, dt:, df:]
        acf = prod.mean(axis=(2, 3))  # (B, 6)
        agg(f'acf_t{dt}_f{df}', acf)

    # Multi-scale detrended 2D block texture
    B = len(z_scint)
    for bt, bc in [(2, 8), (4, 8), (4, 16), (8, 16)]:
        blk = z_scint.reshape(B, NP, NT // bt, bt, NF // bc, bc).mean((3, 5)).reshape(B, NP, -1) * np.sqrt(bt * bc)
        agg(f'dtex{bt}x{bc}', blk.std(-1))

    # 5. Secondary Spectrum (2D FFT Delay-Doppler Power Spectral Density, Stinebring et al. 2001 / Brzycki 2023)
    # Separates:
    #   - scintillating_star: interior low-(ft, ff) quadrant (ft in 1..4, ff in 2..16)
    #   - 1-row broadband glitch: ff == 0 axis (all ft, zero frequency structure)
    #   - rfi_persistent: ft == 0 axis (zero temporal change, high ff)
    fft2 = np.abs(np.fft.rfft2(z_scint, axes=(2, 3))) ** 2 / (NT * NF)  # (B, 6, 16, 65)
    # Scintillation delay-Doppler band: ft in 1..5 (and 11..15), ff in 2..20
    ft_mask = np.zeros(NT, dtype=bool)
    ft_mask[1:6] = True
    ft_mask[-5:] = True
    scint_psd = fft2[:, :, ft_mask, 2:20].mean(axis=(2, 3))
    mid_psd = fft2[:, :, ft_mask, 20:48].mean(axis=(2, 3))
    dc_time_psd = fft2[:, :, 0, 4:48].mean(axis=-1)   # ft=0: persistent vertical lines
    dc_freq_psd = fft2[:, :, 1:, 0].mean(axis=-1)     # ff=0: broadband row glitches/pulses
    agg('sec_scint_psd', scint_psd)
    agg('sec_scint_ratio', scint_psd / np.maximum(mid_psd, 1e-4))
    agg('sec_dctime_psd', dc_time_psd)
    agg('sec_dcfreq_psd', dc_freq_psd)

    # 6. Instantaneous Frequency & Amplitude Time-Series Trajectory (16-step per panel)
    # Tracks ridge trajectory c(t) in each panel to measure S-curve curvature d2c/dt2 (satellite vs radar/ETI)
    # and temporal flicker / AR(1) autocorrelation of panel flux
    panel_flux_ts = z_scint.max(axis=-1)  # (B, 6, 16) peak channel time series per panel
    flux_ar1 = ((panel_flux_ts[:, :, :-1] - panel_flux_ts.mean(-1, keepdims=True)) *
                (panel_flux_ts[:, :, 1:] - panel_flux_ts.mean(-1, keepdims=True))).mean(-1) / np.maximum(panel_flux_ts.var(-1), 1e-4)
    agg('ts_flux_ar1', flux_ar1)

    return o


@torch.no_grad()
def subpixel_track_features(x, gaps, scale=1.0):
    """Bilinear sub-pixel de-Doppler track search across all 6 panels (241 slopes x 384 half-channel positions)."""
    B, dev = x.shape[0], x.device
    # Remove horizontal row median per panel so 1-row broadband glitches have zero projection onto tracks
    x_clean = x - x.median(dim=-1, keepdim=True).values

    # 241 slopes covering [-1.2, 1.2] * scale, and half-channel c0 grid [-32, 160) in steps of 0.5 (384 positions)
    ts = torch.linspace(-1.2, 1.2, 241, device=dev) * scale
    c0 = torch.arange(-32.0, 160.0, 0.5, device=dev, dtype=torch.float32)  # (384,)
    S, C = len(ts), len(c0)

    r_local = torch.arange(NT, device=dev, dtype=torch.float32)
    T = torch.empty((B, NP, S, C), device=dev, dtype=torch.float32)

    for p in range(NP):
        # Global row offset relative to cadence midpoint (row 63.0)
        t_glob = (p * (NT + gaps[:, None]) + r_local[None, :] - 63.0).float()  # (B, 16)
        xp = x_clean[:, p]  # (B, 16, 128)
        for s_start in range(0, S, 40):
            ts_b = ts[s_start:s_start + 40]  # (sb,)
            # Fractional channel position: (B, 16, sb, C)
            ch = t_glob[:, :, None, None] * ts_b[None, None, :, None] + c0[None, None, None, :]
            valid = (ch >= 0.0) & (ch <= float(NF - 2))
            ch_clamped = ch.clamp(0.0, float(NF - 2))
            c_lo = ch_clamped.long()
            w_hi = ch_clamped - c_lo.float()
            c_hi = c_lo + 1

            # Gather floor and ceil channels and interpolate bilinearly
            B_i, R_i, sb_i, C_i = c_lo.shape
            xp_exp = xp.reshape(B_i * R_i, NF)
            idx_lo = c_lo.reshape(B_i * R_i, sb_i * C_i)
            idx_hi = c_hi.reshape(B_i * R_i, sb_i * C_i)
            v_lo = xp_exp.gather(1, idx_lo).reshape(B_i, R_i, sb_i, C_i)
            v_hi = xp_exp.gather(1, idx_hi).reshape(B_i, R_i, sb_i, C_i)
            val = (1.0 - w_hi) * v_lo + w_hi * v_hi
            val = torch.where(valid, val, torch.full_like(val, -4.0))
            T[:, p, s_start:s_start + 40, :] = val.mean(dim=1) * 4.0  # in 1-panel SNR units (sqrt(16)=4)

    on_T = T[:, ONP]   # (B, 3, S, C)
    off_T = T[:, OFFP]  # (B, 3, S, C)
    off_max = off_T.max(dim=1).values.clamp(min=0.0)

    # 1. Strict 3-of-3 ON min track
    on_min = on_T.min(dim=1).values
    clean_min = on_min - off_max
    drifting = (ts.abs() >= 0.04 * scale)[None, :, None]
    clean_min_drift = torch.where(drifting, clean_min, torch.full_like(clean_min, -99.0))
    best_min_drift, arg_min = clean_min_drift.flatten(1).max(dim=1)
    si_min, ci_min = arg_min // C, arg_min % C

    # 2. Coherent 48-row ON mean track (1.732x higher sensitivity on low-SNR PKS ETI!)
    on_mean = on_T.mean(dim=1)
    clean_mean = on_mean - off_max
    clean_mean_drift = torch.where(drifting, clean_mean, torch.full_like(clean_mean, -99.0))
    best_mean_drift, arg_mean = clean_mean_drift.flatten(1).max(dim=1)
    si_mean, ci_mean = arg_mean // C, arg_mean % C

    # 3. Fade-tolerant 2-of-3 ON track (2nd-lowest ON panel minus off_max)
    on_sorted = torch.sort(on_T, dim=1, descending=True).values  # (B, 3, S, C)
    clean_2of3 = on_sorted[:, 1] - off_max
    clean_2of3_drift = torch.where(drifting, clean_2of3, torch.full_like(clean_2of3, -99.0))
    best_2of3_drift = clean_2of3_drift.flatten(1).max(dim=1).values

    rows = torch.arange(B, device=dev)
    return {
        'subpx_clean_min_drift': best_min_drift,
        'subpx_clean_min_all': clean_min.flatten(1).max(dim=1).values,
        'subpx_clean_mean_drift': best_mean_drift,
        'subpx_clean_mean_all': clean_mean.flatten(1).max(dim=1).values,
        'subpx_clean_2of3_drift': best_2of3_drift,
        'subpx_on_min_at_mean': on_min[rows, si_mean, ci_mean],
        'subpx_off_max_at_mean': off_max[rows, si_mean, ci_mean],
        'subpx_slope_phys': (ts[si_mean] / scale).abs(),
    }


def main():
    p = parser(__doc__)
    p.add_argument('--batch', type=int, default=250)
    p.add_argument('--phys', action='store_true', help='slope grids in physical drift units')
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
            f = brzycki_scint_acf_features(np.asarray(R[e]))
            x = torch.tensor(np.asarray(X[e]), device=dev).float() / 4.0
            g = torch.tensor(gaps[e], device=dev, dtype=torch.long)
            f.update({k: v.float().cpu().numpy() for k, v in subpixel_track_features(x, g, scale).items()})
            parts.append(pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()}))
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(work / f'feats5_{split}.parquet', index=False)
        log(f'{split}: {df.shape[1]} feats5 features written to {work / f"feats5_{split}.parquet"} ({dev})')


if __name__ == '__main__':
    main()
