"""Step 2c - cnn4: multi-channel OFF-referenced ResNet-SE + temporal BiGRU sequence head (Phase 2).

Why: prep2.py removes each panel's OWN 15-channel time-median spectrum, which also erases broad ON-only
scintillation patches, so every earlier CNN (cnn2/cnn3) saw scintillating_star with its main evidence removed.
cnn4 adds, per panel, the feats5.py (Brzycki et al. 2023) view of the raw cadence:
  ch 0  norm      prep2.py image (kept as is: PKS pulsars are undispersed, a row-median scrub would erase them)
  ch 1  Z_offref  raw, row-median scrubbed (1-row broadband glitches -> 0), minus the OFF-panel reference spectrum and a
                  degree-3 Legendre tilt, in OFF-panel sigma units (keeps broad scintillation patches)
  ch 2  Z_diff    Z_offref of a panel minus its neighbour panels (ON - OFF), computed after augmentation
All channels share every augmentation (flip, PKS 0.55-1.0x drift compression, gain, noise, dropped rows, roll, SpecAugment).
Extra branch: a BiGRU across the 96 (panel, row) time steps of 1D-conv row embeddings + ON/OFF flag, so the model can
relate rows across the slew gaps directly. Z_offref is computed once on the GPU at start-up (no extra disk).
Output: WORK/preds/cnn<tag>.npz
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn3 import H, ResBlock
from common2 import CLASSES, N_THREADS, NF, NP, NT, labels, load_feats, log, macro_f1, parser, report, save_preds
from feats2 import OFFP, ONP
from feats5 import LEG_BASIS


@torch.no_grad()
def z_offref(raw, dev, bs=1000):
    """feats5.brzycki_scint_acf_features' Z_clean on the GPU: (N, 6, 16, 128) uint8 raw -> float16 Z, clipped to +-6."""
    L = torch.tensor(LEG_BASIS, device=dev)
    out = torch.empty(raw.shape, dtype=torch.float16, device=dev)
    for s in range(0, len(raw), bs):
        v = torch.tensor(np.asarray(raw[s:s + bs]), device=dev).float()
        v[v == 0] = float('nan')
        v[..., 64] = float('nan')
        v = v - v.nanmedian(dim=3, keepdim=True).values                    # 1-row broadband glitches and gain -> 0
        spec = torch.nan_to_num(v.nanmedian(dim=2).values, nan=0.0)        # (b, 6, 128)
        ref = torch.empty_like(spec)
        ref[:, ONP] = spec[:, OFFP].median(dim=1, keepdim=True).values
        for j in OFFP:
            o = [q for q in OFFP if q != j]
            ref[:, j] = 0.5 * (spec[:, o[0]] + spec[:, o[1]])
        trend = ((spec - ref) @ L.T) @ L                                    # degree-3 bandpass tilt / roll-off
        z = v - (ref + trend)[:, :, None, :]
        zo = z[:, OFFP].reshape(len(z), -1)
        mad = (zo - zo.nanmedian(dim=1, keepdim=True).values).abs().nanmedian(dim=1).values
        z = torch.nan_to_num(z / (1.4826 * mad).clamp(min=1e-3)[:, None, None, None], nan=0.0)
        out[s:s + bs] = z.clamp(-6.0, 6.0).half()
    return out


def z_diff(z):
    """Panel minus its neighbouring panels (ON - OFF for ON panels, OFF - ON for OFF panels)."""
    nb = torch.empty_like(z)
    nb[:, 0], nb[:, -1] = z[:, 1], z[:, -2]
    nb[:, 1:-1] = 0.5 * (z[:, :-2] + z[:, 2:])
    return z - nb


def spec_augment(x):
    """SpecAugment on (B, C, 6, 16, 128): the same channel block / panel rows masked in every channel."""
    if torch.rand(1).item() < 0.5:
        f_len = torch.randint(2, 16, (1,)).item()
        f0 = torch.randint(0, NF - f_len, (1,)).item()
        x[..., f0:f0 + f_len] = 0.0
    if torch.rand(1).item() < 0.5:
        t_len = torch.randint(1, 8, (1,)).item()
        t0 = torch.randint(0, NT - t_len, (1,)).item()
        x[:, :, torch.randint(0, NP, (1,)).item(), t0:t0 + t_len, :] = 0.0
    return x


def augment_v4(x):
    """cnn3.augment_v3 for (B, C, 6, 16, 128) with every random draw shared across the C channels."""
    B, C, dev = x.shape[0], x.shape[1], x.device
    sel = lambda pr: torch.rand(B, device=dev) < pr
    x = torch.where(sel(0.5)[:, None, None, None, None], x.flip(-1), x)
    m = sel(0.5)
    if m.any():                                                             # PKS drift is 0.64x GBT in pixels
        w = max(16, int(round(NF * float(np.random.uniform(0.55, 1.0)))))
        xs = F.interpolate(x[m].reshape(-1, 1, NF), size=w, mode='linear', align_corners=False).reshape(-1, C, NP, NT, w)
        off = np.random.randint(0, NF - w + 1)
        pad = torch.randn(xs.shape[0], C, NP, NT, NF, device=dev)
        pad[..., off:off + w] = xs
        x[m] = pad
    x = x * (1 + 0.25 * (torch.rand(B, 1, NP, 1, 1, device=dev) - 0.5))
    n = sel(0.5)[:, None, None, None, None] * torch.rand(B, 1, 1, 1, 1, device=dev) * 0.4
    x = x + n * torch.randn_like(x)
    x = x.masked_fill(torch.rand(B, 1, NP, NT, 1, device=dev) < 0.02, 0.0)
    if np.random.rand() < 0.5:
        x = torch.roll(x, int(np.random.randint(-20, 21)), dims=-1)
    return spec_augment(x)


def to_image4(xz, gaps):
    """(B, 3, 6, 16, 128) -> (B, 4, H, 128) global-time image: norm, Z_offref, Z_diff, ON/OFF flag."""
    B, C, dev = xz.shape[0], xz.shape[1], xz.device
    p = torch.arange(NP, device=dev).repeat_interleave(NT)
    r = torch.arange(NT, device=dev).repeat(NP)
    idx = (p[None] * (NT + gaps[:, None]) + r[None])[:, :, None].expand(-1, -1, NF)
    img = torch.zeros(B, C + 1, H, NF, device=dev)
    for c in range(C):
        img[:, c].scatter_(1, idx, xz[:, c].reshape(B, NP * NT, NF))
    onoff = torch.tensor([1.0, -1.0] * 3, device=dev).repeat_interleave(NT)[None, :, None].expand(B, -1, NF)
    img[:, C].scatter_(1, idx, onoff.contiguous())
    return img


class SeqHead(nn.Module):
    """1D conv over frequency per (panel, row) -> BiGRU over the 96 time steps (+ ON/OFF flag, physical row time)."""
    def __init__(self, ci=3, d=64):
        super().__init__()
        self.row = nn.Sequential(
            nn.Conv1d(ci, 32, 5, 1, 2), nn.BatchNorm1d(32), nn.SiLU(),
            nn.Conv1d(32, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.SiLU(),
            nn.Conv1d(64, d, 5, 2, 2), nn.BatchNorm1d(d), nn.SiLU())
        self.gru = nn.GRU(2 * d + 2, d, batch_first=True, bidirectional=True)

    def forward(self, xz, gaps):
        B, C = xz.shape[:2]
        h = self.row(xz.permute(0, 2, 3, 1, 4).reshape(B * NP * NT, C, NF))       # (B*96, d, 32)
        h = torch.cat([h.mean(-1), h.amax(-1)], 1).reshape(B, NP * NT, -1)
        p = torch.arange(NP, device=xz.device).repeat_interleave(NT)
        r = torch.arange(NT, device=xz.device).repeat(NP)
        t = ((p[None] * (NT + gaps[:, None]) + r[None]).float() / 126.0)[..., None]
        onoff = torch.tensor([1.0, -1.0] * 3, device=xz.device).repeat_interleave(NT)[None, :, None].expand(B, -1, 1)
        with torch.autocast(xz.device.type, enabled=False):                     # GRU in float32 (no bf16 RNN kernels)
            g, _ = self.gru(torch.cat([h.float(), onoff, t], -1))
        return torch.cat([g.mean(1), g.amax(1)], 1)                              # (B, 4d)


class NetV4(nn.Module):
    def __init__(self, w=48, n_tab=0):
        super().__init__()
        self.f = nn.Sequential(ResBlock(4, w, 1), ResBlock(w, 2 * w, 2), ResBlock(2 * w, 4 * w, 2),
                               ResBlock(4 * w, 6 * w, 2), ResBlock(6 * w, 8 * w, 2))
        self.g = nn.Sequential(ResBlock(3 * NP, w, 1), ResBlock(w, 2 * w, (1, 2)), ResBlock(2 * w, 4 * w, 2),
                               ResBlock(4 * w, 4 * w, 2))
        self.s = SeqHead(3, 64)
        d = 16 * w + 8 * w + 256
        self.n_tab = n_tab
        if n_tab:
            self.t = nn.Sequential(nn.Linear(n_tab, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.25),
                                   nn.Linear(256, 128), nn.SiLU())
            d += 128
        self.head = nn.Sequential(nn.Dropout(0.35), nn.Linear(d, 256), nn.SiLU(), nn.Linear(256, len(CLASSES)))

    def forward(self, xz, gaps, tab=None):
        h = self.f(to_image4(xz, gaps))
        q = self.g(xz.reshape(len(xz), 3 * NP, NT, NF))
        parts = [h.mean((2, 3)), h.amax((2, 3)), q.mean((2, 3)), q.amax((2, 3)), self.s(xz, gaps)]
        if self.n_tab:
            parts.append(self.t(tab))
        return self.head(torch.cat(parts, 1))


def batch_xz(X, Z, i, aug=False):
    xz = torch.stack([X[i].float() / 4.0, Z[i].float()], 1)                      # (b, 2, 6, 16, 128)
    if aug:
        xz = augment_v4(xz)
    return torch.cat([xz, z_diff(xz[:, 1])[:, None]], 1)                         # + Z_diff after augmentation


@torch.no_grad()
def predict_v4(model, X, Z, G, TAB, idx, ctx, bs=512):
    model.eval()
    out = []
    for b in range(0, len(idx), bs):
        i = idx[b:b + bs]
        xz = batch_xz(X, Z, i)
        t = TAB[i] if TAB is not None else None
        with ctx():
            p = torch.softmax(model(xz, G[i], t).float(), 1) + torch.softmax(model(xz.flip(-1), G[i], t).float(), 1)
        out.append(p / 2.0)
    return torch.cat(out).cpu().numpy()


def main():
    p = parser(__doc__)
    p.add_argument('--epochs', type=int, default=28)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--width', type=int, default=48)
    p.add_argument('--seeds', default='0')
    p.add_argument('--tag', default='_v4')
    p.add_argument('--pseudo', default='', help='probs npz file for pseudo-labeling')
    p.add_argument('--pseudo_thr', type=float, default=0.90)
    p.add_argument('--feats5', action='store_true', help='also feed the feats5.py features to the tabular branch')
    p.add_argument('--folds', default='', help='comma list of folds to train (debug); default all')
    p.add_argument('--smoke', action='store_true', help='CPU check: 3 batches, 256 val/test rows, nothing saved')
    a = p.parse_args()
    data, work = Path(a.data), Path(a.work)
    torch.set_num_threads(N_THREADS)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if dev.type == 'cuda' else (lambda: torch.autocast('cpu', enabled=False))

    y = labels(work)
    folds = np.load(work / 'folds.npy')
    X = torch.cat([torch.tensor(np.load(work / f'norm_{s}.npy'), device=dev) for s in ('train', 'test')])
    raws = []
    for s in ('train', 'test'):
        src = work / f'{s}_cadences.npy'
        raws.append(z_offref(np.load(src if src.exists() else data / f'{s}_cadences.npy', mmap_mode='r'), dev))
    Z = torch.cat(raws)
    del raws
    gaps = np.r_[pd.read_csv(work / 'train_meta.csv').slew_gap_rows.values, pd.read_csv(work / 'test_meta.csv').slew_gap_rows.values]
    G = torch.tensor(gaps, device=dev, dtype=torch.long)
    n, nt = len(y), len(X) - len(y)
    log(f'Z_offref ready: train |Z| mean {Z[:n].float().abs().mean():.3f}, test {Z[n:].float().abs().mean():.3f}')

    yall = np.r_[y, np.full(nt, -1)]
    p_idx = np.array([], int)
    if a.pseudo:
        tp = np.load(a.pseudo)['test']
        p_idx = np.where(tp.max(1) >= a.pseudo_thr)[0]
        yall[n + p_idx] = tp[p_idx].argmax(1)
        log(f'pseudo-labels: {len(p_idx)} cadences added (prob >= {a.pseudo_thr})')
    Y = torch.tensor(yall, device=dev)

    from sklearn.preprocessing import QuantileTransformer
    ftr, fte = load_feats(work, True, True, a.feats5)
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=0)
    TAB = torch.tensor(np.nan_to_num(np.vstack([qt.fit_transform(ftr.values), qt.transform(fte.values)])),
                       dtype=torch.float32, device=dev)

    cw = torch.tensor(n / (len(CLASSES) * np.bincount(y, minlength=len(CLASSES))), dtype=torch.float32, device=dev)
    log(f'device {dev}; {n} train + {nt} test; width {a.width}; features {TAB.shape[1]}')
    seeds = [int(s) for s in a.seeds.split(',')]
    ks = [int(k) for k in a.folds.split(',')] if a.folds else list(range(folds.max() + 1))
    oof, test = np.zeros((n, len(CLASSES))), np.zeros((nt, len(CLASSES)))
    test_idx = torch.arange(n, n + nt, device=dev)

    for seed in seeds:
        for k in ks:
            torch.manual_seed(seed * 100 + k)
            np.random.seed(seed * 100 + k)
            tri = torch.tensor(np.r_[np.where(folds != k)[0], n + p_idx], device=dev)
            vai = torch.tensor(np.where(folds == k)[0], device=dev)
            if a.smoke:
                tri, vai, test_idx = tri[torch.randperm(len(tri), device=dev)[:3 * a.bs]], vai[:256], test_idx[:256]
            model = NetV4(a.width, TAB.shape[1]).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * math.ceil(len(tri) / a.bs), eta_min=1e-5)
            for ep in range(a.epochs):
                model.train()
                perm = tri[torch.randperm(len(tri), device=dev)]
                for b in range(0, len(perm), a.bs):
                    i = perm[b:b + a.bs]
                    if len(i) < 2:
                        continue
                    xz = batch_xz(X, Z, i, aug=True)
                    with ctx():
                        logits = model(xz, G[i], TAB[i])
                    loss = F.cross_entropy(logits.float(), Y[i], weight=cw, label_smoothing=0.05)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    opt.step()
                    sched.step()
                if (ep + 1) % 7 == 0 or ep == a.epochs - 1:
                    pv = predict_v4(model, X, Z, G, TAB, vai, ctx)
                    log(f'seed {seed} fold {k} epoch {ep + 1}: val macro-F1 {macro_f1(y[vai.cpu().numpy()], pv.argmax(1))[0]:.4f}')
            if a.smoke:
                pt = predict_v4(model, X, Z, G, TAB, test_idx, ctx)
                log(f'SMOKE OK: val {pv.shape} test {pt.shape}, params {sum(q.numel() for q in model.parameters()) / 1e6:.2f} M')
                return
            oof[vai.cpu().numpy()] += pv / len(seeds)
            test += predict_v4(model, X, Z, G, TAB, test_idx, ctx) / (len(seeds) * len(ks))

    if a.folds:
        log('debug run (subset of folds): preds not saved')
        return
    save_preds(work, f'cnn{a.tag}', oof, test, report(y, oof, f'cnn{a.tag} FINAL OOF'))


if __name__ == '__main__':
    main()
