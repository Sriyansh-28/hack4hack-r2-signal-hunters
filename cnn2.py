"""Step 2 - 2D CNN on the global-time image of each cadence.

The 6 panels are placed at their real time rows (panel p, row r -> p*(16+gap)+r, gaps left empty), so an ETI
signal that drifts steadily through all three ON panels is ONE straight line in the image. A second channel
marks ON (+1) / OFF (-1) rows. Class-balanced loss (test label mix differs), 5 folds, flip TTA.
Augmentations imitate the PKS telescope: drift slopes ~0.64x (frequency compression), gain/noise changes,
dropped rows, frequency flips (flip = drift sign flip, allowed) and shifts.
Output: WORK/preds/cnn<tag>.npz
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common2 import CLASSES, N_THREADS, NF, NP, NT, labels, load_feats, log, macro_f1, parser, report, save_preds

H = NP * NT + 5 * 6                                    # 126 rows: 6 panels + up to 5 gaps of 6 rows


def to_image(x, gaps):
    """x (B, 6, 16, 128) float, gaps (B,) long -> (B, 2, 126, 128)."""
    B, dev = x.shape[0], x.device
    p = torch.arange(NP, device=dev).repeat_interleave(NT)
    r = torch.arange(NT, device=dev).repeat(NP)
    idx = (p[None] * (NT + gaps[:, None]) + r[None])[:, :, None].expand(-1, -1, NF)   # (B, 96, 128)
    img = torch.zeros(B, 2, H, NF, device=dev)
    img[:, 0].scatter_(1, idx, x.reshape(B, NP * NT, NF))
    onoff = torch.tensor([1.0, -1.0] * 3, device=dev).repeat_interleave(NT)[None, :, None].expand(B, -1, NF)
    img[:, 1].scatter_(1, idx, onoff.contiguous())
    return img


def augment(x, snr_max=0.0):
    B, dev = x.shape[0], x.device
    sel = lambda pr: (torch.rand(B, device=dev) < pr)
    f = sel(0.5)                                                    # frequency flip (drift sign flip)
    x = torch.where(f[:, None, None, None], x.flip(-1), x)
    m = sel(0.5)                                                    # frequency compression: PKS drifts look ~0.64x
    if m.any():
        fac = float(np.random.uniform(0.55, 1.0))
        w = max(16, int(round(NF * fac)))
        xs = F.interpolate(x[m].reshape(-1, 1, NF), size=w, mode='linear', align_corners=False).reshape(-1, NP, NT, w)
        off = np.random.randint(0, NF - w + 1)
        pad = torch.randn(xs.shape[0], NP, NT, NF, device=dev)
        pad[..., off:off + w] = xs
        x[m] = pad
    x = x * (1 + 0.25 * (torch.rand(B, NP, 1, 1, device=dev) - 0.5)) # per-panel gain jitter
    if snr_max > 0:                                                 # PKS is noisier: add noise and renormalise,
        n = sel(0.7)[:, None, None, None] * torch.rand(B, 1, 1, 1, device=dev) * snr_max   # so signals get fainter
        x = (x + n * torch.randn_like(x)) / torch.sqrt(1 + n ** 2)
    else:
        n = sel(0.5)[:, None, None, None] * torch.rand(B, 1, 1, 1, device=dev) * 0.4
        x = x + n * torch.randn_like(x)                             # extra noise (PKS is noisier)
    drop = torch.rand(B, NP, NT, 1, device=dev) < 0.02              # dropped rows (PKS has more)
    x = x.masked_fill(drop, 0.0)
    if np.random.rand() < 0.5:                                       # frequency shift
        x = torch.roll(x, int(np.random.randint(-20, 21)), dims=-1)
    return x


def block(ci, co, s):
    return nn.Sequential(nn.Conv2d(ci, co, 3, s, 1, bias=False), nn.BatchNorm2d(co), nn.SiLU(),
                         nn.Conv2d(co, co, 3, 1, 1, bias=False), nn.BatchNorm2d(co), nn.SiLU())


class Net(nn.Module):
    """Branch 1: global-time image (tracks through real time). Branch 2 (optional): the 6 panels stacked as
    channels (16 x 128), so ON and OFF are compared at the same time row and channel. Optional tabular features."""
    def __init__(self, w=32, panel=False, n_tab=0):
        super().__init__()
        self.f = nn.Sequential(block(2, w, 1), block(w, 2 * w, 2), block(2 * w, 4 * w, 2), block(4 * w, 8 * w, 2),
                               block(8 * w, 8 * w, 2))
        d = 16 * w
        self.panel = panel
        if panel:
            self.g = nn.Sequential(block(6, w, 1), block(w, 2 * w, (1, 2)), block(2 * w, 4 * w, 2), block(4 * w, 4 * w, 2))
            d += 8 * w
        self.n_tab = n_tab
        if n_tab:
            self.t = nn.Sequential(nn.Linear(n_tab, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.SiLU())
            d += 128
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, 256), nn.SiLU(), nn.Linear(256, len(CLASSES)))

    def forward(self, img, x=None, tab=None):
        h = self.f(img)
        parts = [h.mean((2, 3)), h.amax((2, 3))]
        if self.panel:
            q = self.g(x)
            parts += [q.mean((2, 3)), q.amax((2, 3))]
        if self.n_tab:
            parts.append(self.t(tab))
        return self.head(torch.cat(parts, 1))


def run(model, x, g, tab):
    return model(to_image(x, g), x, tab)


@torch.no_grad()
def predict(model, X, G, TAB, idx, ctx, bs=512):
    model.eval()
    out = []
    for b in range(0, len(idx), bs):
        i = idx[b:b + bs]
        x = X[i].float() / 4
        t = TAB[i] if TAB is not None else None
        with ctx():
            p = torch.softmax(run(model, x, G[i], t).float(), 1) + torch.softmax(run(model, x.flip(-1), G[i], t).float(), 1)
        out.append(p / 2)
    return torch.cat(out).cpu().numpy()


def main():
    p = parser(__doc__)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--seeds', default='0')
    p.add_argument('--tag', default='')
    p.add_argument('--panel', action='store_true', help='add the stacked-panel branch')
    p.add_argument('--max_folds', type=int, default=0, help='smoke test: train only this many folds, save nothing')
    p.add_argument('--pseudo', default='', help='probs .npz (e.g. WORK/sub_r2_stack_probs.npz): add confident test cadences')
    p.add_argument('--pseudo_thr', type=float, default=0.9)
    p.add_argument('--tab', action='store_true', help='also feed the step-3 features (needs feats_*.parquet)')
    p.add_argument('--extra', action='store_true', help='with --tab: also the feats3.py features')
    p.add_argument('--snr_max', type=float, default=0.0,
                   help='>0: add up to this much noise (sigma units) and renormalise, so signals look as faint as on PKS')
    a = p.parse_args()
    work = Path(a.work)
    torch.set_num_threads(N_THREADS)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = dev.type == 'cuda'
    ctx = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if cuda else (lambda: torch.autocast('cpu', enabled=False))
    y = labels(work)
    folds = np.load(work / 'folds.npy')
    Xtr = torch.tensor(np.load(work / 'norm_train.npy'), device=dev)          # int8, 0.25-sigma steps
    Xte = torch.tensor(np.load(work / 'norm_test.npy'), device=dev)
    X = torch.cat([Xtr, Xte])
    gaps = np.r_[pd.read_csv(work / 'train_meta.csv').slew_gap_rows.values, pd.read_csv(work / 'test_meta.csv').slew_gap_rows.values]
    G = torch.tensor(gaps, device=dev, dtype=torch.long)
    yall = np.r_[y, np.full(len(Xte), -1)]
    p_idx = np.array([], int)
    if a.pseudo:                                     # test cadences the ensemble is sure about -> extra training data
        tp = np.load(a.pseudo)['test']
        p_idx = np.where(tp.max(1) >= a.pseudo_thr)[0]
        yall[len(y) + p_idx] = tp[p_idx].argmax(1)
        log(f'pseudo-labels: {len(p_idx)} test cadences (prob >= {a.pseudo_thr}), per class {np.bincount(tp[p_idx].argmax(1), minlength=len(CLASSES))}')
    Y = torch.tensor(yall, device=dev)
    n, nt = len(y), len(Xte)
    TAB = None
    if a.tab:
        from sklearn.preprocessing import QuantileTransformer
        ftr, fte = load_feats(work, a.extra)
        qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=0)
        tab = np.nan_to_num(np.vstack([qt.fit_transform(ftr.values), qt.transform(fte.values)]))
        TAB = torch.tensor(tab, dtype=torch.float32, device=dev)
    cw = torch.tensor(len(y) / (len(CLASSES) * np.bincount(y, minlength=len(CLASSES))), dtype=torch.float32, device=dev)
    log(f'device {dev}; {n} train + {nt} test; class weights {np.round(cw.cpu().numpy(), 2)}')
    seeds = [int(s) for s in a.seeds.split(',')]
    oof, test = np.zeros((n, len(CLASSES))), np.zeros((nt, len(CLASSES)))
    test_idx = torch.arange(n, n + nt, device=dev)
    for seed in seeds:
        for k in range(folds.max() + 1):
            if a.max_folds and k >= a.max_folds:
                log('smoke test OK'); return
            torch.manual_seed(seed * 100 + k)
            np.random.seed(seed * 100 + k)
            tri = torch.tensor(np.r_[np.where(folds != k)[0], len(y) + p_idx], device=dev)
            vai = torch.tensor(np.where(folds == k)[0], device=dev)
            model = Net(a.width, a.panel, TAB.shape[1] if TAB is not None else 0).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
            spe = math.ceil(len(tri) / a.bs)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * spe, pct_start=0.15)
            for ep in range(a.epochs):
                model.train()
                perm = tri[torch.randperm(len(tri), device=dev)]
                for b in range(0, len(perm), a.bs):
                    i = perm[b:b + a.bs]
                    if len(i) < 2:
                        sched.step()
                        continue
                    xb = augment(X[i].float() / 4, a.snr_max)
                    with ctx():
                        logits = run(model, xb, G[i], TAB[i] if TAB is not None else None)
                    loss = F.cross_entropy(logits.float(), Y[i], weight=cw, label_smoothing=0.05)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    opt.step()
                    sched.step()
                if (ep + 1) % 5 == 0 or ep == a.epochs - 1:
                    pv = predict(model, X, G, TAB, vai, ctx)
                    log(f'seed {seed} fold {k} epoch {ep + 1}: val macro-F1 {macro_f1(y[vai.cpu().numpy()], pv.argmax(1))[0]:.4f}')
            oof[vai.cpu().numpy()] += pv / len(seeds)
            test += predict(model, X, G, TAB, test_idx, ctx) / (len(seeds) * (folds.max() + 1))
    save_preds(work, f'cnn{a.tag}', oof, test, report(y, oof, f'cnn{a.tag} FINAL OOF'))


if __name__ == '__main__':
    main()
