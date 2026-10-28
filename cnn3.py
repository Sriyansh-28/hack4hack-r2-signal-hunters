"""Step 2b - Next-generation 2D CNN with SpecAugment and Squeeze-and-Excitation (SE) Residual blocks.

Key architectural improvements:
1. ResNet-SE Backbone: adds residual skip-connections and channel attention (SE) to capture faint drift lines
   and cross-panel correlations without vanishing gradients.
2. SpecAugment (Time & Frequency Masking): blocks out contiguous blocks of channels/time rows during training.
   Forces the CNN to learn robust multi-panel coherence rather than memorizing single bright bursts.
3. Multi-sample Dropout head: stabilizes training on minority classes (eti_candidate, scintillating_star).
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

H = NP * NT + 5 * 6  # 126 rows

def to_image(x, gaps):
    B, dev = x.shape[0], x.device
    p = torch.arange(NP, device=dev).repeat_interleave(NT)
    r = torch.arange(NT, device=dev).repeat(NP)
    idx = (p[None] * (NT + gaps[:, None]) + r[None])[:, :, None].expand(-1, -1, NF)
    img = torch.zeros(B, 2, H, NF, device=dev)
    img[:, 0].scatter_(1, idx, x.reshape(B, NP * NT, NF))
    onoff = torch.tensor([1.0, -1.0] * 3, device=dev).repeat_interleave(NT)[None, :, None].expand(B, -1, NF)
    img[:, 1].scatter_(1, idx, onoff.contiguous())
    return img

def spec_augment(x, max_mask_t=8, max_mask_f=16):
    """SpecAugment on (B, 6, 16, 128) cadences: random time & freq channel masking."""
    B, dev = x.shape[0], x.device
    # Mask random channels across panels
    if torch.rand(1).item() < 0.5:
        f_len = torch.randint(2, max_mask_f, (1,)).item()
        f_start = torch.randint(0, NF - f_len, (1,)).item()
        x[:, :, :, f_start:f_start + f_len] = 0.0
    # Mask random rows in random panels
    if torch.rand(1).item() < 0.5:
        t_len = torch.randint(1, max_mask_t, (1,)).item()
        t_start = torch.randint(0, NT - t_len, (1,)).item()
        p = torch.randint(0, NP, (1,)).item()
        x[:, p, t_start:t_start + t_len, :] = 0.0
    return x

def augment_v3(x, snr_max=0.0):
    B, dev = x.shape[0], x.device
    sel = lambda pr: (torch.rand(B, device=dev) < pr)
    
    # Frequency flip (drift sign inversion)
    f = sel(0.5)
    x = torch.where(f[:, None, None, None], x.flip(-1), x)
    
    # Frequency compression for PKS drift mismatch
    m = sel(0.5)
    if m.any():
        fac = float(np.random.uniform(0.55, 1.0))
        w = max(16, int(round(NF * fac)))
        xs = F.interpolate(x[m].reshape(-1, 1, NF), size=w, mode='linear', align_corners=False).reshape(-1, NP, NT, w)
        off = np.random.randint(0, NF - w + 1)
        pad = torch.randn(xs.shape[0], NP, NT, NF, device=dev)
        pad[..., off:off + w] = xs
        x[m] = pad
        
    x = x * (1 + 0.25 * (torch.rand(B, NP, 1, 1, device=dev) - 0.5))
    if snr_max > 0:
        n = sel(0.7)[:, None, None, None] * torch.rand(B, 1, 1, 1, device=dev) * snr_max
        x = (x + n * torch.randn_like(x)) / torch.sqrt(1 + n ** 2)
    else:
        n = sel(0.5)[:, None, None, None] * torch.rand(B, 1, 1, 1, device=dev) * 0.4
        x = x + n * torch.randn_like(x)
        
    drop = torch.rand(B, NP, NT, 1, device=dev) < 0.02
    x = x.masked_fill(drop, 0.0)
    if np.random.rand() < 0.5:
        x = torch.roll(x, int(np.random.randint(-20, 21)), dims=-1)
    
    # SpecAugment
    x = spec_augment(x)
    return x

class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, max(4, c // r)),
            nn.SiLU(),
            nn.Linear(max(4, c // r), c),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)[:, :, None, None]
        return x * w

class ResBlock(nn.Module):
    def __init__(self, ci, co, s=1):
        super().__init__()
        self.conv1 = nn.Conv2d(ci, co, 3, s, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(co)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(co, co, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(co)
        self.se = SEBlock(co)
        self.skip = nn.Identity() if (ci == co and s == 1) else nn.Sequential(
            nn.Conv2d(ci, co, 1, s, bias=False),
            nn.BatchNorm2d(co)
        )
        self.act2 = nn.SiLU()

    def forward(self, x):
        res = self.skip(x)
        h = self.act1(self.bn1(self.conv1(x)))
        h = self.se(self.bn2(self.conv2(h)))
        return self.act2(h + res)

class NetV3(nn.Module):
    def __init__(self, w=48, panel=True, n_tab=0):
        super().__init__()
        self.f = nn.Sequential(
            ResBlock(2, w, 1),
            ResBlock(w, 2 * w, 2),
            ResBlock(2 * w, 4 * w, 2),
            ResBlock(4 * w, 6 * w, 2),
            ResBlock(6 * w, 8 * w, 2)
        )
        d = 16 * w
        self.panel = panel
        if panel:
            self.g = nn.Sequential(
                ResBlock(6, w, 1),
                ResBlock(w, 2 * w, (1, 2)),
                ResBlock(2 * w, 4 * w, 2),
                ResBlock(4 * w, 4 * w, 2)
            )
            d += 8 * w
        self.n_tab = n_tab
        if n_tab:
            self.t = nn.Sequential(
                nn.Linear(n_tab, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Dropout(0.25),
                nn.Linear(256, 128),
                nn.SiLU()
            )
            d += 128
        self.head = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(d, 256),
            nn.SiLU(),
            nn.Linear(256, len(CLASSES))
        )

    def forward(self, img, x=None, tab=None):
        h = self.f(img)
        parts = [h.mean((2, 3)), h.amax((2, 3))]
        if self.panel:
            q = self.g(x)
            parts += [q.mean((2, 3)), q.amax((2, 3))]
        if self.n_tab:
            parts.append(self.t(tab))
        return self.head(torch.cat(parts, 1))

def run_v3(model, x, g, tab):
    return model(to_image(x, g), x, tab)

@torch.no_grad()
def predict_v3(model, X, G, TAB, idx, ctx, bs=512):
    model.eval()
    out = []
    for b in range(0, len(idx), bs):
        i = idx[b:b + bs]
        x = X[i].float() / 4.0
        t = TAB[i] if TAB is not None else None
        with ctx():
            p = torch.softmax(run_v3(model, x, G[i], t).float(), 1) + torch.softmax(run_v3(model, x.flip(-1), G[i], t).float(), 1)
        out.append(p / 2.0)
    return torch.cat(out).cpu().numpy()

def main():
    p = parser(__doc__)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--width', type=int, default=48)
    p.add_argument('--seeds', default='0')
    p.add_argument('--tag', default='_v3')
    p.add_argument('--panel', action='store_true', default=True)
    p.add_argument('--pseudo', default='', help='probs npz file for pseudo-labeling')
    p.add_argument('--pseudo_thr', type=float, default=0.92)
    p.add_argument('--tab', action='store_true', default=True)
    p.add_argument('--extra', action='store_true', default=True)
    p.add_argument('--feats4', action='store_true', default=True)
    p.add_argument('--snr_max', type=float, default=0.0)
    a = p.parse_args()
    work = Path(a.work)
    torch.set_num_threads(N_THREADS)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = dev.type == 'cuda'
    ctx = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if cuda else (lambda: torch.autocast('cpu', enabled=False))

    y = labels(work)
    folds = np.load(work / 'folds.npy')
    Xtr = torch.tensor(np.load(work / 'norm_train.npy'), device=dev)
    Xte = torch.tensor(np.load(work / 'norm_test.npy'), device=dev)
    X = torch.cat([Xtr, Xte])
    gaps = np.r_[pd.read_csv(work / 'train_meta.csv').slew_gap_rows.values, pd.read_csv(work / 'test_meta.csv').slew_gap_rows.values]
    G = torch.tensor(gaps, device=dev, dtype=torch.long)
    yall = np.r_[y, np.full(len(Xte), -1)]

    p_idx = np.array([], int)
    if a.pseudo:
        tp = np.load(a.pseudo)['test']
        p_idx = np.where(tp.max(1) >= a.pseudo_thr)[0]
        yall[len(y) + p_idx] = tp[p_idx].argmax(1)
        log(f'pseudo-labels: {len(p_idx)} cadences added (prob >= {a.pseudo_thr})')
    Y = torch.tensor(yall, device=dev)
    n, nt = len(y), len(Xte)

    TAB = None
    if a.tab:
        from sklearn.preprocessing import QuantileTransformer
        ftr, fte = load_feats(work, a.extra, a.feats4)
        qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=0)
        tab = np.nan_to_num(np.vstack([qt.fit_transform(ftr.values), qt.transform(fte.values)]))
        TAB = torch.tensor(tab, dtype=torch.float32, device=dev)

    cw = torch.tensor(len(y) / (len(CLASSES) * np.bincount(y, minlength=len(CLASSES))), dtype=torch.float32, device=dev)
    log(f'device {dev} (H100); {n} train + {nt} test; width {a.width}; features {TAB.shape[1] if TAB is not None else 0}')
    seeds = [int(s) for s in a.seeds.split(',')]
    oof, test = np.zeros((n, len(CLASSES))), np.zeros((nt, len(CLASSES)))
    test_idx = torch.arange(n, n + nt, device=dev)

    for seed in seeds:
        for k in range(folds.max() + 1):
            torch.manual_seed(seed * 100 + k)
            np.random.seed(seed * 100 + k)
            tri = torch.tensor(np.r_[np.where(folds != k)[0], len(y) + p_idx], device=dev)
            vai = torch.tensor(np.where(folds == k)[0], device=dev)
            model = NetV3(a.width, a.panel, TAB.shape[1] if TAB is not None else 0).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
            spe = math.ceil(len(tri) / a.bs)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * spe, eta_min=1e-5)

            for ep in range(a.epochs):
                model.train()
                perm = tri[torch.randperm(len(tri), device=dev)]
                for b in range(0, len(perm), a.bs):
                    i = perm[b:b + a.bs]
                    if len(i) < 2:
                        continue
                    xb = augment_v3(X[i].float() / 4.0, a.snr_max)
                    with ctx():
                        logits = run_v3(model, xb, G[i], TAB[i] if TAB is not None else None)
                    loss = F.cross_entropy(logits.float(), Y[i], weight=cw, label_smoothing=0.05)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    opt.step()
                    sched.step()

                if (ep + 1) % 10 == 0 or ep == a.epochs - 1:
                    pv = predict_v3(model, X, G, TAB, vai, ctx)
                    log(f'seed {seed} fold {k} epoch {ep + 1}: val macro-F1 {macro_f1(y[vai.cpu().numpy()], pv.argmax(1))[0]:.4f}')

            oof[vai.cpu().numpy()] += pv / len(seeds)
            test += predict_v3(model, X, G, TAB, test_idx, ctx) / (len(seeds) * (folds.max() + 1))

    save_preds(work, f'cnn{a.tag}', oof, test, report(y, oof, f'cnn{a.tag} FINAL OOF'))

if __name__ == '__main__':
    main()
