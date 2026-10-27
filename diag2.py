"""Diagnostics 2 (~1 min): why is disp_on_max (dispersed broadband sweep, ON panels) median 9.0 on PKS vs 3.0 on GBT
while OFF panels barely change? For noise cadences (train labels / confident test predictions): the median sweep SNR
per de-dispersion curve, the winning row, and the mean cleaned ON-minus-OFF image on a coarse grid (0.01 sigma)."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common2 import CLASSES, labels, parser
from feats2 import OFFP, ONP, dedisperse

CURVES = [0, 2, 4, 6, 9, 12]


def main():
    p = parser(__doc__)
    p.add_argument('--probs', default='sub_r2_stack6_pl_probs.npz')
    a = p.parse_args()
    work = Path(a.work)
    y, pt = labels(work), np.load(work / a.probs)['test']
    k = CLASSES.index('noise')
    sel = {'train': np.where(y == k)[0][:1500], 'test': np.where((pt.argmax(1) == k) & (pt.max(1) >= 0.9))[0][:1500]}
    for split, idx in sel.items():
        X = np.load(work / f'norm_{split}.npy', mmap_mode='r')
        x = torch.tensor(np.asarray(X[np.sort(idx)])).float() / 4          # (n,6,16,128) sigma units
        D = dedisperse(x, CURVES)                                          # (n,6,A,16)
        on, off = D[:, ONP], D[:, OFFP]
        print(f'== {split}: {len(idx)} noise cadences')
        for name, d in (('ON ', on), ('OFF', off)):
            print(f'  median sweep SNR per curve {CURVES} {name}:',
                  [round(float(d[:, :, i].amax(-1).amax(1).median()), 2) for i in range(len(CURVES))])
        print('  winning row (ON):', np.bincount(on.amax(2).amax(1).argmax(-1).numpy(), minlength=16).tolist())
        grid = (x[:, ONP].mean((0, 1)) - x[:, OFFP].mean((0, 1))).reshape(16, 8, 16).mean(-1)
        print('  mean ON - OFF (rows x 16-channel groups, 0.01 sigma):')
        print(pd.DataFrame((grid * 100).round().int().numpy()).to_string())
        for name, pp in (('ON ', ONP), ('OFF', OFFP)):
            print(f'  row profile {name}:', (x[:, pp].mean((0, 1, 3)) * 100).round().int().tolist())
            print(f'  channel-group profile {name}:', (x[:, pp].mean((0, 1, 2)).reshape(8, 16).mean(-1) * 100).round().int().tolist())


if __name__ == '__main__':
    main()
