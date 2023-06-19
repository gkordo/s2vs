"""
Acknowledgement to Filip Radenovic(@filipradenovic) for providing the code in:
https://github.com/filipradenovic/cnnimageretrieval-pytorch
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def spoc(x, dim=1, **kwargs):
    if x.ndim != 4 or dim != 1:
        x = x.transpose(dim, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1, 1)
    return F.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)


def mac(x, dim=1, **kwargs):
    if x.ndim != 4 or dim != 1:
        x = x.transpose(dim, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1, 1)
    return F.adaptive_max_pool2d(x, (1, 1)).reshape(x.shape[0], -1)


def gem(x, dim=1, p=3., eps=1e-6, **kwargs):
    return spoc(x.clamp(min=eps).pow(p), dim).pow(1.0 / p)


def rpool(x, L=[1, 2, 3], aggr=mac, p=1e-6, eps=1e-6):
    ovr = 0.4  # desired overlap of neighboring regions
    steps = torch.Tensor(
        [2, 3, 4, 5, 6, 7]
    )  # possible regions for the long dimension

    W = x.shape[3]
    H = x.shape[2]

    w = min(W, H)
    w2 = math.floor(w / 2.0 - 1)

    b = (max(H, W) - w) / (steps - 1)
    (tmp, idx) = torch.min(
        torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0
    )  # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0
    Hd = 0
    if H < W:
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    vecs = []
    for l in L:
        if l == 1:
            vecs.append(aggr(x, dim=1, p=p, eps=eps).unsqueeze(1))
            continue
        wl = math.floor(2 * w / (l + 1))
        wl2 = math.floor(wl / 2 - 1)

        if l + Wd == 1:
            b = 0
        else:
            b = (W - wl) / (l + Wd - 1)
        cenW = (
                torch.floor(wl2 + torch.tensor(range(l - 1 + Wd + 1)) * b) - wl2
        ).long()  # center coordinates
        if l + Hd == 1:
            b = 0
        else:
            b = (H - wl) / (l + Hd - 1)
        cenH = (
                torch.floor(wl2 + torch.tensor(range(l - 1 + Hd + 1)) * b) - wl2
        ).long()  # center coordinates

        for i in cenH.tolist():
            for j in cenW.tolist():
                if wl == 0:
                    continue
                f = x[:, :, i: i + wl, j: j + wl]
                f = aggr(f, dim=1, p=p, eps=eps).unsqueeze(1)
                vecs.append(f)
    return torch.cat(vecs, dim=1)


class MAC(nn.Module):

    def __init__(self):
        super(MAC,self).__init__()

    def forward(self, x):
        return mac(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SPoC(nn.Module):

    def __init__(self):
        super(SPoC,self).__init__()

    def forward(self, x):
        return spoc(x)
        
    def __repr__(self):
        return self.__class__.__name__ + '()'


class GeM(nn.Module):

    def __init__(self, p=3., eps=1e-6):
        super(GeM, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p) + ', ' + 'eps=' + str(self.eps) + ')'


class RMAC(nn.Module):

    def __init__(self, L=[1, 2, 3], eps=1e-6):
        super(RMAC, self).__init__()
        self.L = L
        self.eps = eps

    def forward(self, x):
        return rpool(x, L=self.L, eps=self.eps, aggr=mac).mean(1)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'L=' + '{}'.format(self.L) + ')'


class LMAC(nn.Module):

    def __init__(self, L=[3], eps=1e-6):
        super(LMAC, self).__init__()
        self.L = L
        self.eps = eps

    def forward(self, x):
        return rpool(x, L=self.L, eps=self.eps, aggr=mac)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'L=' + '{}'.format(self.L) + ')'


class RGeM(nn.Module):

    def __init__(self, L=[3], p=3., eps=1e-6):
        super(RGeM, self).__init__()
        self.p = p
        self.L = L
        self.eps = eps

    def forward(self, x):
        return rpool(x, L=self.L, aggr=gem, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p) + ', ' + 'eps=' + str(
            self.eps) + ')'
