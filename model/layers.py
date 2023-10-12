import enum
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from model.constraints import L2Constrain


class VideoNormalizer(nn.Module):

    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        super(VideoNormalizer, self).__init__()
        self.scale = nn.Parameter(torch.Tensor([255.]), requires_grad=False)
        self.mean = nn.Parameter(torch.Tensor(mean), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std), requires_grad=False)

    def forward(self, video):
        if video.dtype == torch.uint8:
            video = video.float() / self.scale
        if video.shape[-1] != 3 and video.shape[1] == 3:
            video = video.permute(0, 2, 3, 1)
        video = (video - self.mean) / self.std
        return video.permute(0, 3, 1, 2)


class PCALayer(nn.Module):
    def __init__(self, file_name=None, n_components=None, eps=1e-7, pretrained=True, trainable=False):
        super(PCALayer, self).__init__()
        self.dims = n_components
        self.eps = eps
        self.trainable = trainable
        self.mean = None
        self.d = None
        self.V = None
        self.DVt = None
        if file_name is not None:
            self.load(file_name)
        elif pretrained:
            pretrained_url = 'http://ndd.iti.gr/visil/pca_resnet50_vcdb_1M.pth'
            white = torch.hub.load_state_dict_from_url(pretrained_url)
            self.init_params(white["mean"].numpy(), white["d"].numpy(), white["V"].numpy())

    def save(self, file_name):
        np.savez_compressed(file_name, mean=self.mean.cpu().numpy(), d=self.d, V=self.V)

    def load(self, file_name):
        white = np.load(file_name)
        self.init_params(white["mean"], white["d"], white["V"])

    def train_pca(self, trainset):
        mean = trainset.mean(axis=0)
        trainset = trainset - mean
        Xcov = np.dot(trainset.T, trainset)
        d, V = np.linalg.eigh(Xcov)
        self.init_params(mean, d, V)

    def init_params(self, mean, d, V):
        self.d = d
        self.V = V

        idx = np.argsort(d)[::-1][: self.dims]
        d = d[idx]
        V = V[:, idx]
        D = np.diag(1.0 / np.sqrt(d + self.eps))

        self.mean = nn.Parameter(
            torch.from_numpy(mean.astype(np.float32)), requires_grad=self.trainable
        )
        self.DVt = nn.Parameter(
            torch.from_numpy(np.dot(D, V.T).T.astype(np.float32)), requires_grad=self.trainable
        )

    def forward(self, x):
        x -= self.mean.expand_as(x)
        x = torch.matmul(x, self.DVt)
        x = F.normalize(x, p=2, dim=-1)
        return x


class Attention(nn.Module):

    def __init__(self, dims, norm=False, activation=torch.tanh):
        super(Attention, self).__init__()
        self.norm = norm
        if self.norm:
            self.constrain = L2Constrain()
        else:
            self.transform = nn.Linear(dims, dims)
        self.context_vector = nn.Linear(dims, 1, bias=False)
        self.activation = activation
        self.reset_parameters()

    def forward(self, x):
        if self.norm:
            weights = self.context_vector(x) / torch.norm(self.context_vector.weight, p=2)
            weights = torch.add(torch.div(weights, 2.), .5)
        else:
            x_tr = self.activation(self.transform(x))
            weights = self.context_vector(x_tr)
            weights = torch.sigmoid(weights)
        x = x * weights
        return x, weights

    def reset_parameters(self):
        if self.norm:
            nn.init.normal_(self.context_vector.weight)
            self.constrain(self.context_vector)
        else:
            nn.init.xavier_uniform_(self.context_vector.weight)
            nn.init.xavier_uniform_(self.transform.weight)
            nn.init.zeros_(self.transform.bias)

    def apply_constraint(self):
        if self.norm:
            self.constrain(self.context_vector)


class BinarizationLayer(nn.Module):

    def __init__(self, file_name=None, dims=None, bits=None, sigma=1e-6, pretrained=True, trainable=True):
        super(BinarizationLayer, self).__init__()
        self.bits = bits
        self.dims = dims
        self.sigma = sigma
        self.trainable = trainable
        self.W = None
        if file_name is not None:
            self.load(file_name)
        elif pretrained:
            pretrained_url = 'https://mever.iti.gr/distill-and-select/models/itq_resnet50W_dns100k_1M.pth'
            weights = torch.hub.load_state_dict_from_url(pretrained_url)
            self.init_params(weights['proj'])
        elif dims is not None:
            self.bits = bits if bits is None else dims
            self.init_params()

    def save(self, file_name):
        np.savez_compressed(file_name, proj=self.W.detach().cpu().numpy())

    def load(self, file_name):
        white = np.load(file_name)
        proj = torch.from_numpy(white['proj']).float()
        self.init_params(proj)

    def init_params(self, proj=None):
        if proj is None:
            print('test')
            proj = torch.randn(self.dims, self.bits)
        self.W = nn.Parameter(proj, requires_grad=self.trainable)
        self.dims, self.bits = self.W.shape

    @staticmethod
    def _itq_rotation(v, n_iter, bit):
        r = np.random.randn(bit, bit)
        u11, s2, v2 = np.linalg.svd(r)

        r = u11[:, :bit]

        for _ in range(n_iter):
            z = np.dot(v, r)
            ux = np.ones(z.shape) * (-1.)
            ux[z >= 0] = 1
            c = np.dot(ux.transpose(), v)
            ub, sigma, ua = np.linalg.svd(c)
            r = np.dot(ua, ub.transpose())
        z = np.dot(v, r)
        b = np.ones(z.shape) * -1.
        b[z >= 0] = 1
        return b, r

    def train_itq(self, trainset):
        c = np.cov(trainset.transpose())

        l, pc = np.linalg.eig(c)

        l_pc_ordered = sorted(zip(l, pc.transpose()), key=lambda _p: _p[0], reverse=True)
        pc_top = np.array([p[1] for p in l_pc_ordered[:self.bits]]).transpose()

        v = np.dot(trainset, pc_top)

        b, rotation = self._itq_rotation(v, 50, self.bits)

        proj = np.dot(pc_top, rotation)
        self.init_params(proj)
        return proj

    def forward(self, x):
        x = F.normalize(x, p=2, dim=-1)
        x = torch.matmul(x, self.W)
        if self.training and self.trainable:
            x = torch.erf(x / np.sqrt(2 * self.sigma))
        else:
            x = torch.sign(x)
        return x

    def __repr__(self, ):
        return '{}(dims={}, bits={})'.format(self.__class__.__name__, self.W.shape[0], self.W.shape[1])


class NetVLAD(nn.Module):
    """Acknowledgement to @lyakaap and @Nanne for providing their implementations"""

    def __init__(self, dims, num_clusters, outdims=None):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dims = dims

        self.centroids = nn.Parameter(torch.randn(num_clusters, dims) / math.sqrt(self.dims))
        self.conv = nn.Conv2d(dims, num_clusters, kernel_size=1, bias=False)

        if outdims is not None:
            self.outdims = outdims
            self.reduction_layer = nn.Linear(self.num_clusters * self.dims, self.outdims, bias=False)
        else:
            self.outdims = self.num_clusters * self.dims
        self.norm = nn.LayerNorm(self.outdims)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.weight = nn.Parameter(self.centroids.detach().clone().unsqueeze(-1).unsqueeze(-1))
        if hasattr(self, 'reduction_layer'):
            nn.init.normal_(self.reduction_layer.weight, std=1 / math.sqrt(self.num_clusters * self.dims))

    def forward(self, x, mask=None):
        b, d, t, r = x.shape

        # soft-assignment
        soft_assign = self.conv(x)
        soft_assign = F.softmax(soft_assign, dim=1)

        vlad = torch.zeros([b, self.num_clusters, d], dtype=x.dtype, layout=x.layout, device=x.device)
        for cluster in range(self.num_clusters):
            residual = x - rearrange(self.centroids[cluster], 'd -> () d () ()')
            residual *= soft_assign[:, cluster].unsqueeze(1)
            if mask is not None:
                residual = residual.masked_fill((1 - rearrange(mask, 'b t -> b () t ()')).bool(), 0.0)
            vlad[:, cluster] = residual.sum([-2, -1])

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(b, -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        if hasattr(self, 'reduction_layer'):
            vlad = self.reduction_layer(vlad)
        return self.norm(vlad)
