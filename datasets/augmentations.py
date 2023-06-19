import torch
import numpy as np
import torch.nn as nn

from torchvision import transforms as tv_trans
from pytorchvideo import transforms as pv_trans
from .transforms.frame_trans import FrameTransforms
from .transforms.temporal_trans import TemporalTransformations
from .transforms.video_in_video import VideoInVideo


class Augmentations(nn.Module):

    def __init__(self, frame_sz=224, window_sz=32):
        super().__init__()
        self.frame_sz = frame_sz
        self.window_sz = window_sz

    def forward(self, video):
        pass

    def random_temporal_crop(self, tensor, min_size=None):
        if min_size is None:
            min_size = self.window_sz
        while tensor.shape[0] < min_size:
            tensor = np.concatenate([tensor, tensor], 0)
        offset = np.random.randint(max(len(tensor) - min_size, 1))
        return tensor[offset:offset + min_size]


class WeakAugmentations(Augmentations):

    def __init__(self, frame_sz=224, window_sz=32, **kargs):
        super().__init__(frame_sz, window_sz)
        self.transforms = pv_trans.ApplyTransformToKey(
            key="video",
            transform=tv_trans.Compose(
                [
                    pv_trans.Permute((0, 3, 1, 2)),
                    tv_trans.RandomResizedCrop(self.frame_sz, scale=(0.8, 1.)),
                    tv_trans.RandomHorizontalFlip(),
                    pv_trans.Permute((0, 2, 3, 1)),
                ]
            ),
        )

    def forward(self, video):
        video = self.random_temporal_crop(video)
        assert video.shape[0] == self.window_sz
        return self.transforms({'video': torch.from_numpy(video)})['video']

    def __repr__(self):
        return '{}(frame_sz={}, window_sz={})'.format(self.__class__.__name__, self.frame_sz, self.window_sz)


class StrongAugmentations(Augmentations):

    def __init__(
            self,
            frame_sz=224,
            window_sz=32,
            augmentations='GT,FT,TT,ViV',
            n_raug=2,
            m_raug=9,
            p_overlay=0.3,
            p_blur=0.5,
            p_tsd=0.5,
            p_ff=0.1,
            p_sm=0.1,
            p_rev=0.1,
            p_pau=0.1,
            p_shuffle=0.5,
            p_dropout=0.3,
            p_content=0.5,
            p_viv=0.3,
            lambda_viv=(0.3, 0.7),
            **kargs
    ):
        super().__init__(frame_sz, window_sz)
        self.global_transformations = 'GT' in augmentations
        self.frame_transformations = 'FT' in augmentations
        self.temporal_transformations = 'TT' in augmentations
        self.video_in_video = 'ViV' in augmentations

        self.transforms = [
            pv_trans.Permute((0, 3, 1, 2)),
            tv_trans.RandomResizedCrop(self.frame_sz, scale=(0.5, 1.)),
            tv_trans.RandomHorizontalFlip()
        ]
        self.global_trn = None
        if self.global_transformations:
            self.global_trn = tv_trans.RandAugment(num_ops=n_raug,
                                                   magnitude=m_raug)
            self.transforms.append(self.global_trn)

        self.frame_trn = None
        if self.frame_transformations:
            self.frame_trn = FrameTransforms(p_overlay, p_blur)
            self.transforms.append(self.frame_trn)
        self.transforms.append(pv_trans.Permute((0, 2, 3, 1)))

        self.temporal_trn = None
        if self.temporal_transformations:
            self.temporal_trn = TemporalTransformations(self.window_sz, p_tsd=p_tsd, p_ff=p_ff,
                                                        p_sm=p_sm, p_rev=p_rev, p_pau=p_pau, **kargs)

        self.viv_trn = None
        if self.video_in_video:
            self.viv_trn = VideoInVideo(p=p_viv, mix_lamda=lambda_viv)

        self.transforms = pv_trans.ApplyTransformToKey(
            key="video",
            transform=tv_trans.Compose(self.transforms),
        )

    def forward(self, video):
        if self.temporal_transformations:
            video = self.temporal_trn(video)
        video = self.random_temporal_crop(video)
        if not isinstance(video, torch.Tensor):
            video = torch.from_numpy(video.copy())
        assert video.shape[0] == self.window_sz
        return self.transforms({'video': video})['video']

    def mixup(self, videos, labels):
        if self.video_in_video:
            return self.viv_trn(videos, labels)
        return videos, labels

    def __repr__(self):
        return '{}(frame_sz={}, window_sz={}' \
               '\n\tglobal={},\n\tframe={},' \
               '\n\ttemporal={},\n\tviv={}\n)'.format(self.__class__.__name__, self.frame_sz, self.window_sz,
                                                      self.global_trn, self.frame_trn, self.temporal_trn, self.viv_trn)
