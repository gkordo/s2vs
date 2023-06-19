import torch
import numpy as np
import torch.nn as nn

from torchvision import transforms
from .samplers import UniformSampler, TupleSampler, UniformIntSampler
from .transforms import MaybeApplyTransform, BlurTransform
from .overlay_text import OverlayTextTransform
from .overlay_emoji import OverlayEmojiTransform


class FrameTransforms(nn.Module):

    def __init__(self, p_overlay=0.3, p_blur=0.5):
        super().__init__()
        self.p_overlay = p_overlay
        self.p_blur = p_blur

        overlay_text = OverlayTextTransform(
            font_size_sampler=UniformSampler(0.2, .5),
            opacity_sampler=UniformSampler(0.7, 1.),
            color_sampler=TupleSampler((
                UniformIntSampler(0, 255),
                UniformIntSampler(0, 255),
                UniformIntSampler(0, 255)
            )),
            fx_sampler=UniformSampler(0., 1.),
            fy_sampler=UniformSampler(0., 1.),
        )
        overlay_emoji = OverlayEmojiTransform(
            emoji_size_sampler=UniformSampler(0.2, .5),
            opacity_sampler=UniformSampler(0.7, 1.),
            fx_sampler=UniformSampler(0., 1.),
            fy_sampler=UniformSampler(0., 1.),
        )
        blur = BlurTransform(UniformSampler(0.1, 2.))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            MaybeApplyTransform(p_overlay, overlay_text),
            MaybeApplyTransform(p_overlay, overlay_emoji),
            MaybeApplyTransform(p_blur, blur)
        ])

    def forward(self, video):
        frames = []
        for frame in video:
            frame = np.array(self.transform(frame), copy=True)
            frames.append(frame)
        frames = np.stack(frames)
        frames = torch.from_numpy(frames)
        frames = frames.permute(0, 3, 1, 2)
        return frames.byte()

    def __repr__(self):
        return '{}(p_overlay={}, p_blur={})'.format(self.__class__.__name__, self.p_overlay, self.p_blur)
