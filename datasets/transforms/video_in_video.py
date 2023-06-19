import random
import torch
import torch.nn as nn

from torchvision import transforms
from .samplers import UniformSampler


class VideoInVideo(nn.Module):

    def __init__(self, p=0.3, mix_lamda=(0.3, 0.7)):
        super().__init__()
        self.p = p
        self.mix_lamda_range = mix_lamda
        self.mix_p_sampler = UniformSampler(0., 1.)
        self.mix_lamda_sampler = UniformSampler(*self.mix_lamda_range)

    def _clip(self, value: int, min_value: int, max_value: int) -> int:
        return min(max(value, min_value), max_value)

    def _get_rand_box(self, input_shape, mix_lamda):
        ratio = (1 - mix_lamda) ** 0.5
        input_h, input_w = input_shape[-2:]
        cut_h, cut_w = int(input_h * ratio), int(input_w * ratio)
        cy = torch.randint(input_h, (1,)).item()
        cx = torch.randint(input_w, (1,)).item()
        yl = self._clip(cy - cut_h // 2, 0, input_h)
        yh = self._clip(cy + cut_h // 2, 0, input_h)
        xl = self._clip(cx - cut_w // 2, 0, input_w)
        xh = self._clip(cx + cut_w // 2, 0, input_w)
        return yl, yh, xl, xh

    def _intersection_area(self, box1, box2) -> float:
        inter_q_start = max(box1[0], box2[0])
        inter_q_end = min(box1[1], box2[1])
        inter_r_start = max(box1[2], box2[2])
        inter_r_end = min(box1[3], box2[3])

        return abs(
            max((inter_q_end - inter_q_start, 0))
            * max((inter_r_end - inter_r_start), 0)
        )

    def _mix_videos(self, donor, host):
        mix_lamda = self.mix_lamda_sampler()
        yl, yh, xl, xh = self._get_rand_box(donor.size(), mix_lamda)
        donor = transforms.Resize((yh - yl, xh - xl))(donor)
        host[..., yl:yh, xl:xh] = donor
        return host, mix_lamda

    def forward(self, x_videos, labels=None):
        assert x_videos.size(0) > 1, "Video-in-Video cannot be applied to a single instance."
        assert x_videos.dim() == 4 or x_videos.dim() == 5, "Please correct input shape."

        x_videos = x_videos.permute(0, 1, 4, 2, 3)

        if labels is None:
            labels = torch.eye(x_videos.size(0), device=x_videos.device)

        new_videos = []
        for i in range(x_videos.size(0)):
            if self.mix_p_sampler() < self.p:
                j = random.choice(list(set(range(x_videos.size(0))) - {i}))
                donor = x_videos[i].clone()
                host = x_videos[j].clone()
                mixed_video, mix_lamda = self._mix_videos(donor, host)
                new_videos.append(mixed_video)
                labels[i, j] = mix_lamda
            else:
                new_videos.append(x_videos[i].clone())

        x_videos = torch.stack(new_videos)
        x_videos = x_videos.permute(0, 1, 3, 4, 2)

        return x_videos, labels

    def __repr__(self):
        return '{}(p={}, mix_lamda={})'.format(
            self.__class__.__name__, self.p, self.mix_lamda_range)
