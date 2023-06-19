import enum
import torch
import torch.nn as nn

from model import extractors
from model.layers import PCALayer, VideoNormalizer


class FeatureExtractor(enum.Enum):
    RESNET = enum.auto()

    def get_model(self, dims=512):
        normalizer = VideoNormalizer()
        backbone = self._get_config(self)(dims)
        return nn.Sequential(normalizer, backbone)

    def get_extension(self,):
        return self._get_extension(self)

    def _get_config(self, value):
        return {
            self.RESNET: self._get_resnet,
        }[value]

    def _get_extension(self, value):
        return {
            self.RESNET: 'res',
        }[value]

    def _get_resnet(self, dims):
        backbone = extractors.ResNet()
        white_layer = PCALayer(n_components=dims, pretrained=True)
        return nn.Sequential(backbone, white_layer)
