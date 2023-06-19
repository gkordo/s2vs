import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet(nn.Module):

    def __init__(self, ):
        super(ResNet, self).__init__()

        self.cnn = models.resnet50(pretrained=True)
        del self.cnn.fc

        self.layers = {'layer1': 28, 'layer2': 14, 'layer3': 6, 'layer4': 3}

    def extract_region_vectors(self, x):
        tensors = []
        for nm, module in self.cnn._modules.items():
            if nm not in {'avgpool', 'fc', 'classifier'}:
                x = module(x).contiguous()
                if nm in self.layers:
                    s = self.layers[nm]
                    region_vectors = F.max_pool2d(x, [s, s], int(np.ceil(s / 2)))
                    region_vectors = F.normalize(region_vectors, p=2, dim=1)
                    tensors.append(region_vectors)
        for i in range(len(tensors)):
            tensors[i] = F.normalize(F.adaptive_max_pool2d(tensors[i], tensors[-1].shape[2:]), p=2, dim=1)
        x = torch.cat(tensors, 1)
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        return F.normalize(x, p=2, dim=-1)

    def forward(self, x):
        return self.extract_region_vectors(x)
