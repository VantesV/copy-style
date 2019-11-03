"""
A module describing a CNN for extracting style from an image

‘conv1 1’ (a)
‘conv1 1’ and ‘conv2 1’ (b)
‘conv1 1’, ‘conv2 1’ and ‘conv3 1’ (c)
‘conv1 1’, ‘conv2 1’, ‘conv3 1’ and ‘conv4 1’ (d)
‘conv1 1’, ‘conv2 1’, ‘conv3 1’, ‘conv4 1’ and ‘conv5 1’ (e)
"""
import copy

import torch.nn.functional as F
from torch import nn

from project.art_style.content import ContentLoss
from project.art_style.gram import gram_matrix
from project.art_style.normalization import Normalization


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
