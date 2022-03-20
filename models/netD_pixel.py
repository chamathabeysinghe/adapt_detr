# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class NetDPixel(nn.Module):
    def __init__(self, context=False):
        super(NetDPixel, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1,
                               padding=0, bias=False)

        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1,
                               padding=0, bias=False)
        # self.conv1 = conv1x1(512, 256)
        # self.conv2 = conv1x1(256, 128)
        # self.conv3 = conv1x1(128, 1)
        self.context = context
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                # m.bias.data.zero_()
                # normal_init(self.conv1, 0, 0.01)
                # normal_init(self.conv2, 0, 0.01)
                # normal_init(self.conv3, 0, 0.01)

    def forward(self, x):
        # default no droopout
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.context:
            feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
            # feat = x
            x = self.conv3(x)
            return F.sigmoid(x), feat  #F.sigmoid(x),feat#torch.cat((feat1,feat2),1)#F
        else:
            x = self.conv3(x)
            return F.sigmoid(x)  # F.sigmoid(x)


def build_discriminator(args):
    discriminator = NetDPixel()
    return discriminator
