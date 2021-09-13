# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianKLMatcher(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, train_dis, val_dis):
        assert train_dis.shape == val_dis.shape
        N, c, w, h = val_dis.shape
        train_dis_cp = train_dis.repeat(N, 1, 1, 1)
        val_dis_cp = val_dis.repeat(N, 1, 1, 1)

        val_dis_cp = val_dis_cp.reshape(N, N, c, w, h)
        val_dis_cp = val_dis_cp.transpose(0, 1)
        val_dis_cp = val_dis_cp.reshape(N * N, c, w, h)
        cost = (train_dis_cp - val_dis_cp).abs().sum(dim=[1, 2, 3]).reshape(N, N)
        row_ind, col_ind = linear_sum_assignment(cost)
        list(zip(row_ind, col_ind))
        return train_dis, val_dis[col_ind]


def build_kl_matcher():
    return HungarianKLMatcher()
