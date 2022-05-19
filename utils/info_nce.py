# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: info_nce.py
@version: 1.0
@time: 2022/05/19 15:20:42
@contact: jinxy@pku.edu.cn

my implementation of info nce
"""


import torch
import torch.nn.functional as F
from torch import nn


class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean'):
        super(InfoNCE, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, anchors, positives, labels):
        """

        Args:
            anchors: [B_a, D]
            positives: [B_p, D]
            labels: [B_p]

        Returns:
            Cross Entropy Loss
        """
        anchors, positives = normalize(anchors, positives)
        logits = positives @ anchors.transpose(-2, -1)  # [B_p, B_a]
        labels = torch.LongTensor(labels).to(anchors.device)
        return F.cross_entropy(logits / self.temperature, labels, reduction=self.reduction)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]
