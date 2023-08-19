# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import torch
import numpy as np


class Mixup(object):
    """
    "Mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)". In ICLR, 2018.
    https://github.com/facebookresearch/mixup-cifar10
    """

    def single_domain_mix(self, x, y, alpha=1.0, device='cpu'):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def dual_domains_mix(self, x1, x2, y1, y2, alpha=1.0, device='cpu'):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        mixed_x = lam * x1 + (1 - lam) * x2
        return mixed_x, y1, y2, lam

    def criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
