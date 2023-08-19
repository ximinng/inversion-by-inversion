# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import torch


def extract(a, t, x_shape):
    b, *_ = t.shape
    assert x_shape[0] == b
    out = a.gather(-1, t)  # 1-D tensor, shape: (b,)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # shape: [b, 1, 1, 1]


def unnormalize(x):
    """unnormalize_to_zero_to_one"""
    x = (x + 1) * 0.5  # Map the data interval to [0, 1]
    return torch.clamp(x, 0.0, 1.0)


def normalize(x):
    """normalize_to_neg_one_to_one"""
    x = x * 2 - 1  # Map the data interval to [-1, 1]
    return torch.clamp(x, -1.0, 1.0)
