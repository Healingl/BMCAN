#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Project: Brain2DSeg
# @IDE: PyCharm
# @File: gd_loss.py
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 20-10-9
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super(GeneralizedDiceLoss, self).__init__()
        self.eps = eps

    def forward(self, predict, true):
        """
        Implementation of generalized dice loss for multi-class semantic segmentation
        Args:
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        true: a tensor of shape [B, H, W].
        Returns:
        dice_loss: the Sørensen–Dice loss.
        """
        num_classes = predict.shape[1]

        true_dummy = torch.eye(num_classes)[true.long()]
        true_dummy = true_dummy.permute(0, 3, 1, 2)

        probas = F.softmax(predict, dim=1)

        true_dummy = true_dummy.type(predict.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_dummy, dims)
        cardinality = torch.sum(probas + true_dummy, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return 1 - dice_loss


# from lib.loss.loss_utils import encode_one_hot_label
import numpy as np
import os
if __name__ == "__main__":


    # output: (batch_size, channel(category num), shape0,shape1)
    outputs= torch.rand((2, 3, 16, 16))
    # gt: (batch_size, shape0, shape1)
    labels = torch.from_numpy(np.random.randint(3,size=(2, 16,16)))

    seg_criterion = GeneralizedDiceLoss()
    loss = seg_criterion(predict=outputs,true=labels)
    print(loss)
#