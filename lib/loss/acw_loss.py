import torch
import torch.nn as nn
import torch.nn.functional as F


class ACW_loss(nn.Module):
    def __init__(self,  ini_weight=0, ini_iteration=0, eps=1e-5, ignore_index=255):
        super(ACW_loss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = ini_weight
        self.itr = ini_iteration
        self.eps = eps

    def forward(self, prediction, target):
        """
        pred :    shape (N, C, H, W)
        target :  shape (N, H, W) ground truth
        return:  loss_acw
        """
        prediction = prediction.float()
        target = target.long()
        pred = F.softmax(prediction, 1)
        one_hot_label, mask = self.encode_one_hot_label(pred, target)

        acw = self.adaptive_class_weight(pred, one_hot_label, mask)

        err = torch.pow((one_hot_label - pred), 2)
        # one = torch.ones_like(err)

        pnc = err - ((1. - err + self.eps) / (1. + err + self.eps)).log()
        loss_pnc = torch.sum(acw * pnc, 1)


        intersection = 2 * torch.sum(pred * one_hot_label, dim=(0, 2, 3)) + self.eps
        union = pred + one_hot_label

        if mask is not None:
            union[mask] = 0

        union = torch.sum(union, dim=(0, 2, 3)) + self.eps

        dice = intersection / union

        return loss_pnc.mean() - dice.mean().log()

    def adaptive_class_weight(self, pred, one_hot_label, mask=None):
        self.itr += 1

        sum_class = torch.sum(one_hot_label, dim=(0, 2, 3))
        sum_norm = sum_class / sum_class.sum()

        self.weight = (self.weight * (self.itr - 1) + sum_norm) / self.itr
        mfb = self.weight.mean() / (self.weight + self.eps)
        mfb = mfb / mfb.sum()
        acw = (1. + pred + one_hot_label) * mfb.unsqueeze(-1).unsqueeze(-1)

        if mask is not None:
            acw[mask] = 0

        return acw

    def encode_one_hot_label(self, pred, target):
        one_hot_label = pred.detach() * 0
        if self.ignore_index is not None:
            mask = (target == self.ignore_index)
            target = target.clone()
            target[mask] = 0
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(one_hot_label)
            one_hot_label[mask] = 0
            return one_hot_label, mask
        else:
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)  #scatter_(dim: _int, index: Tensor, value:Number), fill input by value
            return one_hot_label, None

import os
import numpy as np

"""
item()返回的是tensor中的值，且只能返回单个值（标量），不能返回向量，使用返回loss等。
detach()阻断反向传播，返回值仍为tensor
cpu()将变量放在cpu上，仍为tensor：

numpy()将tensor转换为numpy：
注意cuda上面的变量类型只能是tensor，不能是其他


gpu_info.cpu().numpy()
"""

if __name__ == "__main__":
    # output: (batch_size, channel(category num), shape0,shape1)
    outputs = torch.rand((2, 3, 16, 16)).float()
    # gt: (batch_size, shape0, shape1)
    labels = torch.from_numpy(np.random.randint(3, size=(2, 16, 16))).long()
    print(labels.shape, labels)

    criterion = ACW_loss()
    main_loss = criterion(outputs, labels)

    print(main_loss.item())

