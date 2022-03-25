#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Project: BrainCMDABaselinePS
# @IDE: PyCharm
# @File: test_BCE_loss.py
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 21-4-29
# @Desc: 相同shape
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
import torch.nn.functional as F
import torch.nn as nn

torch.manual_seed(2019)
# shape: (5,2)
"""
tensor([[-0.1187,  0.2110],
        [ 0.7463, -0.6136],
        [-0.1186,  1.5565],
        [ 1.3662,  1.0199],
        [ 2.4644,  1.1630]])
"""
output = torch.randn(5, 2)  # 网络输出
print(output)
# shape: (5,2)
"""
tensor([[0., 1.],
        [1., 1.],
        [0., 0.],
        [1., 0.],
        [0., 1.]])
"""
target = torch.ones((5,2), dtype=torch.float).random_(2)  # 真实标签
print(target)

# 实例化类
criterion = nn.BCEWithLogitsLoss()
loss = criterion(output, target)
print('loss:{}'.format(loss))