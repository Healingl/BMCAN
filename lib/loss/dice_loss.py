# !/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Project: MSCGNet
# @IDE: PyCharm
# @File: dice_loss.py
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 2020/10/2
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from lib.loss.loss_utils import *


class DiceLoss(nn.Module):
	def __init__(self, eps=1e-5, ignore_index=None, mode='array'):
		super(DiceLoss, self).__init__()
		self.ignore_index = ignore_index
		self.eps = eps
		self.mode = mode

	def forward(self, prediction, target):
		"""
		pred :    shape (N, C, H, W)
		target :  shape (N, H, W) ground truth
		return:  dice loss
		"""
		target = target.long()
		pred = F.softmax(prediction, 1)
		one_hot_label, mask = self.encode_one_hot_label(pred, target)

		if self.mode == 'array':
			dice_loss = self.cal_dice_loss_by_array(pred, one_hot_label, mask)
		elif self.mode == 'flat':
			dice_loss = self.cal_dice_loss_by_flat(pred,one_hot_label)
		else:
			assert False
		return dice_loss

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
			# print(type(target))
			# target.unsqueeze(1)在第一维度上扩展
			one_hot_label.scatter_(1, target.unsqueeze(1), 1)  #scatter_(dim: _int, index: Tensor, value:Number), fill input by value
			return one_hot_label, None

	def cal_dice_loss_by_array(self,pred,one_hot_label,mask=None):
		intersection = 2 * torch.sum(pred * one_hot_label, dim=(0, 2, 3)) + self.eps
		union = pred + one_hot_label

		if mask is not None:
			union[mask] = 0

		union = torch.sum(union, dim=(0, 2, 3)) + self.eps

		dice = intersection / union
		dice_loss = 1 - dice.mean()
		return dice_loss

	def cal_dice_loss_by_flat(self,pred,one_hot_label):
		num = one_hot_label.size(0)
		pred = pred.view(num, -1)
		one_hot_label = one_hot_label.view(num, -1)
		intersection = (pred * one_hot_label)

		# sum(1) means: add sum at 1 axis
		dice = (2. * intersection.sum(1) + self.eps) / (pred.sum(1) + one_hot_label.sum(1) +  self.eps)
		dice = 1 - dice.sum() / num
		return dice

class BCEDiceLoss(DiceLoss):
	def __init__(self, alpha=1, beta=1):
		super(BCEDiceLoss, self).__init__()
		self.alpha = alpha
		self.beta = beta
	def forward(self, pred, target, ):
		"""
		pred :    shape (N, C, H, W)
		target :  shape (N, H, W) ground truth
		return:  dice loss
		"""
		one_hot_label, _ = self.encode_one_hot_label(pred.long(), target.long())
		bce = F.binary_cross_entropy_with_logits(pred.float(), one_hot_label.float())
		input = F.softmax(pred,dim=1)
		dice = self.cal_dice_loss_by_array(pred=input,one_hot_label=one_hot_label)
		return self.alpha * bce + self.beta * dice




from lib.loss.loss_utils import encode_one_hot_label

import os
if __name__ == "__main__":


	# output: (batch_size, channel(category num), shape0,shape1)
	outputs= torch.rand((2, 3, 16, 16)).long()
	# gt: (batch_size, shape0, shape1)
	labels = torch.from_numpy(np.random.randint(3,size=(2, 16, 16))).long()

	print(labels.shape,labels)
	print('>>>')
	onehot_labels = encode_one_hot_label(pred=outputs, target=labels)

	print(onehot_labels.shape, onehot_labels)

	criterion = nn.CrossEntropyLoss()
	loss = criterion(outputs.float(), onehot_labels.float())
	print(loss)
	# assert False
	# print(outputs.shape)
	# print(labels)
	#
	# dice_loss_array = DiceLoss(mode='array').cuda()
	# dice_loss_array_loss = dice_loss_array(outputs, labels)
	# #
	# print(dice_loss_array_loss.item())
	#
	# dice_loss_flat = DiceLoss(mode='flat').cuda()
	# dice_loss_flat_loss =dice_loss_flat(outputs,labels)
	# print(dice_loss_flat_loss.item())
	#
	# bce_loss = BCEDiceLoss().cuda()
	# bce_loss_score = bce_loss(outputs,labels)
	# print(bce_loss_score.item())