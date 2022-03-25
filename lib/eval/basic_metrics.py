#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Project: MSCGNet
# @IDE: PyCharm
# @File: metrics.py
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 20-9-17
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from scipy import ndimage

def dice(pred, gt):
    smooth = 1e-5

    num = np.sum(pred & gt)
    denom = np.sum(pred) + np.sum(gt)

    if denom == 0:
        return 1
    else:
        return (2.0 * num + smooth) / (denom + smooth)

def cal_batch_dice_mean(outputs, gts):
    """
    训练过程中的dice评估
    :param outputs:
    :param gts:
    :return:
    """
    outputs_np = outputs.data.cpu().numpy()

    # 将(1,2,512,512)转换到(1,512,512,2)
    outputs_np_t = np.transpose(outputs_np, (0, 2, 3, 1))
    outputs_np_label = np.argmax(outputs_np_t, axis=-1)

    outputs_np_label = outputs_np_label.astype(np.uint8)
    target_np = gts.data.cpu().numpy().astype(np.uint8)

    dice_mean_list = []
    for slice_idx in range(outputs_np_label.shape[0]):
        pred_img = outputs_np_label[slice_idx]
        gt_img = target_np[slice_idx]
        current_dice_score = dice(pred=pred_img, gt=gt_img)
        dice_mean_list.append(current_dice_score)
    batch_mean_dice = sum(dice_mean_list) / len(dice_mean_list)

    return batch_mean_dice


def cal_batch_dice_mean_with_class(outputs, gts, class_list = [1,2,3]):
    """
    训练过程中的dice评估
    :param outputs:
    :param gts:
    :return:
    """
    outputs_np = outputs.data.cpu().numpy()

    # 将(1,channels,512,512)转换到(1,512,512,channels)
    outputs_np_t = np.transpose(outputs_np, (0, 2, 3, 1))
    outputs_np_label = np.argmax(outputs_np_t, axis=-1)

    outputs_np_label = outputs_np_label.astype(np.uint8)
    target_np = gts.data.cpu().numpy().astype(np.uint8)

    dice_mean_list = []
    for slice_idx in range(outputs_np_label.shape[0]):
        current_slice_dice_list = []

        pred_img = outputs_np_label[slice_idx]
        gt_img = target_np[slice_idx]

        # class
        for current_class in class_list:
            current_pred_class_img = np.zeros_like(pred_img)
            current_gt_class_img = np.zeros_like(gt_img)

            current_pred_class_img[pred_img == current_class] = 1
            current_gt_class_img[gt_img==current_class] = 1

            current_define_class_dice_score = dice(pred=current_pred_class_img,gt=current_gt_class_img)
            current_slice_dice_list.append(current_define_class_dice_score)

        dice_mean_list.append(current_slice_dice_list)

    batch_mean_dice = np.mean(np.array(dice_mean_list),axis=0)
    batch_class_mean_dice_list =[round(dice_score,4) for dice_score in batch_mean_dice.tolist()]

    all_class_mean_dice = sum(batch_class_mean_dice_list)/len(batch_class_mean_dice_list)

    return all_class_mean_dice, batch_class_mean_dice_list