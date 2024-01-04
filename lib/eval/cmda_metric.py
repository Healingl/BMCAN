#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Project: BrainCMDABaseline
# @IDE: PyCharm
# @File: cmda_metric.py
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 21-4-21
# @Desc:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from lib.eval.binary_metric import dc, assd, jc, asd

def cmda_eval_metrics(gt, pred, class_label_list=[1,2]):
    """
    {'0':'bg', '1':'la_myo',  '2':'la_blood', '3':'lv_blood', '4':'aa'}
    dc, assd
    :param gt: 3D array
    :param pred: 3D array
    :param class_label_list: [1,2,3]
    :return:
    """

    label_dict = {'0':'bg', '1':'la_myo',  '2':'la_blood', '3':'lv_blood', '4':'aa'}

    pred = pred.astype(dtype='int')
    gt=gt.astype(dtype='int')

    dsc_dict = {}
    assd_dict = {}

    for current_label in class_label_list:
        current_label = int(current_label)

        gt_c = np.zeros(gt.shape)
        y_c = np.zeros(gt.shape)
        gt_c[np.where(gt==current_label)]=1
        y_c[np.where(pred==current_label)]=1

        try:
            current_label_dsc = dc(y_c,gt_c)
        except:
            print('dc error gt:max %s, min %s, y_c:max %s, min %s' % (gt_c.max(), gt_c.min(), y_c.max(), y_c.min()))
            current_label_dsc = 0

        try:
            # current_label_assd = assd(y_c,gt_c) # too slow
            # current_label_assd = asd(y_c, gt_c)
            current_label_assd = 0 #for train
        except:
            print('assd error gt:max %s, min %s, y_c:max %s, min %s' % (gt_c.max(), gt_c.min(), y_c.max(), y_c.min()))
            current_label_assd = 0


        dsc_dict['%s' % (label_dict[str(current_label)])] = round(current_label_dsc,4)
        assd_dict['%s' % (label_dict[str(current_label)])] = round(current_label_assd,4)

    return dsc_dict,assd_dict