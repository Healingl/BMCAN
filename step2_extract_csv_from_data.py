#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Project: BrainCMDABaseline
# @IDE: PyCharm
# @File: step1_extract_csv_from_data.py
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 21-4-16
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


import os
import nibabel
import pandas as pd
import numpy as np
from copy import copy
import cv2
import random
from scipy import ndimage
import SimpleITK as sitk
import copy
from tqdm import tqdm
from matplotlib import pyplot as plt
import shutil
import nibabel as nib
from lib.utils.simple_parser import Parser

if __name__ == '__main__':
    yaml_config = Parser('./config/data_config/mmwh.yaml')

    data_csv_dir = yaml_config.csv_dir
    if not os.path.exists(data_csv_dir): os.makedirs(data_csv_dir)
    # # # # # # # # # # # # # # # # # # # # # # # #
    # # # Real Test Data Convert to CSV
    # # # # # # # # # # # # # # # # # # # # # # # #

    mmwh_cmda_real_test_ct_data_dir = yaml_config.mmwh_cmda_real_test_ct_data_dir
    mmwh_cmda_real_test_mr_data_dir = yaml_config.mmwh_cmda_real_test_mr_data_dir

    # test_ct
    mmwh_cmda_real_test_ct_img_name_list = [item for item in os.listdir(mmwh_cmda_real_test_ct_data_dir) if 'image_' in item]
    all_mmwh_cmda_real_test_ct_data_list = []
    for data_idx, current_data_name in enumerate(tqdm(mmwh_cmda_real_test_ct_img_name_list)):
        current_base_file_name = current_data_name.replace('image_','')

        current_data_img_path = os.path.abspath(os.path.join(mmwh_cmda_real_test_ct_data_dir, 'image_'+current_base_file_name))
        current_data_label_path = os.path.abspath(os.path.join(mmwh_cmda_real_test_ct_data_dir, 'gth_'+current_base_file_name))
        assert os.path.exists(current_data_img_path), "%s not exist!" % (current_data_img_path)
        assert os.path.exists(current_data_label_path), "%s not exist!" % (current_data_label_path)

        all_mmwh_cmda_real_test_ct_data_list.append([current_data_img_path, current_data_label_path])

    all_mmwh_cmda_real_test_ct_data_csv = pd.DataFrame(columns=["img_path", "label_path"],
                                                     data=all_mmwh_cmda_real_test_ct_data_list)
    all_mmwh_cmda_real_test_ct_data_csv.to_csv(yaml_config.all_mmwh_cmda_real_test_ct_data_csv_path, index=False)

    # test_mr
    mmwh_cmda_real_test_mr_img_name_list = [item for item in os.listdir(mmwh_cmda_real_test_mr_data_dir) if
                                            'image_' in item]
    all_mmwh_cmda_real_test_mr_data_list = []
    for data_idx, current_data_name in enumerate(tqdm(mmwh_cmda_real_test_mr_img_name_list)):
        current_base_file_name = current_data_name.replace('image_', '')

        current_data_img_path = os.path.abspath(
            os.path.join(mmwh_cmda_real_test_mr_data_dir, 'image_' + current_base_file_name))
        current_data_label_path = os.path.abspath(
            os.path.join(mmwh_cmda_real_test_mr_data_dir, 'gth_' + current_base_file_name))
        assert os.path.exists(current_data_img_path), "%s not exist!" % (current_data_img_path)
        assert os.path.exists(current_data_label_path), "%s not exist!" % (current_data_label_path)

        all_mmwh_cmda_real_test_mr_data_list.append([current_data_img_path, current_data_label_path])

    all_mmwh_cmda_real_test_mr_data_csv = pd.DataFrame(columns=["img_path", "label_path"],
                                                       data=all_mmwh_cmda_real_test_mr_data_list)
    all_mmwh_cmda_real_test_mr_data_csv.to_csv(yaml_config.all_mmwh_cmda_real_test_mr_data_csv_path, index=False)


    # # # # # # # # # # # # # # # # # # # # # # # #
    # # # Sample Data Convert to CSV
    # # # # # # # # # # # # # # # # # # # # # # # #

    sample_ct_train_dir = yaml_config['sample_ct_train_dir']
    sample_mr_train_dir = yaml_config['sample_mr_train_dir']

    sample_ct_val_dir = yaml_config['sample_ct_val_dir']
    sample_mr_val_dir = yaml_config['sample_mr_val_dir']

    # ct_train
    sample_ct_train_img_dir = os.path.join(sample_ct_train_dir,'images/')
    sample_ct_train_label_dir = os.path.join(sample_ct_train_dir,'labels/')
    data_name_list = sorted(os.listdir(sample_ct_train_img_dir))

    ct_train_result_data_list = []
    for data_idx, current_data_name in enumerate(tqdm(data_name_list)):
        current_data_img_path = os.path.abspath(os.path.join(sample_ct_train_img_dir,current_data_name))
        current_data_label_path = os.path.abspath(os.path.join(sample_ct_train_label_dir,current_data_name))
        assert os.path.exists(current_data_img_path), "%s not exist!"%(current_data_img_path)
        assert os.path.exists(current_data_label_path), "%s not exist!"%(current_data_label_path)

        ct_train_result_data_list.append([current_data_img_path,current_data_label_path])

    all_mmwh_cmda_ct_train_sample_csv = pd.DataFrame(columns=["img_path", "label_path"],
                                                     data=ct_train_result_data_list)
    all_mmwh_cmda_ct_train_sample_csv.to_csv(yaml_config.all_mmwh_cmda_ct_train_sample_csv_path, index=False)

    # mr_train
    sample_mr_train_img_dir = os.path.join(sample_mr_train_dir, 'images/')
    sample_mr_train_label_dir = os.path.join(sample_mr_train_dir, 'labels/')
    data_name_list = sorted(os.listdir(sample_mr_train_img_dir))

    mr_train_result_data_list = []
    for data_idx, current_data_name in enumerate(tqdm(data_name_list)):
        current_data_img_path = os.path.abspath(os.path.join(sample_mr_train_img_dir, current_data_name))
        current_data_label_path = os.path.abspath(os.path.join(sample_mr_train_label_dir, current_data_name))
        assert os.path.exists(current_data_img_path), "%s not exist!" % (current_data_img_path)
        assert os.path.exists(current_data_label_path), "%s not exist!" % (current_data_label_path)

        mr_train_result_data_list.append([current_data_img_path, current_data_label_path])

    all_mmwh_cmda_mr_train_sample_csv = pd.DataFrame(columns=["img_path", "label_path"],
                                                     data=mr_train_result_data_list)
    all_mmwh_cmda_mr_train_sample_csv.to_csv(yaml_config.all_mmwh_cmda_mr_train_sample_csv_path, index=False)

    # ct_val
    sample_ct_val_img_dir = os.path.join(sample_ct_val_dir, 'images/')
    sample_ct_val_label_dir = os.path.join(sample_ct_val_dir, 'labels/')
    data_name_list = sorted(os.listdir(sample_ct_val_img_dir))

    ct_val_result_data_list = []
    for data_idx, current_data_name in enumerate(tqdm(data_name_list)):
        current_data_img_path = os.path.abspath(os.path.join(sample_ct_val_img_dir, current_data_name))
        current_data_label_path = os.path.abspath(os.path.join(sample_ct_val_label_dir, current_data_name))
        assert os.path.exists(current_data_img_path), "%s not exist!" % (current_data_img_path)
        assert os.path.exists(current_data_label_path), "%s not exist!" % (current_data_label_path)

        ct_val_result_data_list.append([current_data_img_path, current_data_label_path])

    all_mmwh_cmda_ct_val_sample_csv = pd.DataFrame(columns=["img_path", "label_path"],
                                                     data=ct_val_result_data_list)
    all_mmwh_cmda_ct_val_sample_csv.to_csv(yaml_config.all_mmwh_cmda_ct_val_sample_csv_path, index=False)

    # mr_val
    sample_mr_val_img_dir = os.path.join(sample_mr_val_dir, 'images/')
    sample_mr_val_label_dir = os.path.join(sample_mr_val_dir, 'labels/')
    data_name_list = sorted(os.listdir(sample_mr_val_img_dir))

    mr_val_result_data_list = []
    for data_idx, current_data_name in enumerate(tqdm(data_name_list)):
        current_data_img_path = os.path.abspath(os.path.join(sample_mr_val_img_dir, current_data_name))
        current_data_label_path = os.path.abspath(os.path.join(sample_mr_val_label_dir, current_data_name))
        assert os.path.exists(current_data_img_path), "%s not exist!" % (current_data_img_path)
        assert os.path.exists(current_data_label_path), "%s not exist!" % (current_data_label_path)

        mr_val_result_data_list.append([current_data_img_path, current_data_label_path])

    all_mmwh_cmda_mr_val_sample_csv = pd.DataFrame(columns=["img_path", "label_path"],
                                                   data=mr_val_result_data_list)
    all_mmwh_cmda_mr_val_sample_csv.to_csv(yaml_config.all_mmwh_cmda_mr_val_sample_csv_path, index=False)


