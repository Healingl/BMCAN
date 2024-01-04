#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel

from albumentations import (
    Compose,
    OneOf,
    Flip,
    PadIfNeeded,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    MotionBlur,
    OpticalDistortion,
    RandomSizedCrop,
    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    ShiftScaleRotate,
    CenterCrop,
    Transpose,
    GridDistortion,
    ElasticTransform,
    RandomGamma,
    RandomBrightnessContrast,
    RandomContrast,
    RandomBrightness,
    CLAHE,
    HueSaturationValue,
    Blur,
    MedianBlur,
    ChannelShuffle,
)

def mmwh_process_img( image_data, modality):
    assert modality in ['ct', 'mr']

    if modality == 'ct':
        # ct
        param1 = -2.8
        param2 = 3.2
    else:
        # mr
        param1 = -1.8
        param2 = 4.4

    process_image_data = 2 * (image_data - param1) / (param2 - param1) - 1.0
    return process_image_data

def load_nifty_volume_as_array(filename, with_header=False, is_mmwh=False):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    but mmwh is [z,x,y]
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
        with_header: return affine and hearder infomation
    outputs:
        data: a numpy data array
    """
    img = nibabel.load(filename)
    data = img.get_data()

    assert len(data.shape) == 3

    try:
        data = np.transpose(data, [2, 1, 0])
    except ValueError:
        data = data.reshape(data.shape[0],data.shape[1],-1)
        data = np.transpose(data, [2, 1, 0])
    if is_mmwh:
        # [z,y,x] -> [z,x,y]
        data = np.transpose(data, [0, 2, 1])
        data = data[:, ::-1, ::-1]
    if (with_header):
        return data, img.affine, img.header
    else:
        return data

def itensity_normalization(image_narray, norm_type='max_min'):
    if norm_type == 'full_volume_mean':
        norm_img_narray = (image_narray - image_narray.mean()) / image_narray.std()
    elif norm_type == 'max_min':
        norm_img_narray = (image_narray - image_narray.min()) / (image_narray.max() - image_narray.min())
    elif norm_type == 'non_normal':
        norm_img_narray = image_narray
    elif norm_type == 'non_zero_normal':
        pixels = image_narray[image_narray > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (image_narray - mean) / std
        out_random = np.random.normal(0, 1, size=image_narray.shape)
        out[image_narray == 0] = out_random[image_narray == 0]
        norm_img_narray = out
    elif norm_type == 'mr_normal':
        image_narray[image_narray > 4095] = 4095
        norm_img_narray = image_narray * 2. / 4095 - 1

    else:
        assert False
    return norm_img_narray

class MMWHPairedDataset(Dataset):
    def __len__(self):
        return len(self.mmwh_ct_sample_data_csv)

    def __init__(self,
                 mmwh_ct_sample_data_csv_path,
                 mmwh_mr_sample_data_csv_path,
                 mode='train',
                 data_num=-1,
                 use_aug=False):

        assert mode in ['train', 'val']

        self.mode = mode
        self.use_aug = use_aug


        self.mmwh_ct_sample_data_csv = pd.read_csv(mmwh_ct_sample_data_csv_path)
        self.mmwh_mr_sample_data_csv = pd.read_csv(mmwh_mr_sample_data_csv_path)



        if data_num == -1:
            self.mmwh_ct_sample_data_csv = self.mmwh_ct_sample_data_csv
            self.mmwh_mr_sample_data_csv = self.mmwh_mr_sample_data_csv
        else:
            self.mmwh_ct_sample_data_csv = self.mmwh_ct_sample_data_csv.sample(data_num)
            self.mmwh_mr_sample_data_csv = self.mmwh_mr_sample_data_csv.sample(data_num)

        print(">>" * 30, "read mmwh_ct_sample_data_csv:", mmwh_ct_sample_data_csv_path, 'data num: ',len(self.mmwh_ct_sample_data_csv), ">>" * 30)
        print(">>" * 30, "read mmwh_mr_sample_data_csv:", mmwh_mr_sample_data_csv_path, 'data num: ',len(self.mmwh_mr_sample_data_csv), ">>" * 30)


        self.mmwh_ct_sample_data_img_path_list = self.mmwh_ct_sample_data_csv['img_path'].tolist()
        self.mmwh_ct_sample_data_label_path_list = self.mmwh_ct_sample_data_csv['label_path'].tolist()

    def __getitem__(self, index):

        f_ct_sample_image_path = self.mmwh_ct_sample_data_img_path_list[index]
        f_ct_sample_label_path = self.mmwh_ct_sample_data_label_path_list[index]

        # based randomly
        f_mr_randome_select_csv = self.mmwh_mr_sample_data_csv.sample(n=1,random_state=np.random.randint(0,2021))
        f_mr_sample_image_path = f_mr_randome_select_csv['img_path'].tolist()[0]
        f_mr_sample_label_path = f_mr_randome_select_csv['label_path'].tolist()[0]

        current_sample_ct_image = np.load(f_ct_sample_image_path)
        current_sample_ct_label = np.load(f_ct_sample_label_path)

        current_sample_mr_image = np.load(f_mr_sample_image_path)
        current_sample_mr_label = np.load(f_mr_sample_label_path)

        assert len(current_sample_ct_image.shape) == len(current_sample_ct_label.shape) == len(current_sample_mr_image.shape) ==len(current_sample_mr_label.shape) == 3

        # preprocess, select middle as ground truth
        # [256,256,3]
        current_sample_ct_image_prep = self.process_img(image_data=current_sample_ct_image,modality='ct')
        # [256,256]
        current_sample_ct_label_prep = current_sample_ct_label[:, :, 1]

        # [256,256,3] [y,x,c]
        current_sample_mr_image_prep = self.process_img(image_data=current_sample_mr_image, modality='mr')
        # [256,256]
        current_sample_mr_label_prep = current_sample_mr_label[:, :, 1]


        # #
        if self.use_aug and self.mode == 'train':
            current_sample_ct_aug_feature, current_sample_ct_aug_label = self.train_augmentation(current_sample_ct_image_prep, current_sample_ct_label_prep)
            current_sample_mr_aug_feature, current_sample_mr_aug_label = self.train_augmentation(current_sample_mr_image_prep, current_sample_mr_label_prep)
        else:
            current_sample_ct_aug_feature, current_sample_ct_aug_label = current_sample_ct_image_prep, current_sample_ct_label_prep
            current_sample_mr_aug_feature, current_sample_mr_aug_label = current_sample_mr_image_prep, current_sample_mr_label_prep

        # transpose: [y, x, c] -> [c, y, x]
        current_sample_ct_aug_feature = np.transpose(current_sample_ct_aug_feature, axes=[2, 0, 1])
        current_sample_ct_aug_feature = np.array([current_sample_ct_aug_feature[1]])
        current_sample_ct_aug_label = np.array(current_sample_ct_aug_label)

        current_sample_mr_aug_feature = np.transpose(current_sample_mr_aug_feature, axes=[2, 0, 1])
        current_sample_mr_aug_feature = np.array([current_sample_mr_aug_feature[1]])
        current_sample_mr_aug_label = np.array(current_sample_mr_aug_label)

        # ct
        current_sample_ct_feature_tensor = torch.from_numpy(current_sample_ct_aug_feature).float()
        current_sample_ct_label_tensor = torch.from_numpy(current_sample_ct_aug_label).long()

        # mr
        current_sample_mr_feature_tensor = torch.from_numpy(current_sample_mr_aug_feature).float()
        current_sample_mr_label_tensor = torch.from_numpy(current_sample_mr_aug_label).long()

        return current_sample_ct_feature_tensor, current_sample_ct_label_tensor, current_sample_mr_feature_tensor, current_sample_mr_label_tensor

    @classmethod
    def train_augmentation(cls, img, mask):
        aug = Compose([
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            # RandomRotate90(p=0.5),
        ])

        auged = aug(image=img, mask=mask)
        return auged['image'], auged['mask']

    def process_img(self, image_data, modality):
        assert modality in ['ct','mr']

        if modality == 'ct':
            # ct
            param1 = -2.8
            param2 = 3.2
        else:
            # mr
            param1 = -1.8
            param2 = 4.4

        process_image_data = 2 * (image_data - param1) / (param2 - param1) - 1.0
        return process_image_data

