#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import tensorflow as tf
import os
import numpy as np
import glob
from tqdm import tqdm
import sys

tf.enable_eager_execution()


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={  # configuration for decoding tf_record file
            'dsize_dim0': tf.FixedLenFeature([], tf.int64),
            'dsize_dim1': tf.FixedLenFeature([], tf.int64),
            'dsize_dim2': tf.FixedLenFeature([], tf.int64),
            'lsize_dim0': tf.FixedLenFeature([], tf.int64),
            'lsize_dim1': tf.FixedLenFeature([], tf.int64),
            'lsize_dim2': tf.FixedLenFeature([], tf.int64),
            'data_vol': tf.FixedLenFeature([], tf.string),
            'label_vol': tf.FixedLenFeature([], tf.string)})
    return features


def convert_tf_record_numpy(input_folders, output_folder, output_prefix):
    """
    input_folders: input_folders containing data
    output_folders: images saved as numpy in subfolders images, labels
    """
    IMG_SIZE = [256, 256, 3]
    all_tf_files = []
    for folder in input_folders:
        all_tf_files += [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('tfrecords')]

    n_files = len(all_tf_files)

    if not os.path.exists(os.path.join(output_folder, 'images')):
        os.makedirs(os.path.join(output_folder, 'images'))

    if not os.path.exists(os.path.join(output_folder, 'labels')):
        os.makedirs(os.path.join(output_folder, 'labels'))

    file_queue = tf.train.string_input_producer(all_tf_files)

    with tf.Session() as sess:
        all_data = read_and_decode(file_queue)
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(n_files):
            sample = sess.run(all_data)
            img_vol = tf.decode_raw(sample['data_vol'], tf.float32)
            label_vol = tf.decode_raw(sample['label_vol'], tf.float32)

            img_vol = tf.reshape(img_vol, IMG_SIZE)
            label_vol = tf.reshape(label_vol, IMG_SIZE)

            # take a slice of size 3 for contextual information as mentioned in the paper
            img_vol = img_vol.eval()
            label_vol = label_vol.eval()

            label_vol = label_vol

            np.save(os.path.join(output_folder, 'images', output_prefix + str(i).zfill(6)), img_vol)
            np.save(os.path.join(output_folder, 'labels', output_prefix + str(i).zfill(6)), label_vol)

        coord.request_stop()
        coord.join(threads)


features = {  # configuration for decoding tf_record file
    'dsize_dim0': tf.FixedLenFeature([], tf.int64),
    'dsize_dim1': tf.FixedLenFeature([], tf.int64),
    'dsize_dim2': tf.FixedLenFeature([], tf.int64),
    'lsize_dim0': tf.FixedLenFeature([], tf.int64),
    'lsize_dim1': tf.FixedLenFeature([], tf.int64),
    'lsize_dim2': tf.FixedLenFeature([], tf.int64),
    'data_vol': tf.FixedLenFeature([], tf.string),
    'label_vol': tf.FixedLenFeature([], tf.string)}


def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, features)


def convert_tf_record_numpy_fast(input_folders, output_folder, output_prefix):
    if not os.path.exists(os.path.join(output_folder, 'images')):
        os.makedirs(os.path.join(output_folder, 'images'))

    if not os.path.exists(os.path.join(output_folder, 'labels')):
        os.makedirs(os.path.join(output_folder, 'labels'))

    IMG_SIZE = [256, 256, 3]
    all_tf_files = glob.glob(input_folders[0] + '/*.tfrecords')
    image_dataset = []
    print(input_folders[0],'reading tf_record files')
    for it, i in enumerate(tqdm(all_tf_files)):
        image_dataset += [tf.data.TFRecordDataset(i).map(_parse_image_function)]

    print(output_folder,'writing tf_record files')
    for i, j in enumerate(tqdm(image_dataset)):
        for data in j:
            img_numpy = tf.decode_raw(data['data_vol'], tf.float32).numpy()
            label_numpy = tf.decode_raw(data['label_vol'], tf.float32).numpy()

            img_numpy = img_numpy.reshape(IMG_SIZE)
            label_numpy = label_numpy.reshape(IMG_SIZE)

            # only labels
            if np.sum(label_numpy) == 0:
                continue

            np.save(os.path.join(output_folder, 'images', output_prefix + str(i).zfill(6)), img_numpy)
            np.save(os.path.join(output_folder, 'labels', output_prefix + str(i).zfill(6)), label_numpy)

from lib.utils.simple_parser import Parser

if __name__ == '__main__':
    yaml_config = Parser('./config/data_config/mmwh.yaml')

    mmwh_cmda_tfrecord_data_dir = yaml_config['mmwh_cmda_tfrecord_data_dir']
    mmwh_cmda_preprocess_npy_data_dir = yaml_config['mmwh_cmda_preprocess_npy_data_dir']

    assert os.path.exists(mmwh_cmda_tfrecord_data_dir), 'source tfrecord not exist! '
    if not os.path.exists(mmwh_cmda_preprocess_npy_data_dir): os.makedirs(mmwh_cmda_preprocess_npy_data_dir)

    # # # # # # # # # # # # # # # # # # # # # # # #
    # # # Convert CT
    # # # # # # # # # # # # # # # # # # # # # # # #
    print('Convert CT')
    modality_ct = 'ct'
    ct_output_train_dir = os.path.join(mmwh_cmda_preprocess_npy_data_dir, modality_ct + '_train')
    ct_output_test_dir = os.path.join(mmwh_cmda_preprocess_npy_data_dir, modality_ct + '_test')

    if not os.path.exists(ct_output_train_dir): os.makedirs(ct_output_train_dir)
    if not os.path.exists(ct_output_test_dir): os.makedirs(ct_output_test_dir)

    # input_folders, output_folder, output_prefix
    convert_tf_record_numpy_fast(input_folders=[mmwh_cmda_tfrecord_data_dir+'%s_train_tfs' % modality_ct], output_folder=ct_output_train_dir, output_prefix=modality_ct)
    convert_tf_record_numpy_fast(input_folders=[mmwh_cmda_tfrecord_data_dir+'%s_val_tfs' % modality_ct], output_folder=ct_output_test_dir, output_prefix=modality_ct)

    # # # # # # # # # # # # # # # # # # # # # # # #
    # # # Convert MR
    # # # # # # # # # # # # # # # # # # # # # # # #
    print('Convert MR')
    modality_mr = 'mr'
    mr_output_train_dir = os.path.join(mmwh_cmda_preprocess_npy_data_dir, modality_mr + '_train')
    mr_output_test_dir = os.path.join(mmwh_cmda_preprocess_npy_data_dir, modality_mr + '_test')

    if not os.path.exists(mr_output_train_dir): os.makedirs(mr_output_train_dir)
    if not os.path.exists(mr_output_test_dir): os.makedirs(mr_output_test_dir)

    # input_folders, output_folder, output_prefix
    convert_tf_record_numpy_fast(input_folders=[mmwh_cmda_tfrecord_data_dir + '%s_train_tfs' % modality_mr],
                            output_folder=mr_output_train_dir, output_prefix=modality_mr)
    convert_tf_record_numpy_fast(input_folders=[mmwh_cmda_tfrecord_data_dir + '%s_val_tfs' % modality_mr],
                            output_folder=mr_output_test_dir, output_prefix=modality_mr)


