# -*- coding: utf-8 -*-
#
# from skimage.exposure import exposure
import os
import nibabel
import numpy as np
import random
from scipy import ndimage
import SimpleITK as sitk

import cv2
def crop_img_from_slice(origin_volumn,crop_point, crop_size):
    """

    :param origin_volumn: (y,x)
    :param crop_point: (current_crop_y, current_crop_x )
    :param crop_size: (crop_y,crop_x)
    :return:
    """
    current_crop_y, current_crop_x = crop_point

    cube = origin_volumn[current_crop_y:current_crop_y + crop_size[0], current_crop_x:current_crop_x + crop_size[1]]

    return cube


def put_img_to_slice(origin_img, input_img,crop_point, crop_size):
    """

    :param origin_volumn: (y,x)
    :param crop_point: (current_crop_y, current_crop_x )
    :param crop_size: (crop_y,crop_x)
    :return:
    """
    current_crop_y, current_crop_x = crop_point

    origin_img[current_crop_y:current_crop_y + crop_size[0], current_crop_x:current_crop_x + crop_size[1]] = input_img

    return origin_img


# 对比度增强
def apply_contrast(image):
    # Apply random brightness but keep values in [0, 1]
    # We apply a quadratic function with the form y = ax^2 + bx
    # Visualization: https://www.desmos.com/calculator/zzz75gguna
    delta = 0.04
    a = -4 * delta
    b = 1 - a
    return a * (image*image) + b * (image)

# 对三维numpy array的cube进行不同方向的翻转
def rotate_cube_on_axis(img_cube, slice_direction=None):
    """
    transpose a list of volumes
    inputs:
        volumes: a list of nd volumes
        slice_direction: 'axial', 'sagittal', or 'coronal'
    outputs:
        tr_volumes: a list of transposed volumes
    """
    if (slice_direction == 'axial'):
        rotate_cube = img_cube
    elif (slice_direction == 'sagittal'):
        rotate_cube = np.transpose(img_cube, (2, 0, 1))
    elif (slice_direction == 'coronal'):
        rotate_cube = np.transpose(img_cube, (1, 0, 2))
    else:
        print('undefined slice direction:', slice_direction)
        rotate_cube = img_cube
    return rotate_cube




# 裁剪较小的值和较大的值
def cut_off_values_upper_lower_percentile(image,mask=None,percentile_lower = 0.2 , percentile_upper =99.8):
    if mask is None:
        mask = image != image[0,0,0]
    cut_off_lower = np.percentile(image[mask!=0].ravel(),percentile_lower)
    cut_off_upper = np.percentile(image[mask!=0].ravel(),percentile_upper)
    res = np.copy(image)
    res[(res < cut_off_lower) & (mask != 0)] = cut_off_lower
    res[(res > cut_off_upper) & (mask != 0)] = cut_off_upper
    return res

# 将nii文件或者mha文件转化为numpy array
# 例子：
# img_array = load_3d_volume_as_array('t1.nii')
#
def load_3d_volume_as_array(filename):
    if ('.nii' in filename):
        return load_nifty_volume_as_array(filename)
    elif ('.nii.gz' in filename):
        return load_nifty_volume_as_array(filename)
    elif ('.mha' in filename):
        return load_mha_volume_as_array(filename)
    raise ValueError('{0:} unspported file format'.format(filename))


# 读取mha文件并转为numpy array
# 例子：
# img_array = load_mha_volume_as_array('t1.mha')
#
def load_mha_volume_as_array(filename):
    img = sitk.ReadImage(filename)
    nda = sitk.GetArrayFromImage(img)
    return nda

# 体素缩放
def volumn_resize(volume, target_shape):
    '''
    resize volume to specified shape
    '''
    if target_shape[0] <= 0:
        target_shape[0] = volume.shape[0]
    if target_shape[1] <= 0:
        target_shape[1] = volume.shape[1]
    if target_shape[2] <= 0:
        target_shape[2] = volume.shape[2]

    D, H, W = volume.shape

    # cv2 can not process image with channels > 512
    if W <= 512:
        res = cv2.resize(np.float32(volume), dsize=(target_shape[1], target_shape[0]))
    else:
        N = 512
        results = []
        for i in range(0, int(W / N + 1)):
            l = i * N
            r = min((i + 1) * N, W)
            patch = volume[:, :, l:r]
            resized_patch = cv2.resize(np.float32(patch), dsize=(target_shape[1], target_shape[0]))
            if len(resized_patch.shape) == 2:
                resized_patch = np.expand_dims(resized_patch, axis=-1)
            results.append(resized_patch)

        res = np.concatenate(results, axis=-1)

    res = np.transpose(res, (2, 1, 0))
    D, H, W = res.shape
    if W <= 512:
        res = cv2.resize(np.float32(res), dsize=(target_shape[1], target_shape[2]))
    else:
        N = 512
        results = []
        for i in range(0, int(W / N + 1)):
            l = i * N
            r = min((i + 1) * N, W)
            patch = res[:, :, l:r]
            resized_patch = cv2.resize(np.float32(patch), dsize=(target_shape[1], target_shape[2]))
            if len(resized_patch.shape) == 2:
                resized_patch = np.expand_dims(resized_patch, axis=-1)
            results.append(resized_patch)

        res = np.concatenate(results, axis=-1)

    res = np.transpose(res, (2, 1, 0))
    return res

# 等方性变换（spacing变换）
def interpolate_volume(volume, org_spacing, expect_spacing):
    D, H, W = volume.shape
    scale = np.array(org_spacing) / np.array(expect_spacing)
    nW, nH, nD = np.int32(np.array([W, H, D]) * scale)
    new_volume = volumn_resize(volume, [nD, nH, nW])
    return new_volume


def read_nii_as_narray(nii_file_path,x_spacing=None, y_spacing=None, is_mmwh=False):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # simple itk读取nii图像
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # 读取当前病人的CT图像
    patient_itk_data = sitk.ReadImage(nii_file_path)

    # 获取CT图像的相关信息，原始坐标，spacing，方向
    origin = np.array(patient_itk_data.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(patient_itk_data.GetSpacing())  # spacing of voxels in world coor. (mm)
    direction = np.array(patient_itk_data.GetDirection())

    # 获得numpy格式的volume数据
    patient_volume_narray = sitk.GetArrayFromImage(patient_itk_data)  # z, y, x

    if x_spacing != None or y_spacing != None:

        if x_spacing == None:
            x_spacing = spacing[0]

        if y_spacing == None:
            y_spacing = spacing[1]

        new_spacing = (x_spacing, y_spacing, spacing[2])

        # spacing重采样变换
        patient_volume_narray = interpolate_volume(patient_volume_narray, spacing, new_spacing)

        spacing = new_spacing

    return patient_volume_narray, spacing


# 读取nii文件并转化为numpy array
# 例子：
# img_array = load_nifty_volume_as_array('t1.nii')
#
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

def convert_nii_to_hdr_file(nii_file_path,hdr_file_path):
    seg_img = sitk.ReadImage(nii_file_path)
    sitk.WriteImage(seg_img, hdr_file_path)

def save_nparray_as_nifty_volume(input_nparray, save_path, img_affine, img_header):
    """
    (z,y,x) -> (x,y,z)
    :param input_nparray:
    :param affine:
    :return:
    """
    input_nii_nparray = np.transpose(input_nparray, [2, 1, 0])
    output_nii = nibabel.Nifti1Image(input_nii_nparray, img_affine,header=img_header)
    output_nii.to_filename(save_path)

# 将一个三维的numpy array进行归一化并输出
# 例子:
# img_origin = numpy.array((255,255,140))
# img_norm = itensity_normalize_one_volume(img_origin)
#
def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzero region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """

    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean) / std
    out_random = np.random.normal(0, 1, size=volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out

def normalize_one_volumn_enhanced(volume):
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean) / std
    out = apply_contrast(out)
    out[volume == 0] = 0
    return out



# 得到肿瘤label非零区域三维框的左下角和右上角,并以margin进行膨胀（ND：Non-zero Dilate)
# 例子：
# bounding_box = get_ND_bounding_box(label_array,margin=5)
# print(bounding_box)
# z,y,x
# 输出：([64, 75, 75], [131, 180, 147])
# 也可写作：
# bbmin, bbmax = get_ND_bounding_box(label_array, margin)
def get_ND_bounding_box(label, margin=2):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = label.shape
    if (type(margin) is int):
        margin = [margin] * len(input_shape)
    assert (len(input_shape) == len(margin))
    indxes = np.nonzero(label)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    return idx_min, idx_max

# 根据get_ND_bounding_box得到的bounding_box裁剪出三维立方体（长方体或者正方体),即从
# 例子：
# bbmin, bbmax = get_ND_bounding_box(volume, margin)
# volume = crop_ND_volume_with_bounding_box(volume, bbmin, bbmax)
def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert (dim >= 2 and dim <= 5)
    if (dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1))]
    elif (dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1))]
    elif (dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1))]
    elif (dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1),
                               range(min_idx[4], max_idx[4] + 1))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output

# 尽管看起来这里和上面的crop_ND_volume_with_bounding_box很相似，但是set_ND_volume_roi_with_bounding_box_range的作用是将volume中bb_min,bb_max之间的三维区域赋值为sub_volume，
# 即：将sub_volume赋值到输入的三维数组volumn中
# 例子：
# label3_roi = set_ND_volume_roi_with_bounding_box_range(label3_roi, bbox2[0], bbox2[1], pred3)
def set_ND_volume_roi_with_bounding_box_range(volume, bb_min, bb_max, sub_volume):
    """
    set a subregion to an nd image.
    """
    dim = len(bb_min)
    out = volume
    if (dim == 2):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1))] = sub_volume
    elif (dim == 3):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1))] = sub_volume
    elif (dim == 4):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1),
                   range(bb_min[3], bb_max[3] + 1))] = sub_volume
    else:
        raise ValueError("array dimension should be 2, 3 or 4")
    return out

# 将seg.nii转化为的label_array中的值进行转化,如将0,1,2,4的label值转化位0,1,2,3
# 例子：
# label_useful = convert_label(in_volume, label_convert_source=[0, 1, 2, 4], label_convert_target=[0, 1, 2, 3])
def convert_label(in_volume, label_convert_source, label_convert_target):
    """
    convert the label value in a volume
    inputs:
        in_volume: input nd volume with label set label_convert_source
        label_convert_source: a list of integers denoting input labels, e.g., [0, 1, 2, 4]
        label_convert_target: a list of integers denoting output labels, e.g.,[0, 1, 2, 3]
    outputs:
        out_volume: the output nd volume with label set label_convert_target
    """
    mask_volume = np.zeros_like(in_volume)
    convert_volume = np.zeros_like(in_volume)
    for i in range(len(label_convert_source)):
        source_lab = label_convert_source[i]
        target_lab = label_convert_target[i]
        if (source_lab != target_lab):
            temp_source = np.asarray(in_volume == source_lab)
            temp_target = target_lab * temp_source
            mask_volume = mask_volume + temp_source
            convert_volume = convert_volume + temp_target
    out_volume = in_volume * 1
    out_volume[mask_volume > 0] = convert_volume[mask_volume > 0]
    return out_volume

# 在boundingbox中得到随机采样的roi中心点
# 其中，
# input_shape: 被采样的图像三维数组的shape
# output_shape: 采样使用的三维数组shape
# sample_mode: 模式选择
# boundingbox = [mind, maxd, minh, maxh, minw, maxw]
def get_random_roi_sampling_center(input_shape, output_shape, sample_mode, bounding_box=None):
    """
    get a random coordinate representing the center of a roi for sampling
    inputs:
        input_shape: the shape of sampled volume
        output_shape: the desired roi shape
        sample_mode: 'valid': the entire roi should be inside the input volume
                     'full': only the roi centre should be inside the input volume
        bounding_box: the bounding box which the roi center should be limited to
    outputs:
        center: the output center coordinate of a roi
    """
    center = []
    for i in range(len(input_shape)):
        if (sample_mode[i] == 'full'):
            if (bounding_box):
                x0 = bounding_box[i * 2]
                x1 = bounding_box[i * 2 + 1]
            else:
                x0 = 0
                x1 = input_shape[i]
        else:
            if (bounding_box):
                x0 = bounding_box[i * 2] + int(output_shape[i] / 2)
                x1 = bounding_box[i * 2 + 1] - int(output_shape[i] / 2)
            else:
                x0 = int(output_shape[i] / 2)
                x1 = input_shape[i] - x0
        if (x1 <= x0):
            centeri = int((x0 + x1) / 2)
        else:
            centeri = random.randint(x0, x1)
        center.append(centeri)
    return center

# 对三维numpy array组成的list中的每个numpy array进行不同方向的翻转，
# 例子：
# 进行矢状位翻转
# temp_imgs = [img_array_1,img_array_2,img_array_3]
# tr_volumes1 = transpose_volumes(temp_imgs, 'sagittal')
def transpose_volumes(volumes, slice_direction):
    """
    transpose a list of volumes
    inputs:
        volumes: a list of nd volumes
        slice_direction: 'axial', 'sagittal', or 'coronal'
    outputs:
        tr_volumes: a list of transposed volumes
    """
    if (slice_direction == 'axial'):
        tr_volumes = volumes
    elif (slice_direction == 'sagittal'):
        tr_volumes = [np.transpose(x, (2, 0, 1)) for x in volumes]
    elif (slice_direction == 'coronal'):
        tr_volumes = [np.transpose(x, (1, 0, 2)) for x in volumes]
    else:
        print('undefined slice direction:', slice_direction)
        tr_volumes = volumes
    return tr_volumes

def rescale_data_volume(img_numpy, out_dim):
    """
    Resize the 3d numpy array to the dim size
    :param out_dim is the new 3d tuple
    """
    depth, height, width = img_numpy.shape
    scale = [out_dim[0] * 1.0 / depth, out_dim[1] * 1.0 / height, out_dim[2] * 1.0 / width]
    return ndimage.interpolation.zoom(img_numpy, scale, order=0)


# rescale/resize输入三维数组的shape,即缩放大小,利用ndimage.interpolation.zoom进行缩放插值
# 例子：
# input_array = np.zeros(shape=(255,255,132))
# output_array = resize_ND_volume_to_given_shape(input_array,(255,255,255))
# output_array.shape = (255,255,255)
def resize_ND_volume_to_given_shape(volume, out_shape, order=3):
    """
    resize an nd volume to a given shape
    inputs:
        volume: the input nd volume, an nd array
        out_shape: the desired output shape, a list
        order: the order of interpolation
    outputs:
        out_volume: the reized nd volume with given shape
    """
    shape0 = volume.shape
    assert (len(shape0) == len(out_shape))
    scale = [(out_shape[i] + 0.0) / shape0[i] for i in range(len(shape0))]
    out_volume = ndimage.interpolation.zoom(volume, scale, order=order)
    return out_volume

# 输入图像数组源，roi中心点index，以及期望的roi输出大小,返回一个roi三维区域
# output_roi_volume = extract_roi_from_volume(input_img_array,roi_center,(25,25,25))
def extract_roi_from_volume(volume, in_center, output_shape, fill='random'):
    """
    extract a roi from a 3d volume
    inputs:
        volume: the input 3D volume
        in_center: the center of the roi
        output_shape: the size of the roi
        fill: 'random' or 'zero', the mode to fill roi region where is outside of the input volume
    outputs:
        output: the roi volume
    """
    input_shape = volume.shape
    if (fill == 'random'):
        output = np.random.normal(0, 1, size=output_shape)
    else:
        output = np.zeros(output_shape)
    r0max = [int(x / 2) for x in output_shape]
    r1max = [output_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], in_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], input_shape[i] - in_center[i]) for i in range(len(r0max))]
    out_center = r0max

    output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                  range(out_center[1] - r0[1], out_center[1] + r1[1]),
                  range(out_center[2] - r0[2], out_center[2] + r1[2]))] = \
        volume[np.ix_(range(in_center[0] - r0[0], in_center[0] + r1[0]),
                      range(in_center[1] - r0[1], in_center[1] + r1[1]),
                      range(in_center[2] - r0[2], in_center[2] + r1[2]))]
    output = output.astype(np.uint8)
    return output

# 赋值roi区域
def set_roi_to_volume(volume, center, sub_volume):
    """
    set the content of an roi of a 3d/4d volume to a sub volume
    inputs:
        volume: the input 3D/4D volume
        center: the center of the roi
        sub_volume: the content of sub volume
    outputs:
        output_volume: the output 3D/4D volume
    """
    volume_shape = volume.shape
    patch_shape = sub_volume.shape
    output_volume = volume
    for i in range(len(center)):
        if (center[i] >= volume_shape[i]):
            return output_volume
    r0max = [int(x / 2) for x in patch_shape]
    r1max = [patch_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], volume_shape[i] - center[i]) for i in range(len(r0max))]
    patch_center = r0max

    if (len(center) == 3):
        output_volume[np.ix_(range(center[0] - r0[0], center[0] + r1[0]),
                             range(center[1] - r0[1], center[1] + r1[1]),
                             range(center[2] - r0[2], center[2] + r1[2]))] = \
            sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                              range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                              range(patch_center[2] - r0[2], patch_center[2] + r1[2]))]
    elif (len(center) == 4):
        output_volume[np.ix_(range(center[0] - r0[0], center[0] + r1[0]),
                             range(center[1] - r0[1], center[1] + r1[1]),
                             range(center[2] - r0[2], center[2] + r1[2]),
                             range(center[3] - r0[3], center[3] + r1[3]))] = \
            sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                              range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                              range(patch_center[2] - r0[2], patch_center[2] + r1[2]),
                              range(patch_center[3] - r0[3], patch_center[3] + r1[3]))]
    else:
        raise ValueError("array dimension should be 3 or 4")
    return output_volume

# 将nii的像素值转化位uint8格式的像素值
def convert_to_uint8(input_array):
    input_array = input_array / input_array.max()
    output_array = np.uint8(input_array * 255)
    return output_array

#
def normal_and_convert_to_uint8(input_array):
    input_array = (input_array - input_array.min()) / (input_array.max() - input_array.min())
    output_array = np.uint8(input_array * 255)
    return output_array


#
def get_largest_two_component(img, print_info=False, threshold=None):
    """
    Get the largest two components of a binary volume
    inputs:
        img: the input 3D volume
        threshold: a size threshold
    outputs:
        out_img: the output volume
    """
    s = ndimage.generate_binary_structure(3, 2)  # iterate structure
    labeled_array, numpatches = ndimage.label(img, s)  # labeling
    sizes = ndimage.sum(img, labeled_array, range(1, numpatches + 1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if (print_info):
        print('component size', sizes_list)
    if (len(sizes) == 1):
        out_img = img
    else:
        if (threshold):
            out_img = np.zeros_like(img)
            for temp_size in sizes_list:
                if (temp_size > threshold):
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab
                    out_img = (out_img + temp_cmp) > 0
            return out_img
        else:
            max_size1 = sizes_list[-1]
            max_size2 = sizes_list[-2]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            max_label2 = np.where(sizes == max_size2)[0] + 1
            component1 = labeled_array == max_label1
            component2 = labeled_array == max_label2
            if (max_size2 * 10 > max_size1):
                component1 = (component1 + component2) > 0
            out_img = component1
    return out_img

# 填充孔洞
def fill_holes(img):
    """
    filling small holes of a binary volume with morphological operations
    """
    neg = 1 - img
    s = ndimage.generate_binary_structure(3, 1)  # iterate structure
    labeled_array, numpatches = ndimage.label(neg, s)  # labeling
    sizes = ndimage.sum(neg, labeled_array, range(1, numpatches + 1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    max_size = sizes_list[-1]
    max_label = np.where(sizes == max_size)[0] + 1
    component = labeled_array == max_label
    return 1 - component

# 取出wt区域外的core tumor区域
def remove_external_core(lab_main, lab_ext):
    """
    remove the core region that is outside of whole tumor
    """

    # for each component of lab_ext, compute the overlap with lab_main
    s = ndimage.generate_binary_structure(3, 2)  # iterate structure
    labeled_array, numpatches = ndimage.label(lab_ext, s)  # labeling
    sizes = ndimage.sum(lab_ext, labeled_array, range(1, numpatches + 1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    new_lab_ext = np.zeros_like(lab_ext)
    for i in range(len(sizes)):
        sizei = sizes_list[i]
        labeli = np.where(sizes == sizei)[0] + 1
        componenti = labeled_array == labeli
        overlap = componenti * lab_main
        if ((overlap.sum() + 0.0) / sizei >= 0.5):
            new_lab_ext = np.maximum(new_lab_ext, componenti)
    return new_lab_ext

# 二值化交叠部分
def binary_dice3d(s, g):
    """
    dice score of 3d binary volumes
    inputs:
        s: segmentation volume
        g: ground truth volume
    outputs:
        dice: the dice score
    """
    assert (len(s.shape) == 3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    assert (Ds == Dg and Hs == Hg and Ws == Wg)
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = (2.0 * s0 + 1e-10) / (s1 + s2 + 1e-10)
    return dice

# 得到根据label numpy array中心并以targe shape为边界框的左下角坐标和右上角坐标
def  Get_Label_Array_Img_Boundary(label_array,target_shape):

    try:
        assert target_shape[0] == target_shape[1] == target_shape[2]
    except:
        print('target shape should be equal in dimention')

    # 得到label的三维边界框
    bbmin, bbmax = get_ND_bounding_box(label_array, margin=0)

    mask_centre_z, mask_centre_y, mask_centre_x = int((bbmax[0] + bbmin[0]) / 2), int((bbmax[1] + bbmin[1]) / 2), int(
        (bbmax[2] + bbmin[2]) / 2)

    origin_shape = label_array.shape
    target_shape = target_shape

    # 指定大小的三维立方体中心坐标
    # fixed_shape_centre_z,fix_shape_centre_y,fix_shape_centre_x = int(label_array.shape[0] / 2),mask_centre_y,mask_centre_x
    fixed_shape_centre = (int(label_array.shape[0] / 2), mask_centre_y, mask_centre_x)

    # 判断中心坐标是否超出图像边界
    boundary_centre_min = [int(target_shape[i] / 2) + 1 for i in range(len(target_shape))]
    boundary_centre_max = [origin_shape[i] - int(target_shape[i] / 2) - 1 for i in range(len(target_shape))]

    centre_min_diff_vector = [fixed_shape_centre[i] - boundary_centre_min[i] for i in range(len(target_shape))]
    centre_max_diff_vector = [boundary_centre_max[i] - fixed_shape_centre[i] for i in range(len(target_shape))]

    affine_min_vector = []
    for i in range(len(centre_min_diff_vector)):
        if centre_min_diff_vector[i] < 0:
            print('min boundary beyond!')
            affine_min_vector.append(-centre_min_diff_vector[i])
        else:
            affine_min_vector.append(0)
    affine_max_vector = []
    for i in range(len(centre_max_diff_vector)):
        if centre_max_diff_vector[i] < 0:
            print('max boundary beyond!')
            affine_max_vector.append(centre_max_diff_vector[i])
        else:
            affine_max_vector.append(0)

    fixed_shape_centre = [fixed_shape_centre[i] + affine_min_vector[i] + affine_max_vector[i] for i in
                          range(len(target_shape))]

    fixed_shape_centre_z, fix_shape_centre_y, fix_shape_centre_x = fixed_shape_centre

    fixed_shape_bbmin = (
    fixed_shape_centre_z - int((target_shape[0] / 2) - 1), fix_shape_centre_y - int((target_shape[1] / 2) - 1),
    fix_shape_centre_x - int((target_shape[2] / 2) - 1))

    fixed_shape_bbmax = (
    fixed_shape_centre_z + int((target_shape[0] / 2)), fix_shape_centre_y + int((target_shape[1] / 2)),
    fix_shape_centre_x + int((target_shape[2] / 2)))

    return fixed_shape_bbmin,fixed_shape_bbmax


if __name__ == '__main__':
    import cv2
    img_array = load_3d_volume_as_array('/mnt/data5/zyz/BraTS18_Val_Patient_Data/Brats18_2013_2_1/Brats18_2013_2_1_t1.nii')

    norm_img_array = itensity_normalize_one_volume(img_array)
    norm_enhanced_img_array = normalize_one_volumn_enhanced(img_array)

    origin_img_array_uint8 = convert_to_uint8(img_array)
    norm_img_array_uint8 = convert_to_uint8(norm_img_array)
    norm_enhanced_img_array = convert_to_uint8(norm_enhanced_img_array)

    cv2.imshow('origin_75',origin_img_array_uint8[60])
    cv2.waitKey()
    cv2.imshow('norm_75',norm_img_array_uint8[60])
    cv2.waitKey()
    cv2.imshow('enhanced_norm_75',norm_enhanced_img_array[60])
    cv2.waitKey()



