import nibabel as nib
import numpy as np
import torch
from PIL import Image
from nibabel.processing import resample_to_output
from scipy import ndimage
import math
import torch.nn.functional as F
import SimpleITK as sitk


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


def get_sample_area_by_img_centre(full_vol_dim, crop_size):
    """
    :param full_vol_dim: (y,x)
    :param crop_size: (y,x)
    :return:
    """
    assert full_vol_dim[0] >= crop_size[0], "crop size y is too big"
    assert full_vol_dim[1] >= crop_size[1], "crop size z is too big"


    # np.random.seed(seed)
    # 左上角最小情况
    centre_y = (full_vol_dim[0]) // 2
    centre_x = (full_vol_dim[1]) // 2

    min_width = centre_y - crop_size[0]/2
    min_height = centre_x - crop_size[1]/2



    if min_width < 0: min_width = 0
    if min_height < 0: min_height = 0

    if min_width >= full_vol_dim[0] - crop_size[0]: min_width = full_vol_dim[0] - crop_size[0]
    if min_height >= full_vol_dim[1] - crop_size[1]: min_height = full_vol_dim[1] - crop_size[1]

    return (int(min_width), int(min_height))

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

def read_nii_as_narray(nii_file_path):
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
    return patient_volume_narray, spacing

def save_narray_as_nii_file(input_narray, save_nii_file_path, spacing, origin, direction):
    savedImg = sitk.GetImageFromArray(input_narray)
    savedImg.SetOrigin(origin)
    savedImg.SetDirection(direction)
    savedImg.SetSpacing(spacing)

    print('Save:', save_nii_file_path)
    sitk.WriteImage(savedImg, save_nii_file_path)


def save_feature_narray_as_nii_file(input_narray, save_nii_file_path):
    savedImg = sitk.GetImageFromArray(input_narray)
    sitk.WriteImage(savedImg, save_nii_file_path)

"""
concentrate all pre-processing here here
"""
def crop_cube_from_volumn(origin_volumn,crop_point, crop_size):
    """

    :param origin_volumn: (z,y,x)
    :param crop_point: (current_crop_z, current_crop_y, current_crop_x )
    :param crop_size: (crop_z,crop_y,crop_x)
    :return:
    """
    current_crop_z, current_crop_y, current_crop_x = crop_point

    cube = origin_volumn[current_crop_z:current_crop_z + crop_size[0], current_crop_y:current_crop_y + crop_size[1], current_crop_x:current_crop_x + crop_size[2]]

    return cube

def load_medical_image_full(path,type=None, normalization='full_volume_mean', clip_intenisty=True):
    if type not in ['label','feature']:
        assert False
    img_nii = nib.load(path)

    img_np = np.squeeze(img_nii.get_fdata(dtype=np.float32))

    # transpose,如果转置之后crop_dim也要换
    img_np = np.transpose(img_np, [2, 1, 0])

    # 1. Intensity outlier clipping
    if clip_intenisty and type != "label":
        img_np = percentile_clip(img_np)


    # 3. intensity normalization
    img_tensor = torch.from_numpy(img_np)

    MEAN, STD, MAX, MIN = 0., 1., 1., 0.
    if type != 'label':
        MEAN, STD = img_tensor.mean(), img_tensor.std()
        MAX, MIN = img_tensor.max(), img_tensor.min()

    if type != "label":
        img_tensor = normalize_intensity(img_tensor, normalization=normalization, norm_values=(MEAN, STD, MAX, MIN))

    return img_tensor


def load_medical_image(path,type='label', crop=(0, 0, 0), crop_size=(0, 0, 0),
                       resample=None, rescale=None, normalization='full_volume_mean',
                       clip_intenisty=False,  padding=False, isOrigin=False):
    """

    :param path: nii 路径
    :param type: 类型，feature, label
    :param crop:
    :param crop_size:
    :param resample:
    :param rescale:
    :param normalization:
    :param clip_intenisty:
    :param padding:
    :return: (z, y, x)
    """
    if type not in ['label','feature']:
        assert False
    img_nii = nib.load(path)

    # 重采样输入应该是nii.afine
    if resample is not None:
        img_nii = resample_to_output(img_nii, voxel_sizes=resample)

    img_np = np.squeeze(img_nii.get_fdata(dtype=np.float32))

    # transpose
    img_np = np.transpose(img_np, [2, 1, 0])

    if isOrigin:
        return torch.from_numpy(img_np)

    # 1. Intensity outlier clipping
    if clip_intenisty and type != "label":
        img_np = percentile_clip(img_np)

    # 2. Rescale to specified output shape
    if rescale is not None:
        rescale_data_volume(img_np, rescale)

    # 3. intensity normalization
    img_tensor = torch.from_numpy(img_np)
    # img_tensor = img_np
    MEAN, STD, MAX, MIN = 0., 1., 1., 0.
    if type != 'label':
        MEAN, STD = img_tensor.mean(), img_tensor.std()
        MAX, MIN = img_tensor.max(), img_tensor.min()

    if padding:
        img_tensor = pad_medical_image(img_tensor, kernel_dim=crop_size)

    if type != "label":
        img_tensor = normalize_intensity(img_tensor, normalization=normalization, norm_values=(MEAN, STD, MAX, MIN))

    img_tensor = crop_img(img_tensor, crop_size, crop)
    return img_tensor


def roundup(x, base=32):
    return int(math.ceil(x / base)) * base


def pad_medical_image(img_tensor, kernel_dim=(32, 32, 32)):
    _, D, H, W = img_tensor.shape
    kc, kh, kw = kernel_dim
    dc, dh, dw = 4, 4, 4

    # stride
    # Pad to multiples of kernel_dim
    a = ((roundup(W, kw) - W) // 2 + W % 2, (roundup(W, kw) - W) // 2,
         (roundup(H, kh) - H) // 2 + H % 2, (roundup(H, kh) - H) // 2,
         (roundup(D, kc) - D) // 2 + D % 2, (roundup(D, kc) - D) // 2)
    x = F.pad(img_tensor, a, value=img_tensor[0, 0, 0, 0])

    return x, a


def medical_image_transform(img_tensor, type=None,
                            normalization="full_volume_mean",
                            norm_values=(0., 1., 1., 0.)):
    MEAN, STD, MAX, MIN = norm_values
    # Numpy-based transformations/augmentations here

    if type != 'label':
        MEAN, STD = img_tensor.mean(), img_tensor.std()
        MAX, MIN = img_tensor.max(), img_tensor.min()

    if type != "label":
        img_tensor = normalize_intensity(img_tensor, normalization=normalization, norm_values=(MEAN, STD, MAX, MIN))

    return img_tensor


def crop_img(img_tensor, crop_size, crop):
    if crop_size[0] == 0:
        return img_tensor
    slices_crop, w_crop, h_crop = crop
    dim1, dim2, dim3 = crop_size
    inp_img_dim = img_tensor.dim()
    assert inp_img_dim >= 3
    if img_tensor.dim() == 3:
        full_dim1, full_dim2, full_dim3 = img_tensor.shape
    elif img_tensor.dim() == 4:
        _, full_dim1, full_dim2, full_dim3 = img_tensor.shape
        img_tensor = img_tensor[0, ...]

    if full_dim1 == dim1:
        img_tensor = img_tensor[:, w_crop:w_crop + dim2,
                     h_crop:h_crop + dim3]
    elif full_dim2 == dim2:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, :,
                     h_crop:h_crop + dim3]
    elif full_dim3 == dim3:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, w_crop:w_crop + dim2, :]
    else:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, w_crop:w_crop + dim2,
                     h_crop:h_crop + dim3]

    if inp_img_dim == 4:
        return img_tensor.unsqueeze(0)

    return img_tensor


def load_affine_matrix(path):
    """
    Reads an path to nifti file and returns the affine matrix as numpy array 4x4
    """
    img = nib.load(path)
    return img.affine


def load_2d_image(img_path, resize_dim=0, type='RGB'):
    image = Image.open(img_path)
    if type == 'RGB':
        image = image.convert(type)
    if resize_dim != 0:
        image = image.resize(resize_dim)
    pix = np.array(image)
    return pix


def rescale_data_volume(img_numpy, out_dim):
    """
    Resize the 3d numpy array to the dim size
    :param out_dim is the new 3d tuple
    """
    depth, height, width = img_numpy.shape
    scale = [out_dim[0] * 1.0 / depth, out_dim[1] * 1.0 / height, out_dim[2] * 1.0 / width]
    return ndimage.interpolation.zoom(img_numpy, scale, order=0)


def transform_coordinate_space(modality_1, modality_2):
    """
    Accepts nifty objects
    Transfers coordinate space from modality_2 to modality_1
    """
    aff_t1 = modality_1.affine
    aff_t2 = modality_2.affine
    inv_af_2 = np.linalg.inv(aff_t2)

    out_shape = modality_1.get_fdata().shape

    # desired transformation
    T = inv_af_2.dot(aff_t1)
    transformed = ndimage.affine_transform(modality_2.get_fdata(), T, output_shape=out_shape)

    return transformed


def normalize_intensity(img_tensor, normalization="full_volume_mean", norm_values=(0, 1, 1, 0)):
    """
    Accepts an image tensor and normalizes it
    :param normalization: choices = "max", "mean" , type=str
    """
    if normalization == "mean":
        mask = img_tensor.ne(0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / std_val
    elif normalization == "max":
        max_val, _ = torch.max(img_tensor)
        img_tensor = img_tensor / max_val
    elif normalization == 'brats':
        # print(norm_values)
        normalized_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]
        final_tensor = torch.where(img_tensor == 0., img_tensor, normalized_tensor)
        final_tensor = 100.0 * ((final_tensor.clone() - norm_values[3]) / (norm_values[2] - norm_values[3])) + 10.0
        x = torch.where(img_tensor == 0., img_tensor, final_tensor)
        return x

    elif normalization == 'full_volume_mean':
        img_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]

    elif normalization == 'max_min':
        img_tensor = (img_tensor - norm_values[3]) / ((norm_values[2] - norm_values[3]))

    elif normalization == 'non_normal':
        img_tensor = img_tensor
    return img_tensor


## todo percentiles

def clip_range(img_numpy):
    """
    Cut off outliers that are related to detected black in the image (the air area)
    """
    # Todo median value!
    zero_value = (img_numpy[0, 0, 0] + img_numpy[-1, 0, 0] + img_numpy[0, -1, 0] + \
                  img_numpy[0, 0, -1] + img_numpy[-1, -1, -1] + img_numpy[-1, -1, 0] \
                  + img_numpy[0, -1, -1] + img_numpy[-1, 0, -1]) / 8.0
    non_zeros_idx = np.where(img_numpy >= zero_value)
    [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
    [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
    y = img_numpy[min_z:max_z, min_h:max_h, min_w:max_w]
    return y


def percentile_clip(img_numpy, min_val=0.1, max_val=99.8):
    """
    Intensity normalization based on percentile
    Clips the range based on the quarile values.
    :param min_val: should be in the range [0,100]
    :param max_val: should be in the range [0,100]
    :return: intesity normalized image
    """
    low = np.percentile(img_numpy, min_val)
    high = np.percentile(img_numpy, max_val)

    img_numpy[img_numpy < low] = low
    img_numpy[img_numpy > high] = high
    return img_numpy
