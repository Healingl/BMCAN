import cv2

from matplotlib import cm, pyplot as plt
import numpy as np


def get_cmap(cmname):
    colormap_int = np.zeros((256, 3), np.uint8)
    colormap_float = np.zeros((256, 3), np.float)
    cmap = plt.get_cmap(cmname)
    for i in range(0, 256, 1):
        colormap_float[i, 0] = cmap(i)[0]
        colormap_float[i, 1] = cmap(i)[1]
        colormap_float[i, 2] = cmap(i)[2]

        colormap_int[i, 0] = np.int_(np.round(cmap(i)[0] * 255.0))
        colormap_int[i, 1] = np.int_(np.round(cmap(i)[1] * 255.0))
        colormap_int[i, 2] = np.int_(np.round(cmap(i)[2] * 255.0))

    # np.savetxt("jet_float.txt", colormap_float, fmt="%f", delimiter=' ', newline='\n')
    # np.savetxt("jet_int.txt", colormap_int, fmt="%d", delimiter=' ', newline='\n')

    # print(colormap_int)

    return colormap_int


def gray2color(gray_array, color_map):
    '''

    :param gray_array: 必须为二维numpy_narray矩阵
    :param color_map: 保存的cmp彩色映射图
    :return: 返回映射后的热力图(heat_map)
    '''
    rows, cols = gray_array.shape
    color_array = np.zeros((rows, cols, 3), np.uint8)

    for i in range(0, rows):
        for j in range(0, cols):
            color_array[i, j] = color_map[gray_array[i, j]]

    # color_image = Image.fromarray(color_array)

    return color_array


def draw_contours(images, labels, num_class, colors, thickness=3, linetype=None):
    contour_images = []
    for image, label in zip(images, labels):
        # print(image)
        for i in range(num_class):
            cur_label = (label == i).astype(np.uint8)
            cur_label, contours, hierarchy = cv2.findContours(cur_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE);
            # print(len(contours))
            cv2.drawContours(image, contours, -1, colors[i], thickness, linetype)
        contour_images.append(image)
    return np.stack(contour_images, axis=0)


def decode_segmap(img, n_classes, label_colours):
    map = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3), dtype=np.uint8)
    for idx in range(img.shape[0]):
        temp = img[idx, :, :]
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, n_classes):
            b[temp == l] = label_colours[l][0]
            g[temp == l] = label_colours[l][1]
            r[temp == l] = label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = b
        rgb[:, :, 1] = g
        rgb[:, :, 2] = r
        map[idx, :, :, :] = rgb.astype(np.uint8)
    return map


def draw_masks(images, labels, num_class, colors, alpha=0.7):
    masked_images = []
    masks = decode_segmap(labels, num_class, colors)
    for image, mask in zip(images, masks):
        image = cv2.addWeighted(image, alpha, mask, 1 - alpha, 0)
        masked_images.append(image)
    return np.stack(masked_images, axis=0)
