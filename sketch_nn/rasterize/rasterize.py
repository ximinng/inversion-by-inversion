# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import numpy as np
import scipy.ndimage

from .bresenham import bresenham_algo


def get_stroke_num(vector_image):
    return len(np.split(vector_image[:, :2], np.where(vector_image[:, 2])[0] + 1, axis=0)[:-1])


def select_strokes(vector_image, strokes):
    """
    select strokes
    Args:
        vector_image: vector_image(x,y,p) coordinate array
        strokes: after keeping only selected strokes

    Returns:

    """
    c = vector_image
    c_split = np.split(c[:, :2], np.where(c[:, 2])[0] + 1, axis=0)[:-1]

    c_selected = []
    for i in strokes:
        c_selected.append(c_split[i])

    xyp = []
    for i in c_selected:
        p = np.zeros((len(i), 1))
        p[-1] = 1
        xyp.append(np.hstack((i, p)))
    xyp = np.concatenate(xyp)
    return xyp


def batch_points2png(vector_images, Side=256):
    for vector_image in vector_images:
        pixel_length = 0
        # number_of_samples = random
        sample_freq = list(np.round(np.linspace(0, len(vector_image), 18)[1:]))
        Sample_len = []
        raster_images = []
        raster_image = np.zeros((int(Side), int(Side)), dtype=np.float32)
        initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
        for i in range(0, len(vector_image)):
            if i > 0:
                if vector_image[i - 1, 2] == 1:
                    initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

            cordList = list(bresenham_algo(initX, initY, int(vector_image[i, 0]), int(vector_image[i, 1])))
            pixel_length += len(cordList)

            for cord in cordList:
                if (cord[0] > 0 and cord[1] > 0) and (cord[0] < Side and cord[1] < Side):
                    raster_image[cord[1], cord[0]] = 255.0
            initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

            if i in sample_freq:
                raster_images.append(scipy.ndimage.binary_dilation(raster_image, iterations=2) * 255.0)
                Sample_len.append(pixel_length)

        raster_images.append(scipy.ndimage.binary_dilation(raster_image, iterations=3) * 255.0)
        Sample_len.append(pixel_length)

    return raster_images


def points2png(vector_image, Side=256):
    raster_image = np.zeros((int(Side), int(Side)), dtype=np.float32)
    initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
    pixel_length = 0

    for i in range(0, len(vector_image)):
        if i > 0:
            if vector_image[i - 1, 2] == 1:
                initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

        cordList = list(bresenham_algo(initX, initY, int(vector_image[i, 0]), int(vector_image[i, 1])))
        pixel_length += len(cordList)

        for cord in cordList:
            if (cord[0] > 0 and cord[1] > 0) and (cord[0] < Side and cord[1] < Side):
                raster_image[cord[1], cord[0]] = 255.0
        initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

    raster_image = scipy.ndimage.binary_dilation(raster_image) * 255.0
    return raster_image


def preprocess(sketch_points, side=256.0):
    sketch_points = sketch_points.astype(np.float)
    sketch_points[:, :2] = sketch_points[:, :2] / np.array([256, 256])
    sketch_points[:, :2] = sketch_points[:, :2] * side
    sketch_points = np.round(sketch_points)
    return sketch_points


def sketch_vector_rasterize(sketch_points):
    sketch_points = preprocess(sketch_points)
    raster_images = points2png(sketch_points)
    return raster_images


def convert_to_red(image):
    l = image.shape[1]
    image[1] = np.zeros((l, l))
    image[2] = np.zeros((l, l))
    return image


def convert_to_green(image):
    l = image.shape[1]
    image[0] = np.zeros((l, l))
    image[2] = np.zeros((l, l))
    return image


def convert_to_blue(image):
    l = image.shape[1]
    image[0] = np.zeros((l, l))
    image[1] = np.zeros((l, l))
    return image


def convert_to_black(image):
    l = image.shape[1]
    image[0] = np.zeros((l, l))
    image[1] = np.zeros((l, l))
    image[2] = np.zeros((l, l))
    return image
