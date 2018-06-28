# -*- coding: utf-8 -*-
# @Time    : 6/27/2018 6:12 PM
# @Author  : sunyonghai
# @File    : utils_image.py
# @Software: ZJ_AI
import keras
import numpy as np
from PIL import Image

def read_image_bgr(path):
    try:
        image = np.asarray(Image.open(path).convert('RGB'))
    except Exception as ex:
        print(ex)

    return image[:, :, ::-1].copy()

def rezise_image(image, target_size):
    if image is not None:
        im = Image.fromarray(image)
        im = im.resize(target_size)
        im_array = np.asarray(im)
        return im_array # (224,224,3)

def preprocess_image(image):
    image = image.astype(keras.backend.floatx())
    if keras.backend.image_data_format() == 'channels_first':
        if image.ndim == 3:
            image[0, :, :] -= 103.939
            image[1, :, :] -= 116.779
            image[2, :, :] -= 123.68
        else:
            image[:, 0, :, :] -= 103.939
            image[:, 1, :, :] -= 116.779
            image[:, 2, :, :] -= 123.68
    else:
        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68

    return image