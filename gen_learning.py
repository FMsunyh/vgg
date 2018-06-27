# -*- coding: utf-8 -*-
# @Time    : 6/27/2018 3:12 PM
# @Author  : sunyonghai
# @File    : gen_learning.py
# @Software: ZJ_AI

import os
from PIL import Image
import numpy as np
from keras.preprocessing import image

gen = image.ImageDataGenerator(horizontal_flip=True)
batch_size = 2
# import training data
Iter = gen.flow_from_directory('data/smaple',
                                  target_size=(224,224),
                                  class_mode='categorical',
                                  shuffle=True,
                                  save_to_dir = 'data/gen_train',
                                  batch_size=batch_size)

if __name__ == '__main__':
    x, y = Iter.next()
    print(x.shape, y.shape)
    print(x.dtype, y.dtype)
    print(y)

