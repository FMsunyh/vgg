# -*- coding: utf-8 -*-
# @Time    : 6/25/2018 4:23 PM
# @Author  : sunyonghai
# @File    : generator.py
# @Software: ZJ_AI
import random

import keras
from keras.preprocessing import image

from PIL import Image
import numpy as np

import os
data_path = 'data/train/'

class Generator(object):
    def  __init__(self,data_path,target_size=(224,224)):
        self.images_path = []
        self.init(data_path)
        self.target_size = target_size
        self.index = 0

    def init(self, data_path):
        for subdir in os.listdir(data_path):
            sub_path = os.path.join(data_path, subdir)
            for file in os.listdir(sub_path):
                self.images_path.append(os.path.join(sub_path, file))

    def load_image(self, path):
        try:
            im = np.asarray(Image.open(path).convert('RGB').resize(self.target_size))
        except Exception as ex:
            print(ex)

        return im[:, :, ::-1].copy() # RGB => BGR tensorflow 训练的格式

    def compute(self, path):
        x = self.load_image(path)
        name =  os.path.basename(path)
        if 'cat' in name:
            y = np.asarray([0, 1])
        else:
            y = np.asarray([1,0])

        # get the max image shape
        # max_shape = tuple( max(x.shape[i]) for i in range(3) )

        # construct an image batch object
        image_batch = np.zeros((1,) + (224,224,3), dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        image_batch[0, :x.shape[0], :x.shape[1], :x.shape[2]] = x

        y_batch = np.zeros((1,) + (2,), dtype=keras.backend.floatx())
        y_batch[0, :y.shape[0]] = y

        return image_batch, y_batch

    def __len__(self):
        return len(self.images_path)

    def __next__(self):
        return self.next()

    def next(self):
        if self.index == 0:
            random.shuffle(self.images_path)

        if self.index >= self.__len__():
            self.index = 0
        else:
            self.index += 1

        return  self.compute(self.images_path[self.index])

if __name__ == '__main__':
    gen = Generator(data_path)
    # print(gen.__len__())
    # while True:

    x, y = gen.next()
    print(x.shape, y.shape)
    print(y)
