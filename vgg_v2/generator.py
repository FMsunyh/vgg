# -*- coding: utf-8 -*-
# @Time    : 6/25/2018 4:23 PM
# @Author  : sunyonghai
# @File    : generator.py
# @Software: ZJ_AI
import random
import threading

import keras
from keras.preprocessing import image

from PIL import Image
import numpy as np

import os

import utils_image

data_path = '../data/train/'

class Generator(object):
    def  __init__(self, data_path, batch_size = 6, target_size=(224, 224)):
        self.image_instances = []
        self.target_size = target_size
        self.batch_size = batch_size
        self.group_index = 0
        self.lock = threading.Lock()
        self.groups = []

        self.init(data_path)
        self.group_images()

    def init(self, data_path):
        for subdir in os.listdir(data_path):
            sub_path = os.path.join(data_path, subdir)
            for file in os.listdir(sub_path):
                self.image_instances.append(os.path.join(sub_path, file))

    def size(self):
        return len(self.image_instances)

    def resize_image(self, image):
        return utils_image.rezise_image(image, self.target_size)

    def preprocess_image(self,image):
        return utils_image.preprocess_image(image)

    def random_transform_group_entry(self,image):
        return image

    def preprocess_group_entry(self, image):
        # resize image
        image = self.resize_image(image)

        # preprocess the image
        image = self.preprocess_image(image)

        # randomly transform image
        image = self.random_transform_group_entry(image)

        return image

    def preprocess_group(self, image_group):
        for index, image in enumerate(image_group):

            # preprocess a single group entry
            image = self.preprocess_group_entry(image)

            # copy processed data back to group
            image_group[index] = image

        return image_group

    def group_images(self):
        order = list(range(self.size()))
        random.shuffle(order)

        for i in range(0, len(order), self.batch_size):
            groups = []
            for x in range(i, i + self.batch_size):
                groups.append(order[x % len(order)])  # 防止最后一个batch_size 不够数， x也不能越界

            self.groups.append(groups)

    def load_image(self, image_index):
        path = self.image_instances[image_index]
        return utils_image.read_image_bgr(path)     # RGB => BGR tensorflow train format.

    def load_image_group(self,group):
        return [self.load_image(image_index) for image_index in group]

    def load_label(self,image_index):
        path = self.image_instances[image_index]

        name = os.path.basename(path)
        if 'cat' in name:
            label = np.asarray([0, 1])
        else:
            label = np.asarray([1, 0])

        return label

    def load_label_group(self, group):
        return [self.load_label(image_index) for image_index in group]

    def compute(self, path):
        x = self.load_image(path)
        x = self.rezise_image(x)

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

    def compute_inputs(self, image_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    def compute_targets(self,label_group):
        a = label_group[0].shape
        label_batch = np.zeros((self.batch_size,) + (label_group[0].shape[0],), dtype=keras.backend.floatx())
        for index, label in enumerate(label_group):
            label_batch[index, ...] = label

        return label_batch

    def compute_input_output(self, group):
        # load images and annotations
        image_group  = self.load_image_group(group)
        label_group =  self.load_label_group(group)

        image_group = self.preprocess_group(image_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(label_group)

        return inputs, targets

    def __len__(self):
        return len(self.image_instances)

    def __next__(self):
        return self.next()

    def next(self):
        # advance the group index
        with self.lock:
            if self.group_index == 0:
                # shuffle groups at start of epoch
                random.shuffle(self.groups)
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        return self.compute_input_output(group)

if __name__ == '__main__':
    gen = Generator(data_path)
    x, y = gen.next()
    print(x.shape, y.shape)
    print(y)
