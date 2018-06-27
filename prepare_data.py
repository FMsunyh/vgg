#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 3/22/2018 5:46 PM 
# @Author : sunyonghai 
# @File : prepare_data.py 
# @Software: ZJ_AI
# =========================================================
import os
from random import randint
import os.path

import shutil

data_path = 'data/'
rawdir = data_path +'raw/train/'

cats_train_path = data_path + 'train/cats/'
cats_valid_path = data_path + 'valid/cats/'
os.makedirs(cats_train_path)
os.makedirs(cats_valid_path)

dogs_train_path = data_path + 'train/dogs/'
dogs_valid_path = data_path + 'valid/dogs/'
os.makedirs(dogs_train_path)
os.makedirs(dogs_valid_path)

for fn in os.listdir(rawdir):
    if ("jpg" in fn):
        if ("dog" in fn):
            # os.rename(rawdir + fn, data_path+ 'train/dogs/' + fn)
            shutil.copy(rawdir + fn, data_path+ 'train/dogs/' + fn)
            #print (fn)
        if ("cat" in fn):
            # os.rename(rawdir + fn, data_path+ 'train/cats/' + fn)
            shutil.copy(rawdir + fn, data_path+ 'train/cats/' + fn)

            #print (fn)

for i in range(1000):
    # dogs
    while True:
        randy = randint(0, 12499)
        fname = dogs_train_path +'dog.' + str(randy) + '.jpg'
        if os.path.isfile(fname):
            os.rename(fname, dogs_valid_path + 'dog.' + str(randy) + '.jpg')
            break

    # cats
    while True:
        randy = randint(0, 12499)
        fname = cats_train_path + 'cat.' + str(randy) + '.jpg'
        if os.path.isfile(fname):
            os.rename(fname, cats_valid_path + 'cat.' + str(randy) + '.jpg')
            break