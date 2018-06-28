# -*- coding: utf-8 -*-
# @Time    : 6/25/2018 2:09 PM
# @Author  : sunyonghai
# @File    : losses.py
# @Software: ZJ_AI

from keras import backend as K

def mean_squared_error(y_true, y_pred):
    ave = K.mean(K.square(y_pred - y_true), axis=-1)
    return ave