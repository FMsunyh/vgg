# -*- coding: utf-8 -*-
# @Time    : 6/25/2018 2:09 PM
# @Author  : sunyonghai
# @File    : losses.py
# @Software: ZJ_AI

from keras import backend as K
import numpy as np

_EPSILON = 1e-7
def epsilon():
    return _EPSILON

# y_true = np.zeros((2,2))
y_true = np.array([[1.,0],[0,1.0]])
y_pred = np.array([[0.6,0.4],[0.2,0.8]])

def mean_squared_error(y_true, y_pred):
    ave = np.mean(np.square(y_pred - y_true), axis=-1)
    return ave

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true), axis=-1)

def mean_absolute_percentage_error(y_true, y_pred):
   diff = np.abs( (y_pred - y_true) / np.clip(np.abs(y_true), epsilon(), None) )

   return np.mean(diff, axis=-1) * 100.0

def mean_squared_logarithmic_error(y_true,y_pred):
    first_log = np.log(np.clip(y_pred,epsilon(),None) + 1)
    second_log = np.log(np.clip(y_true,epsilon(),None) + 1)

    return np.mean(np.square(first_log - second_log), axis=-1)

def squared_hinge(y_true, y_pred):
    return  np.mean(np.square(np.maximum(1.0 - y_true*y_pred, 0.0)),axis=-1)

def hinge(y_true, y_pred):
    return np.mean(np.maximum(1.0 - y_true * y_pred, 0.0), axis=-1)

def categorical_hinge(y_true, y_pred):
    pos = np.sum(y_true * y_pred, axis=-1)
    neg = np.max((1. - y_true) * y_pred, axis=-1)
    return np.maximum(0., neg - pos + 1.)

def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def sparse_categorical_crossentropy(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_true, y_pred)

def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

if __name__ == '__main__':
    print(mean_squared_error(y_true,y_pred))
    print(mean_absolute_error(y_true,y_pred))
    print(mean_absolute_percentage_error(y_true,y_pred))
    print(mean_squared_logarithmic_error(y_true,y_pred))
    print(squared_hinge(y_true,y_pred))
    print(hinge(y_true,y_pred))
