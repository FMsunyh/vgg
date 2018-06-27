# -*- coding: utf-8 -*-
# @Time    : 6/25/2018 2:22 PM
# @Author  : sunyonghai
# @File    : calu.py
# @Software: ZJ_AI

import tensorflow as tf
from keras import backend as K

y_pred = tf.constant(shape=(3,2), value=[0.4,0.6,0.9,0.1,0,1])
# y_pred = tf.constant(shape=(3,2), value=[(0.4,0.6),(0.9,0.1),(0,1)])
y_true = tf.constant(shape=(3,2), value=[(0,1),(1,0),(0,1)],dtype='float32')
sq = K.square(y_pred - y_true)
mean =  K.mean(sq, axis=-1)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(y_pred))
    print(sess.run(y_true))
    print(sess.run(sq))
    print(sess.run(mean))
    print('--')