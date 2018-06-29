# -*- coding: utf-8 -*-
# @Time    : 6/27/2018 11:44 AM
# @Author  : sunyonghai
# @File    : train.py
# @Software: ZJ_AI
# =========================================================
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam

import os

from vgg_v3 import generator, losses

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set variables
train_data_path = '../data/train'
valid_data_path = '../data/valid'

batch_size = 64
epochs = 10
train_iter = generator.Generator(train_data_path, batch_size)
valid_iter = generator.Generator(valid_data_path, batch_size)

vgg = VGG16(include_top=True, weights='imagenet',input_tensor=None, input_shape=(224,224,3), pooling=None)
for layer in vgg.layers: layer.trainable=False
vgg.summary()
x = vgg.layers[-2].output
output_layer = Dense(2, activation='softmax', name='predictions')(x)
vgg2 = Model(inputs=vgg.input, outputs=output_layer)
vgg2.summary()
vgg2.compile(optimizer=Adam(lr=0.001), loss=losses.categorical_crossentropy, metrics=['accuracy'])
weight_path = '../model/'+ 'weights_{epoch:02d}_{loss:.4f}.hdf5'
checkpointer = ModelCheckpoint(filepath=weight_path, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
earlyStopping = EarlyStopping(monitor='loss', patience=0, verbose=0, mode='min')

# run it!
vgg2.fit_generator(train_iter, steps_per_epoch = train_iter.size()//batch_size,
                   validation_data = valid_iter, validation_steps = valid_iter.size() // batch_size,
                   epochs = epochs, verbose=1, callbacks=[checkpointer, earlyStopping])