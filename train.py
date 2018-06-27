#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 3/22/2018 5:11 PM 
# @Author : sunyonghai 
# @File : train.py 
# @Software: ZJ_AI
# =========================================================
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import Flatten, Dense, Lambda
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from keras.preprocessing import image

import os

import losses

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set variables
gen = image.ImageDataGenerator()
batch_size = 64
epochs = 10
# import training data
batches = gen.flow_from_directory('data/train',
                                  target_size=(224,224),
                                  class_mode='categorical',
                                  shuffle=True,
                                  batch_size=batch_size)


# import validation data
val_batches = gen.flow_from_directory('data/valid',
                                      target_size=(224,224),
                                      class_mode='categorical',
                                      shuffle=True,
                                      batch_size=batch_size)

# retrieve the full Keras VGG model including imagenet weights
vgg = VGG16(include_top=True, weights='imagenet',input_tensor=None, input_shape=(224,224,3), pooling=None)


# set all layers to non-trainable
for layer in vgg.layers: layer.trainable=False

# define a new output layer to connect with the last fc layer in vgg
# thanks to joelthchao https://github.com/fchollet/keras/issues/2371
vgg.summary()
x = vgg.layers[-2].output
output_layer = Dense(2, activation='softmax', name='predictions')(x)

# combine the original VGG model with the new output layer
vgg2 = Model(inputs=vgg.input, outputs=output_layer)

vgg2.summary()

# compile the new model
vgg2.compile(optimizer=Adam(lr=0.001), loss=losses.mean_squared_error, metrics=['accuracy'])

weight_path = 'model/'+ 'weights_{epoch:02d}_{loss:.4f}.hdf5'
checkpointer = ModelCheckpoint(filepath=weight_path, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
earlyStopping = EarlyStopping(monitor='loss', patience=0, verbose=0, mode='min')
# run it!
vgg2.fit_generator(batches,
                   steps_per_epoch = batches.samples // batch_size,
                   #steps_per_epoch = 10,
                   validation_data = val_batches,
                   validation_steps = val_batches.samples // batch_size, epochs = epochs, verbose=1, callbacks=[checkpointer, earlyStopping])