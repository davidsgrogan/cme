#!/usr/bin/env python3

import os
# This small model is faster on the CPU.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocess
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.callbacks import TensorBoard

import matplotlib.pyplot as plt
import IPython.display

import time
import random
import sys

# https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
np.random.seed(345)
tf.random.set_seed(345)
random.seed(345)

# %%

X = np.array([1,2,3,4,5,6,7,8,9,10,1000])
preprocess.robust_scale(X)
preprocess.scale(X)

# %%
#all_data = pd.read_csv('all_data_v7.0.1.csv')
all_data = pd.read_csv('all_data_v7.0.1_min_10_AB.csv')

print("csv shape =", all_data.shape)

# 70/20/10 split.
train_set, val_test_set = train_test_split(all_data,
                                           test_size=0.3,
                                           random_state=35,
                                           shuffle=True)

val_set, test_set = train_test_split(val_test_set,
                                     test_size=0.33333333,
                                     random_state=36)

val_test_set = None
#all_data = None


def split_X_y(dataframe):
  y = dataframe["P_OPS"]
  X = dataframe.drop(columns="P_OPS")
  return (X, y)


train_X, train_y = split_X_y(train_set)
val_X, val_y = split_X_y(val_set)
test_X, test_y = split_X_y(test_set)

train_set = None
val_set = None
test_set = None

print("training on %d columns" % len(train_X.columns), list(train_X.columns))

#%%

# Keras
model = Sequential()

model.add(layers.InputLayer(input_shape=train_X.shape[1]))

model.add(layers.Dense(60, activation=None, kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))

model.add(layers.Dense(20, activation=None, kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))

model.add(layers.Dense(20, activation=None, kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))

model.add(layers.Dense(1, activation=None))

adam_optimizer = optimizers.Adam(lr=0.0001)
model.compile(optimizer=adam_optimizer, loss='mse', metrics=['mse'])

# keras.utils.plot_model(model,
#                        to_file='test_keras_plot_model.png',
#                        show_shapes=True)
# display(IPython.display.Image('test_keras_plot_model.png'))
print(model.summary())
print("Just printed model summary")

#%%

start_time = time.time()

tensor_board = TensorBoard(histogram_freq=1)

history_object = model.fit(
    train_X,
    train_y,
    epochs=1000,
    batch_size=64,
    verbose=2,
    #callbacks=[tensor_board],
    shuffle=True,
    validation_data=(val_X, val_y))
print("%d seconds to train the model" % (time.time() - start_time))

#model.save("cnn_model.h5")

# %%

# Plot training & validation loss values
omit_first = 10
plt.plot(history_object.history['mse'][omit_first:])
plt.plot(history_object.history['val_mse'][omit_first:])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('cnn_loss.png', bbox_inches='tight')
plt.show()
