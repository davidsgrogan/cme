#!/usr/bin/env python3

import os
# This small model is faster on the CPU.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from sklearn.model_selection import train_test_split
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
random.set_seed(345)

# %%
all_data = pd.read_csv('all_data_v7.0.1.csv')
assert len(all_data.index) == 5295, len(all_data.index)

print("csv shape =", all_data.shape)

num_train_samples = 3840  # 30 * 128
num_val_samples = 1055
num_test_samples = 400

assert ((num_train_samples + num_val_samples + num_test_samples) == len(
    all_data.index)), len(all_data.index)

train_set, val_test_set = train_test_split(all_data,
                                           test_size=num_val_samples +
                                           num_test_samples,
                                           random_state=35,
                                           shuffle=True)

val_set, test_set = train_test_split(val_test_set,
                                     test_size=num_test_samples,
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
    epochs=300,
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
