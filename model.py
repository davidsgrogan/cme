#!/usr/bin/env python3

import os
# This small model is faster on the CPU.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocess
import pandas as pd
import tensorflow as tf
import keras.layers as layers
from keras.models import Sequential
import keras.optimizers as optimizers
from keras.callbacks import TensorBoard
import keras.constraints

import matplotlib.pyplot as plt
import IPython.display

import time
import random
import sys

# https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
np.random.seed(345)
tf.set_random_seed(345)
random.seed(345)

# %%

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1000])
preprocess.robust_scale(X)
preprocess.robust_scale(X, quantile_range=(10, 90))
preprocess.scale(X)

# %%
#all_data = pd.read_csv('all_data_v7.0.1.csv')
all_data = pd.read_csv('all_data_v7.0.1_min_10_AB.csv')

#%%
print("csv shape =", all_data.shape)

# I guess MinMaxScaler would be a no-op on binary columns.
binary_columns = set(["AL", "SameTeam", "bats_right", "bats_switch"])
dont_standardize = set([*binary_columns, "P_OPS"])
do_standardize = set(all_data.columns.to_list()) - dont_standardize

scaler = preprocess.StandardScaler()
all_data[list(do_standardize)] = scaler.fit_transform(all_data[do_standardize])

#%%
# 70/20/10 split.
train_set, val_test_set = train_test_split(all_data,
                                           test_size=0.3,
                                           random_state=35,
                                           shuffle=True)

val_set, test_set = train_test_split(val_test_set,
                                     test_size=0.33333333,
                                     random_state=36)

val_test_set = None


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

model = Sequential()

# Switching from LRelu to Relu made validation performance a little worse and
# training slower. PRelu improves performance but not training speed.
model.add(
    layers.Dense(60,
                 activation=None,
                 kernel_initializer='he_normal',
                 use_bias=True,
                 input_shape=(train_X.shape[1],)))
model.add(layers.BatchNormalization())
model.add(layers.PReLU())

model.add(
    layers.Dense(20,
                 activation=None,
                 kernel_initializer='he_normal',
                 use_bias=True))
model.add(layers.BatchNormalization())
model.add(layers.PReLU())

model.add(
    layers.Dense(20,
                 activation=None,
                 kernel_initializer='he_normal',
                 use_bias=True))
model.add(layers.BatchNormalization())
model.add(layers.PReLU())

# W_constraint makes val_loss goes from 0.094 to 0.092.
# That may be just a more favorable random # generation.
model.add(
    layers.Dense(1,
                 activation='linear',
                 W_constraint=keras.constraints.NonNeg()))
#model.add(layers.LeakyReLU(alpha=0.1))

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

np.random.seed(345)
tf.set_random_seed(123)
random.seed(345)

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

os.remove("model.h5")
model.save("model.h5")

# %%

# Plot training & validation loss values
omit_first = 10
plt.plot(history_object.history['mean_squared_error'][omit_first:])
plt.plot(history_object.history['val_mean_squared_error'][omit_first:])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('cnn_loss.png', bbox_inches='tight')
plt.show()

#%%
import deeplift
from deeplift.conversion import kerasapi_conversion as kc

reference = np.full(shape=(train_X.shape[1],), fill_value=0.5)

deeplift_model = kc.convert_model_from_saved_files("model.h5")
deeplift_contribs_func = deeplift_model.get_target_contribs_func(
    find_scores_layer_idx=0, target_layer_idx=-1)
#%%
scores = deeplift_contribs_func(
    task_idx=0,
    input_data_list=[train_X.iloc[0].to_numpy().reshape(1, 29)],
    input_references_list=[reference.reshape(1, 29)],
    batch_size=1,
    progress_update=1)
