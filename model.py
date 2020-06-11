#!/usr/bin/env python3

import os
# This small model is faster on the CPU.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocess
import pandas as pd

# deeplift requires TF 1.x and standalone Keras, not tf.keras.
import tensorflow as tf
import keras.layers as layers
from keras.models import Sequential
import keras.optimizers as optimizers
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import keras.constraints

import matplotlib.pyplot as plt
import IPython.display

import time
import random

# https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
np.random.seed(345)
tf.set_random_seed(345)
random.seed(345)

# %%

# X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1000])
# preprocess.robust_scale(X)
# preprocess.robust_scale(X, quantile_range=(10, 90))
# preprocess.scale(X)

# %%
#all_data = pd.read_csv('all_data_v7.0.1.csv')
all_data = pd.read_csv('all_data_v7.0.1_min_10_AB.csv')

#%%
print("csv shape =", all_data.shape)

binary_columns = set(["AL", "SameTeam", "bats_right", "bats_switch"])
dont_standardize = set([*binary_columns, "P_OPS"])
do_standardize = set(all_data.columns.to_list()) - dont_standardize

scaler = preprocess.QuantileTransformer(output_distribution='normal')
all_data[list(do_standardize)] = scaler.fit_transform(all_data[do_standardize])

no_hand_columns = all_data.loc[:, ~all_data.columns.str.startswith('Hand')]
only_hand_columns = all_data.loc[:, all_data.columns.str.startswith('Hand') | all_data.columns.str.startswith('P_OPS')]


# It's lame to manually select these by commenting code.
all_data = only_hand_columns
#all_data = no_hand_columns

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

test_y = test_y.to_numpy()

train_set = None
val_set = None
test_set = None

print("training on %d columns" % len(train_X.columns), list(train_X.columns))

#%%

np.random.seed(345)
tf.set_random_seed(123)
random.seed(345)

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

model.add(
    layers.Dense(1,
                 activation=None
    #             W_constraint=keras.constraints.NonNeg()
))
#model.add(layers.PReLU())

adam_optimizer = optimizers.Adam(lr=0.0002)
model.compile(optimizer=adam_optimizer, loss='mse', metrics=['mse'])

keras.utils.plot_model(model,
                      to_file='model.png',
                      show_shapes=True)
#display(IPython.display.Image('test_keras_plot_model.png'))
#print(model.summary())
#print("Just printed model summary")

tensor_board = TensorBoard(histogram_freq=1)
mc = ModelCheckpoint("model.h5", monitor='val_mean_squared_error', mode='min', verbose=1, save_best_only=True)

np.random.seed(3455)
tf.set_random_seed(1235)
random.seed(3455)
#%%

start_time = time.time()
history_object = model.fit(
    train_X,
    train_y,
    epochs=180,
    batch_size=64,
    verbose=2,
    callbacks=[mc],
    shuffle=True,
    validation_data=(val_X, val_y))
print("%d seconds to train the model" % (time.time() - start_time))

# %%

# Plot training & validation loss values
omit_first = 10
plt.plot(history_object.history['mean_squared_error'][omit_first:])
plt.plot(history_object.history['val_mean_squared_error'][omit_first:])
plt.ylabel('MSE')
plt.xlabel('Epoch - %d' % omit_first)
plt.legend(['Train', 'Validation'], loc='upper right')
plt.title("Only hand features included")
plt.savefig('cnn_loss.png', bbox_inches='tight')
plt.show()

#%%
from deeplift.conversion import kerasapi_conversion as kc
deeplift_model = kc.convert_model_from_saved_files("model.h5")
deeplift_contribs_func = deeplift_model.get_target_contribs_func(
    find_scores_layer_idx=0, target_layer_idx=-1)
#%%

#reference = np.mean(train_X)
reference = np.min(train_X)
reference = reference.to_numpy()

#reference = np.full(shape=(train_X.shape[1],), fill_value=0)

scores = deeplift_contribs_func(
    task_idx=0, # N/A for my scalar output.
    input_data_list=[train_X],
    input_references_list=[reference.reshape(1, reference.shape[0])],
    batch_size=500,
    progress_update=1)

variable_to_average_score = dict(zip(train_X.columns.tolist(), np.mean(scores, axis=0)))
print(variable_to_average_score)

# {'Hand_OBP': 0.5315988, 'Hand_SLG': 0.13289687, 'Hand_BABIP': -0.056552917,
# 'Hand_ISO': -0.3444691, 'Hand_SBPerG': 0.17026354}

#%%
plt.bar(train_X.columns.tolist(), np.mean(scores, axis=0))
#plt.legend(['Train', 'Validation'], loc='upper right')
fig = plt.gcf()
fig.set_size_inches(6,5)
plt.savefig('bar_graph.png', bbox_inches='tight')
plt.show()

#%% test set
pred_y = model.predict(test_X)
test_y = test_y.reshape(test_y.shape[0], 1)
test_mse = np.mean((pred_y - test_y)**2)