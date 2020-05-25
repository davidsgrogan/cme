#!/usr/bin/env python3
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import IPython.display

import time
import random
import os
import sys

start_time = time.time()

# %%

all_data = pd.read_csv('all_data_v7.0.1.csv')
assert len(all_data.index) == 5295, len(all_data.index)

print("csv shape =", all_data.shape)

num_train_samples = 3840  # 30 * 128
num_val_samples = 1055
num_test_samples = 400

assert ((num_train_samples + num_val_samples + num_test_samples) == len(
    all_data.index)), len(all_data.index)

train_set = all_data.sample()

X_train, X_val_test = train_test_split(all_data,
                                       test_size=num_val_samples +
                                       num_test_samples,
                                       random_state=35,
                                       shuffle=True)
#%%

# Keras
model = Sequential()
model.add(
    layers.Conv1D(filters=40,
                  kernel_size=3,
                  strides=2,
                  activation=None,
                  use_bias=False,
                  kernel_initializer='he_normal',
                  input_shape=(X_train.shape[1], 1)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))

model.add(
    layers.Conv1D(filters=50,
                  kernel_size=3,
                  strides=2,
                  activation=None,
                  use_bias=False,
                  kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))

model.add(
    layers.Conv1D(filters=60,
                  kernel_size=3,
                  strides=2,
                  activation=None,
                  use_bias=False,
                  kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))

model.add(keras.layers.MaxPooling1D(pool_size=2))

model.add(
    layers.Conv1D(filters=60,
                  kernel_size=3,
                  strides=2,
                  activation=None,
                  use_bias=False,
                  kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))

model.add(
    layers.Conv1D(filters=60,
                  kernel_size=3,
                  strides=2,
                  activation=None,
                  use_bias=False,
                  kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))

model.add(keras.layers.MaxPooling1D(pool_size=2))

model.add(
    layers.Conv1D(filters=80,
                  kernel_size=3,
                  strides=2,
                  activation=None,
                  use_bias=False,
                  kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))

model.add(keras.layers.MaxPooling1D(pool_size=2))

model.add(
    layers.Conv1D(filters=100,
                  kernel_size=3,
                  strides=2,
                  activation=None,
                  use_bias=False,
                  kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))

model.add(keras.layers.MaxPooling1D(pool_size=2))

model.add(layers.Flatten())

model.add(
    layers.Dense(40,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))

model.add(layers.Dense(NUM_SPEAKERS, activation='softmax'))

adam_optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=adam_optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

keras.utils.plot_model(model,
                       to_file='test_keras_plot_model.png',
                       show_shapes=True)
# display(IPython.display.Image('test_keras_plot_model.png'))
print(model.summary())
print("Just printed model summary")

start_time = time.time()
# with tf.device('/cpu:0'):
# The baseline model has input size 1911 so we could use batch size 1024.
# But the CNN has input size 11025, so we have to reduce the batch_size or the
# GPU runs out of memory.
tensor_board = keras.callbacks.TensorBoard(histogram_freq=1)
history_object = model.fit(X_train,
                           y_train,
                           epochs=25,
                           batch_size=64,
                           verbose=2,
                           callbacks=[tensor_board],
                           shuffle=True,
                           validation_data=(X_dev, y_dev))
print("%d seconds to train the model" % (time.time() - start_time))

model.save("cnn_model.h5")

#generate_confusion_matrix(model, test_set_inputs, test_set_labels)
generate_confusion_matrix(model, X_dev, y_dev)

# %%
# Plot training & validation accuracy values
plt.plot(history_object.history['acc'])
plt.plot(history_object.history['val_acc'])
plt.title('CNN Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('cnn_accuracy.png', bbox_inches='tight')
plt.close()
# plt.show()

# Plot training & validation loss values
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('CNN Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('cnn_loss.png', bbox_inches='tight')
# plt.show()
