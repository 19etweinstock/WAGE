import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 
import tensorflow as tf
import numpy as np
import Log
import time

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, 3)
x_test = np.expand_dims(x_test, 3)
y_train = y_train / 1.0
y_test = y_test / 1.0

Time = time.strftime('%Y-%m-%d %H%M', time.localtime())

def upper(str):
    return str.upper()

Notes = f'MNIST precision'

pathLog = '../log/' + Time + '(' + Notes + ')' + '.txt'
log = Log.Log(pathLog, 'wt', 1) # set log file
setattr(Log, 'isatty', True)
print(Time)

def sse(y_true, y_pred):
    return tf.math.reduce_sum(tf.square(y_true - y_pred))

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(6,5, padding='valid', data_format='channels_first', use_bias=False),
#     tf.keras.layers.MaxPool2D(strides=(2,2), data_format='channels_first'),
#     tf.keras.layers.ReLU(),
#     tf.keras.layers.Conv2D(8,5, padding='valid', data_format='channels_first', use_bias=False),
#     tf.keras.layers.MaxPool2D(strides=(2,2), data_format='channels_first'),
#     tf.keras.layers.ReLU(),
#     tf.keras.layers.Reshape((-1,)),
#     tf.keras.layers.Dense(120, use_bias=False),
#     tf.keras.layers.ReLU(),
#     tf.keras.layers.Dense(84, use_bias=False),
#     tf.keras.layers.ReLU(),
#     tf.keras.layers.Dense(10, use_bias=False),
#     tf.keras.layers.Softmax()
# ])

import tensorflow.keras.models as models
import tensorflow.keras.layers as layers

model = models.Sequential()
model.add(layers.Conv2D(6, 5, activation='relu', input_shape=x_train.shape[1:]))
model.add(layers.MaxPool2D(2))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(8, 5, activation='relu'))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('relu'))
model.add(layers.Flatten())
model.add(layers.Dense(120, activation='relu'))
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()


model.compile(optimizer=tf.keras.optimizers.SGD(\
              learning_rate=1.0),
              loss=sse,
              metrics=['accuracy'],
              run_eagerly=False)

model.fit(x_train, y_train, epochs=40,batch_size=64, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)
