# the maximum output of the quantized network is 0.5
# try training with quantized output but set the answers to be 0.5 instead of 1 so that loss for perfect is 0
# otherwise there is inherent loss even for correct prediction because it has not achieved maximum value (which it cannot)


import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 
import sys
sys.path.append("..")
sys.path.append("../training")
import tensorflow as tf
import numpy as np
from training import Model

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_init = np.expand_dims(x_train, 1)[0:1]
x_train = np.expand_dims(x_train, 3)
x_test = np.expand_dims(x_test, 3)
y_train = y_train / 1.0
y_test = y_test / 1.0
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

model_quant = Model.lenet5(6,8)

model_quant(x_init)

# loading the model does not load the scaling factor
model_quant.load_weights(f'../model/2022-01-21 1844(MNIST 1288 8 [0, 1.0, 20, 0.5, 35, 0.25] 50 50)')

import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses

model = models.Sequential()
model.add(layers.Conv2D(6, 5, input_shape=x_train.shape[1:], use_bias=False))
model.add(layers.MaxPool2D(2))
model.add(layers.ReLU())
model.add(layers.Conv2D(8, 5, use_bias=False))
model.add(layers.MaxPool2D(2))
model.add(layers.ReLU())
model.add(layers.Flatten())
model.add(layers.Dense(120, use_bias=False))
model.add(layers.ReLU())
model.add(layers.Dense(84, use_bias=False))
model.add(layers.ReLU())
model.add(layers.Dense(10, use_bias=False))

model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'], run_eagerly=False)

model.fit(x_train, y_train, epochs=1,batch_size=256)

model.layers[0].set_weights(model_quant.layers[1].get_weights())
model.layers[3].set_weights(model_quant.layers[4].get_weights())
model.layers[7].set_weights(model_quant.layers[8].get_weights())
model.layers[9].set_weights(model_quant.layers[10].get_weights())
model.layers[11].set_weights(model_quant.layers[12].get_weights())


saved_model_dir = 'saved_model.json'
saved_weights_dir = 'saved_weights.h5'

json_string = model.to_json()
with open(saved_model_dir, "w+") as f:
    f.write(json_string)
model.save_weights(saved_weights_dir)