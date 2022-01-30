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
from training.Top import sse, getlr
from training.Option import lr_schedule

mnist = tf.keras.datasets.mnist
(x_train, _),(_, _) = mnist.load_data()
x_train = x_train / 256.0
x_train = np.expand_dims(x_train, 1)

model = Model.lenet5(6,8)

#initialize kernels and scaling factor
model(x_train[0:1,:,:,:])

# loading the model does not load the scaling factor
model.load_weights(f'../model/2022-01-21 1844(MNIST 1288 8 [0, 1.0, 20, 0.5, 35, 0.25] 50 50)')

lr = getlr(lr_schedule)

model.compile(optimizer=tf.keras.optimizers.SGD(\
              learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr[0], lr[1])),
              loss=sse,
              metrics=['accuracy'],
              run_eagerly=False)

saved_model_dir = 'saved_model.json'
saved_weights_dir = 'saved_weights.h5'

# json_string = model.to_json()
# with open(saved_model_dir, "w+") as f:
#     f.write(json_string)
model.save_weights(saved_weights_dir)