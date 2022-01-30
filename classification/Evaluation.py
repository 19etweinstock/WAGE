# the maximum output of the quantized network is 0.5
# try training with quantized output but set the answers to be 0.5 instead of 1 so that loss for perfect is 0
# otherwise there is inherent loss even for correct prediction because it has not achieved maximum value (which it cannot)


import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 
import tensorflow as tf
import numpy as np
import time
from . import Option
from . import Log
from functools import reduce
from . import Model

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 256.0, x_test / 256.0
x_train = np.expand_dims(x_train, 1)
x_test = np.expand_dims(x_test, 1)
y_test_val = y_test
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

Time = time.strftime('%Y-%m-%d %H%M', time.localtime())

def upper(str):
    return str.upper()

Notes = f'MNIST {Option.bitsW}{Option.bitsA}{upper(hex(Option.bitsG)[2:])}{upper(hex(Option.bitsE)[2:])} {Option.bitsR} {Option.lr_schedule} {Option.Epoch} {Option.batchSize}'

pathLog = '../log/' + Time + '(' + Notes + ')' + '.txt'
# log = Log.Log(pathLog, 'wt', 1) # set log file
# setattr(Log, 'isatty', True)
# print(Time)
# print(open('Option.py').read())


model = Model.lenet5(6,8)

lr = [[],[Option.lr_schedule[1]]]

for i in range (0, len(Option.lr_schedule), 2):
    lr[0].append(Option.lr_schedule[i])
    lr[1].append(Option.lr_schedule[i+1])

print(lr)

#initialize kernels and scaling factor
model(x_train[0:1,:,:,:])

# loading the model does not load the scaling factor
model.load_weights(f'../model/2022-01-21 1844(MNIST 1288 8 [0, 1.0, 20, 0.5, 35, 0.25] 50 50)')

model.summary()

layers = model.layers

# conv_weights = 0
# fc_weights = 0
# print fc & conv weights
# for i in range(0, len(layers)):
#     if (layers[i].weights):
#         if ('conv' in layers[i].name):
#             print(f'conv/{layers[i].name} {layers[i].kernel.shape.as_list()} Scale: {layers[i].scale}')
#             conv_weights += reduce(lambda x, y: x * y, layers[i].kernel.shape.as_list())
#         elif ('fc' in layers[i].name):
#             print(f'fc/{layers[i].name} {layers[i].kernel.shape.as_list()} Scale: {layers[i].scale}')
#             fc_weights += reduce(lambda x, y: x * y, layers[i].kernel.shape.as_list())
# print(f'CONV: {conv_weights} FC: {fc_weights} TOTAL: {conv_weights + fc_weights}')

def sse(y_true, y_pred):
    return tf.math.reduce_sum(tf.square(y_true - y_pred))

model.compile(optimizer=tf.keras.optimizers.SGD(\
              learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr[0], lr[1])),
              loss=sse,
              metrics=['accuracy'],
              run_eagerly=False)

test=model.evaluate(x_test, y_test)
train =model.evaluate(x_train, y_train)              

prediction=tf.math.argmax(model.predict(x_test), 1)
cm = tf.math.confusion_matrix(labels=y_test_val,predictions=prediction, num_classes=10)
print(cm)

plt.figure(figsize = (10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

print(classification_report(y_test_val, prediction))