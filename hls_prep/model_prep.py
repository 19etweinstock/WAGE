# the maximum output of the quantized network is 0.5
# try training with quantized output but set the answers to be 0.5 instead of 1 so that loss for perfect is 0
# otherwise there is inherent loss even for correct prediction because it has not achieved maximum value (which it cannot)


import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 
import sys
sys.path.append("../training")
import tensorflow as tf
import numpy as np
import Model, Quantize

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
model.add(layers.Conv2D(6, 5, input_shape=x_train.shape[1:], use_bias=True, activation='relu'))
model.add(layers.MaxPool2D(2))
# model.add(layers.Activation("relu"))
model.add(layers.Conv2D(8, 5, use_bias=True, activation='relu'))
model.add(layers.MaxPool2D(2))
# model.add(layers.Activation("relu"))
model.add(layers.Flatten())
model.add(layers.Dense(120, use_bias=True, activation='relu'))
# model.add(layers.Activation("relu"))
model.add(layers.Dense(84, use_bias=True, activation='relu'))
# model.add(layers.Activation("relu"))
model.add(layers.Dense(10, use_bias=True, activation='relu'))

model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'], run_eagerly=True)
a=Quantize.W(model_quant.layers[ 1].get_weights()[0], Quantize.W_scale[0]).numpy(), model.layers[ 0].get_weights()[1]
b=Quantize.W(model_quant.layers[ 4].get_weights()[0], Quantize.W_scale[1]).numpy(), model.layers[ 2].get_weights()[1]
c=Quantize.W(model_quant.layers[ 8].get_weights()[0], Quantize.W_scale[2]).numpy(), model.layers[ 5].get_weights()[1]
d=Quantize.W(model_quant.layers[10].get_weights()[0], Quantize.W_scale[3]).numpy(), model.layers[ 6].get_weights()[1]
e=Quantize.W(model_quant.layers[12].get_weights()[0], Quantize.W_scale[4]).numpy(), model.layers[ 7].get_weights()[1]
model.layers[ 0].set_weights(weights=a)
model.layers[ 2].set_weights(weights=b)
model.layers[ 5].set_weights(weights=c)
model.layers[ 6].set_weights(weights=d)
model.layers[ 7].set_weights(weights=e)


saved_model_dir = 'saved_model.json'
saved_weights_dir = 'saved_weights.h5'

json_string = model.to_json()
with open(saved_model_dir, "w+") as f:
    f.write(json_string)
model.save_weights(saved_weights_dir)

vars = model.variables

f = open(f'test.py', "wt")
f.write("\nimport numpy as np\n\n")
f.flush()

for var in range(0,4,2):
    tensor=vars[var].value()
    for in_filter in range(0, tensor.shape.as_list()[2]):
        for out_filter in range(0,tensor.shape.as_list()[3]):
            f.write(f"conv{var//2}_in{in_filter}_out{out_filter} = np.array([\n")
            quant = tensor
            quant_array = quant.numpy()
            for row in range(0,5):
                f.write("\t[")
                for col in range(0,5):
                    f.write(f"{quant_array[row,col,in_filter,out_filter]}{', ' if col != 4 else ''}")
                f.write("]")
                if (row != 4):
                    f.write(",\n")
                
            f.write("])\n\n")
            f.flush()
    for out_filter in range(0, tensor.shape.as_list()[3]):
        f.write(f"weights_conv{var//2}_out{out_filter} = np.array([\n")
        for in_filter in range(0, tensor.shape.as_list()[2]):
            f.write(f"\tconv{var//2}_in{in_filter}_out{out_filter}")
            if (in_filter != tensor.shape.as_list()[2] - 1):
                f.write(",\n")
            else:
                f.write("])\n\n")
    
    f.write(f"weights_conv{var//2} = np.array([\n")
    for out_filter in range(0, tensor.shape.as_list()[3]):
        f.write(f"\tweights_conv{var//2}_out{out_filter}")
        if (out_filter != tensor.shape.as_list()[3] - 1):
            f.write(",\n")
        else:
            f.write("])\n\n")
for var in range(4,10,2):
    tensor=vars[var].value()
    quant = tensor
    quant_array = quant.numpy()
    f.write(f"fc{var//2-2} = np.array([\n")
    rows=tensor.shape.as_list()[0]
    cols=tensor.shape.as_list()[1]
    for row in range(0,rows):
        f.write("\t[")
        for col in range(0, cols):
            f.write(f"{quant_array[row,col]}{', ' if col != (cols-1) else ''}")
        f.write("]")
        if (row != (rows -1)):
            f.write(",\n")
        
    f.write("])\n\n")
    f.flush()
        
f.close()