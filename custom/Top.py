# the maximum output of the quantized network is 0.5
# try training with quantized output but set the answers to be 0.5 instead of 1 so that loss for perfect is 0
# otherwise there is inherent loss even for correct prediction because it has not achieved maximum value (which it cannot)


import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 
import tensorflow as tf
import numpy as np
import Quantize
import time
import Option
import Log
from functools import reduce
import Model

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 256.0, x_test / 256.0
x_train = np.expand_dims(x_train, 1)
x_test = np.expand_dims(x_test, 1)
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)


Time = time.strftime('%Y-%m-%d %H%M', time.localtime())

def upper(str):
    return str.upper()

Notes = f'MNIST {Option.bitsW}{Option.bitsA}{upper(hex(Option.bitsG)[2:])}{upper(hex(Option.bitsE)[2:])} {Option.bitsR} {Option.lr_schedule} {Option.Epoch} {Option.batchSize}'

pathLog = '../log/' + Time + '(' + Notes + ')' + '.txt'
log = Log.Log(pathLog, 'wt', 1) # set log file
setattr(Log, 'isatty', True)
print(Time)
print(open('Option.py').read())


model = Model.lenet5(6,8)

lr = [[],[Option.lr_schedule[1]]]

for i in range (0, len(Option.lr_schedule), 2):
    lr[0].append(Option.lr_schedule[i])
    lr[1].append(Option.lr_schedule[i+1])

print(lr)

#initialize kernels and scaling factor
model(x_train[0:1,:,:,:])

if Option.loadModel is not None:
    # loading the model does not load the scaling factor
    model.load_weights(f'../model/{Option.loadModel}')

model.summary()

layers = model.layers

conv_weights = 0
fc_weights = 0
# print fc & conv weights
for i in range(0, len(layers)):
    if (layers[i].weights):
        if ('conv' in layers[i].name):
            print(f'conv/{layers[i].name} {layers[i].kernel.shape.as_list()} Scale: {layers[i].scale}')
            conv_weights += reduce(lambda x, y: x * y, layers[i].kernel.shape.as_list())
        elif ('fc' in layers[i].name):
            print(f'fc/{layers[i].name} {layers[i].kernel.shape.as_list()} Scale: {layers[i].scale}')
            fc_weights += reduce(lambda x, y: x * y, layers[i].kernel.shape.as_list())
print(f'CONV: {conv_weights} FC: {fc_weights} TOTAL: {conv_weights + fc_weights}')

def sse(y_true, y_pred):
    return tf.math.reduce_sum(tf.square(y_true - y_pred))

model.compile(optimizer=tf.keras.optimizers.SGD(\
              learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr[0], lr[1])),
              loss=sse,
              metrics=['accuracy'],
              run_eagerly=False)

savePath = '../model/' + Time + '(' + Notes + ')'

mycallbacks = [
    tf.keras.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=(savePath),
        save_weights_only=True,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]

model.fit(x_train, y_train, epochs=Option.Epoch,batch_size=Option.batchSize, callbacks=mycallbacks,
            validation_data=(x_test, y_test), 
            validation_batch_size=256)

# get the best possible run
model.load_weights(savePath)
test=model.evaluate(x_test, y_test)
train =model.evaluate(x_train, y_train)

vars = model.variables

f = open(f'../weights/{test[1]:.4f} {train[1]:.4f} {Time} ({Notes}).py', "wt")
f.write("\nimport numpy as np\n\n")
f.write(f'bitsW = {Quantize.bitsW}\nbitsA = {Quantize.bitsA}\nbitsG = {Quantize.bitsG}\nbitsE = {Quantize.bitsE}\n')

f.flush()

for var in range(0,2):
    tensor=vars[var].value()
    for in_filter in range(0, tensor.shape.as_list()[2]):
        for out_filter in range(0,tensor.shape.as_list()[3]):
            f.write(f"conv{var}_in{in_filter}_out{out_filter} = np.array([\n")
            quant = Quantize.W(tensor, Quantize.W_scale[var])
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
        f.write(f"weights_conv{var}_out{out_filter} = np.array([\n")
        for in_filter in range(0, tensor.shape.as_list()[2]):
            f.write(f"\tconv{var}_in{in_filter}_out{out_filter}")
            if (in_filter != tensor.shape.as_list()[2] - 1):
                f.write(",\n")
            else:
                f.write("])\n\n")
    
    f.write(f"weights_conv{var} = np.array([\n")
    for out_filter in range(0, tensor.shape.as_list()[3]):
        f.write(f"\tweights_conv{var}_out{out_filter}")
        if (out_filter != tensor.shape.as_list()[3] - 1):
            f.write(",\n")
        else:
            f.write("])\n\n")
for var in range(2,5):
    tensor=vars[var].value()
    quant = Quantize.W(tensor, Quantize.W_scale[var])
    quant_array = quant.numpy()
    f.write(f"fc{var-2} = np.array([\n")
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