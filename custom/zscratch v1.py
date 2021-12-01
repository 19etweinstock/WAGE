import os
from re import T
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 
import tensorflow as tf
import numpy as np
import GetData
import Quantize
import Log
import time
# mnist = tf.keras.datasets.mnist
# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
# x_train = np.expand_dims(x_train, 1)
# x_test = np.expand_dims(x_test, 1)
# y_train = tf.one_hot(y_train, 10)
# y_test = tf.one_hot(y_test, 10)

Time = time.strftime('%Y-%m-%d %H%M', time.localtime())
pathLog = '../log/' + Time + '.txt'
# Log.Log(pathLog, 'w+', 1) # set log file
# print(time.strftime('%Y-%m-%d %X', time.localtime()), '\n')

with tf.device('/cpu:0'):
  x_train,y_train,x_test,y_test,_ =\
      GetData.loadData('MNIST')

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = tf.cast(tf.transpose(x_train, [0,3,1,2]),dtype=tf.float32)
x_test = tf.cast(tf.transpose(x_test, [0,3,1,2]),dtype=tf.float32)
y_train = tf.constant(y_train / 1.0,dtype=tf.float32)
y_test = tf.constant(y_test / 1.0,dtype=tf.float32)

class quant(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return w

class SSE(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return 0.5 * tf.reduce_sum(tf.square(y_true - y_pred))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,5, padding='valid', data_format='channels_first', use_bias=False, kernel_constraint=quant()),
    tf.keras.layers.MaxPool2D(strides=(2,2), data_format='channels_first'),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(64,5, padding='valid', data_format='channels_first', use_bias=False, kernel_constraint=quant()),
    tf.keras.layers.MaxPool2D(strides=(2,2), data_format='channels_first'),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Reshape((-1,)),
    tf.keras.layers.Dense(120, use_bias=False, kernel_constraint=quant()),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(84, use_bias=False, kernel_constraint=quant()),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(10, use_bias=False, kernel_constraint=quant()),
])


def my_mse_loss():
    def mse(y_true, y_pred):
        return tf.math.reduce_mean(tf.square(y_pred - y_true))
    return mse

# import Model

# model = Model.lenet5()

def lossf(y_true, y_pred):
    return tf.math.reduce_sum(tf.square(y_true - y_pred))
model.compile(optimizer=tf.keras.optimizers.SGD(\
              learning_rate=1.0),
              loss=lossf,
              metrics=['accuracy'],
              run_eagerly=True)

model.fit(x_train, y_train, epochs=5,batch_size=128, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

vars = model.variables

# f = open(f"weights_{Time}.py", "wt")
# f.write("import numpy as np\n\n")
# f.flush()

# for var in range(0,2):
#     tensor=vars[var].value()
#     for in_filter in range(0, tensor.shape.as_list()[2]):
#         for out_filter in range(0,tensor.shape.as_list()[3]):
#             f.write(f"conv{var}_in{in_filter}_out{out_filter} = np.array([\n")
#             quant = Quantize.W(tensor)
#             quant_array = quant.numpy()
#             for row in range(0,5):
#                 f.write("\t[")
#                 for col in range(0,5):
#                     f.write(f"{quant_array[row,col,in_filter,out_filter]}{', ' if col != 4 else ''}")
#                 f.write("]")
#                 if (row != 4):
#                     f.write(",\n")
                
#             f.write("])\n\n")
#             f.flush()
#     for out_filter in range(0, tensor.shape.as_list()[3]):
#         f.write(f"weights_conv{var}_out{out_filter} = np.array([\n")
#         for in_filter in range(0, tensor.shape.as_list()[2]):
#             f.write(f"\tconv{var}_in{in_filter}_out{out_filter}")
#             if (in_filter != tensor.shape.as_list()[2] - 1):
#                 f.write(",\n")
#             else:
#                 f.write("])\n\n")
    
#     f.write(f"weights_conv{var} = np.array([\n")
#     for out_filter in range(0, tensor.shape.as_list()[3]):
#         f.write(f"\tweights_conv{var}_out{out_filter}")
#         if (out_filter != tensor.shape.as_list()[3] - 1):
#             f.write(",\n")
#         else:
#             f.write("])\n\n")
# for var in range(2,5):
#     tensor=vars[var].value()
#     quant = Quantize.W(tensor)
#     quant_array = quant.numpy()
#     f.write(f"fc{var-2} = np.array([\n")
#     rows=tensor.shape.as_list()[0]
#     cols=tensor.shape.as_list()[1]
#     for row in range(0,rows):
#         f.write("\t[")
#         for col in range(0, cols):
#             f.write(f"{quant_array[row,col]}{', ' if col != (cols-1) else ''}")
#         f.write("]")
#         if (row != (rows -1)):
#             f.write(",\n")
        
#     f.write("])\n\n")
#     f.flush()
        
# f.close()

# class MyDenseLayer(tf.keras.layers.Layer):
#   def __init__(self, num_outputs):
#     super(MyDenseLayer, self).__init__()
#     self.num_outputs = num_outputs
#   def build(self, input_shape):
#     print("building")
#     print(input_shape)
#     self.kernel = self.add_weight("kernel",
#                                   shape=[int(input_shape[-1]),
#                                          self.num_outputs])
#   def call(self, inputs):
#     print("calling")
#     return tf.matmul(inputs, self.kernel)

# layer = MyDenseLayer(10)

# layer=qconv2d(5,6)

# mnist = tf.keras.datasets.mnist
# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# x = x_train[0:10]
# x = np.expand_dims(x, 1)

# x.shape

# layer(x)