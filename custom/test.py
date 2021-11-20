import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 
import tensorflow as tf
import numpy as np
import GetData
import Quantize
import time

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, 1)
x_test = np.expand_dims(x_test, 1)
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

# with tf.device('/cpu:0'):
#   x_train,y_train,x_test,y_test,_ =\
#       GetData.loadData('MNIST')

# x_train, x_test = x_train / 255.0, x_test / 255.0
# x_train = tf.cast(tf.transpose(x_train, [0,3,1,2]),dtype=tf.float32)
# x_test = tf.cast(tf.transpose(x_test, [0,3,1,2]),dtype=tf.float32)
# y_train = tf.constant(y_train / 1.0,dtype=tf.float32)
# y_test = tf.constant(y_test / 1.0,dtype=tf.float32)
import Model

model = Model.lenet5(6,8)

model.compile(optimizer=tf.keras.optimizers.SGD(\
              learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay([100], [1.1, 1.1])),
              loss='mse',
              metrics=['accuracy'],
              run_eagerly=False)

model.fit(x_train, y_train, epochs=100,batch_size=128)
result=model.evaluate(x_test, y_test)

vars = model.variables

def upper(str):
    return str.upper()

Time = time.strftime('%Y-%m-%d %H%M', time.localtime())


f = open(f'../weights/{result[1]} {Time} {Quantize.bitsW}{Quantize.bitsA}{upper(hex(Quantize.bitsG)[2:])}{upper(hex(Quantize.bitsE)[2:])} {Quantize.bitsR}', "wt")
f.write("import numpy as np\n\n")
f.flush()

for var in range(0,2):
    tensor=vars[var].value()
    for in_filter in range(0, tensor.shape.as_list()[2]):
        for out_filter in range(0,tensor.shape.as_list()[3]):
            f.write(f"conv{var}_in{in_filter}_out{out_filter} = np.array([\n")
            quant = Quantize.W(tensor)
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
    quant = Quantize.W(tensor)
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