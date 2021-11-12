import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 
import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, 1)
x_test = np.expand_dims(x_test, 1)
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)


import Model

model = Model.lenet5()

def lossf(y_true, y_pred):
  return 0.5 * tf.reduce_sum(tf.square(y_true - y_pred))
model.compile(optimizer=tf.keras.optimizers.SGD(\
              learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay([100], [1.0, 1.0])),
              loss=lossf,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)



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