import os
import numpy as np
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 
import tensorflow as tf
import myInitializer

import Quantize
from functools import reduce

def _arr(_stride):
    return [1, 1, _stride, _stride]

def _QE(x):
    if Quantize.bitsE <= 16:
        x = Quantize.E(x)
    return x
    
def _QA(x):
    if Quantize.bitsA <= 16:
        x = Quantize.A(x)
    return x

class qa(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()

    def call(self, inputs):
        return _QA(inputs)

class qconv2d(tf.keras.layers.Layer):
    def __init__(self, ksize, c_out, stride=1, padding='VALID'):
        super().__init__()
        self.ksize = ksize
        self.c_out = c_out
        self.stride = stride
        self.padding = padding
        
    def build(self, input_shape):
        #input shape is NCHW:
        self.kernel = self.add_weight("kernel",
                                      shape=[self.ksize, self.ksize, input_shape[1], self.c_out],
                                      initializer=myInitializer.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True),
                                      dtype=tf.float32)
        self.scale = Quantize.W_scale[-1]

    def call(self, inputs):
        weights = Quantize.W(self.kernel, self.scale)
        return tf.nn.conv2d(input=inputs, \
                            filters=weights, \
                            strides=_arr(self.stride), \
                            padding=self.padding, \
                            data_format='NCHW', \
                            name=self.name)

class maxpool(tf.keras.layers.Layer):
    def __init__(self, ksize=2, stride=2, padding='SAME'):
        super().__init__()
        self.ksize = ksize
        self.stride = stride
        self.padding = padding

    def call(self, inputs):
        return tf.nn.max_pool(  inputs,\
                                _arr(self.ksize), \
                                _arr(self.stride), \
                                self.padding, \
                                'NCHW')

class qactivation(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        x = tf.nn.relu(inputs)
        x = _QE(x)
        x = _QA(x)
        return x

class qsoftmax(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        x = tf.clip_by_value(inputs, 0, 1)
        x = _QE(x)
        return x

class qfc(tf.keras.layers.Layer):
    def __init__(self, c_out) -> None:
        super().__init__()
        self.c_out = c_out

    def build(self, input_shape):
        #input shape is NCHW:
        self.kernel = self.add_weight("kernel",
                                      shape=[input_shape[1], self.c_out],
                                      initializer=myInitializer.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True),
                                      dtype=tf.float32)
        self.scale = Quantize.W_scale[-1]
    
    def call(self, inputs):
        # self.input_array=inputs
        weights = Quantize.W(self.kernel, self.scale)
        # weights = self.kernel
        return tf.matmul(inputs, weights)

class reshape(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()

    def build(self, input_shape):
        # self.shape=input_shape
        self.shape = reduce(lambda inputs, y: inputs * y, input_shape.as_list()[1:])
    
    def call(self, inputs):
        return tf.reshape(inputs, [-1, self.shape])