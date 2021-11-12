import os
import numpy as np
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 
import tensorflow as tf

import Layers

class lenet5(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.conv0 = Layers.qconv2d(5, 6, name='conv0')
        self.pool0 = Layers.maxpool()
        self.activation0 = Layers.qactivation()

        self.conv1 = Layers.qconv2d(5, 8, name='conv1')
        self.pool1 = Layers.maxpool()
        self.activation1 = Layers.qactivation()

        self.reshape=Layers.reshape()

        self.fc0 = Layers.qfc(120, 'fc0')
        self.activation2 = Layers.qactivation()
        self.fc1 = Layers.qfc(84, 'fc1')
        self.activation3 = Layers.qactivation()
        self.fc2 = Layers.qfc(10, 'fc2')

        self.QA = Layers.QA
        self.QE = Layers.QE

    def call(self, input, training=False):
        x = self.conv0(input)
        x = self.pool0(x)
        x = self.activation0(x)

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.activation1(x)

        x = self.reshape(x)

        x = self.fc0(x)
        x = self.activation2(x)
        x = self.fc1(x)
        x = self.activation3(x)
        x = self.fc2(x)

        x = self.QA(x)
        x = self.QE(x)

        return x
