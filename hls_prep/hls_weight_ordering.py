'''
CPU (Intel i7-7500 CPU @ 2.0GHz)
N = [10, 100, 1000, 10000]
latency = [0.0078, 0.00047, 0.000219, 0.000199], acceleration flattens out due to limited memory on a mobile cpu
GPU (GeForce 940MX)
latency = [0.2383,0.0128, 0.00132, 0.0002]
'''

import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json
import time
import sys
saved_model_dir = 'saved_model.json'
saved_weights_dir = 'saved_weights.h5'

if __name__ == "__main__":
    with open(saved_model_dir) as f:
        json_str = f.read()
    model = model_from_json(json_str)
    model.load_weights(saved_weights_dir)



    """
    TODO:
        load weights and output to c file using nditer()

        set bias files to 0

        fix output and activation layers
        pray

    """

"""
    f.write("[{}]".format(np.prod(a.shape)))
    f.write(" = {")
    
    #fill c++ array.  
    #not including internal brackets for multidimensional case
    i=0
    for x in np.nditer(a, order='C'):
        if i==0:
            f.write("%.12f" % x)
        else:
            f.write(", %.12f" % x)
        i=i+1
    f.write("};\n")
    f.close()

    """