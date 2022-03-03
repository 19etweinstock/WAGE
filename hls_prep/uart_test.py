'''
Saves mnist images to txt files
'''

import numpy as np
import keras

from keras.datasets import mnist

(_, _), (x_test, y_test) = mnist.load_data()

# x_test = x_test.astype('float32')
x_test = x_test / 256

# print('shape shape:', x_test.shape)
# print(, 'test samples')
x_test_len = x_test.shape[0]

index = 1

image = x_test[index]
output = ""
for j in range (27, -1, -1):
    for k in range(27, -1, -1):
        if (image[j][k] > 1/4):
            output = output + "1"
        else:
            output = output + "0"

print(output)



value = 0
index = 2 ** 7
count = 0
for j in range (27, -1, -1):
    for k in range(27, -1, -1):
        if (image[j][k] > 1/4):
            value = value + index
        index = index // 2
        count = count + 1
        if (count == 8):
            s = bin(value).replace("0b",'')
            if (len(s) != 8):
                diff = 8 - len(s)
                s = "0" * diff + s
            print(s, end='')
            value = 0
            index = 2 ** 7
            count = 0