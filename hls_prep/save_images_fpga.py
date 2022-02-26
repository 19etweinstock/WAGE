'''
Saves mnist images to txt files
'''

import numpy as np
import keras

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.astype('float32')
x_test /= 256

# print('shape shape:', x_test.shape)
# print(, 'test samples')
x_test_len = x_test.shape[0]
index_arr =     list([0,1,2,3,4,7,8,11,18,61])
number_arr = list([7,2,1,0,4,9,5, 6, 3, 8])

for i in range(0,10):
    index = index_arr[i]
    number = number_arr[i]

    image = x_test[index]
    output = "b"
    for j in range (27, -1, -1):
        for k in range(27, -1, -1):
            if (image[j][k] > 1/4):
                output = output + "1"
            else:
                output = output + "0"
    with open('fpga_strings/' + str(number)+'.txt', 'w') as outfile:
        outfile.write(output)