'''
Saves mnist images to txt files
'''

import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.astype('float32')
x_test /= 256

# print('shape shape:', x_test.shape)
# print(, 'test samples')
x_test_len = 100

for i in range(x_test_len):
    with open('test_images/'+str(i)+'.txt', 'w') as outfile:
        outfile.write('# '+str(y_test[i])+'\n')
        np.savetxt(outfile, x_test[i])