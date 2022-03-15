import os

#                   test train
# 9670 9613 reaches 9652 
# 9675 9613 reaches 9640 
# 9679 9663 reaches 9656 9632
# 9680 9610 reaches 9656
# 9684 9665 reaches 9666 96365
# 9686 9657 reaches 9656
# 9681 9627 reaches 9658

# 9684 is the best weight set

import weights_9684_9665 as weights

import sys

import numpy as np

from keras.datasets import mnist

from network import runNetwork

def main():
    (trainX, trainY), (testX, testY) = mnist.load_data()

    data = testX / 256.0
    answers = testY

    f = open(f'last_layer_sparsity_test_9684_9665.txt', "wt")

    i = 0
    res = ''
    for index in range(0, data.shape[0]):

    # index = int(sys.argv[1])
    # index = 8

        image = data[index,:,:]
        answer = answers[index]

        result, last_layer = runNetwork(image)
        count = np.count_nonzero(last_layer != 0)
        res = res + str(count) + '\n'
        i += count

    print(f'Average Sparsity: {i/data.shape[0]/84}')
    f.write(f'Average Sparsity: {i/data.shape[0]/84}')
    f.write('\n')
    f.write(res)
    f.close()

if __name__ == '__main__':
    main()
