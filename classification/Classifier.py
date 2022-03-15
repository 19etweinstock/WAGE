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

from network import runNetwork
import numpy as np
from keras.datasets import mnist

def main():
    (trainX, trainY), (testX, testY) = mnist.load_data()

    data = testX / 256.0
    answers = testY

    index = 0

    image = data[index,:,:]
    answer = answers[index]

    result, last_layer_input = runNetwork(image)
    print(testY[index])
    print(result)
    print(np.maximum(result, 0))

    # process result


if __name__ == '__main__':
    main()


"""
for i in range(0,28):
    for j in range(0,28):
        print(int(activate(image)[i,j]*2), end=' ')
    print("")


for i in range(0,24):
    for j in range(0,24):
        print(int(activate(conv2D(image, weights.weights_conv0[0, 0, :, :])[i,j])*2), end=' ')
    print("")
"""
