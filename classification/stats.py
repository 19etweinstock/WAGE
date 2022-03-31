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

    data = trainX / 256.0
    answers = trainY

    f = open(f'train_out_9684_9665.py', "wt")
    f.write("\nimport numpy as np\n\n")

    f.write('output = np.array([')
    i = 0
    for index in range(0, data.shape[0]):

    # index = int(sys.argv[1])
    # index = 8

        image = data[index,:,:]
        answer = answers[index]

        result = runNetwork(image)
        f.write(f"\t[{result[0]}, {result[1]}, {result[2]}, {result[3]}, {result[4]}, {result[5]}, {result[6]}")
        f.write(f", {result[7]}, {result[8]}, {result[9]}]{', ' if index != (data.shape[0] -1) else ''}")

        # print(testY[index])
        # print(result)
        # print(np.maximum(result, 0))

        # process result
        max = np.max(result)
        check = result == max
        if (np.sum(check) == 1 and check[answer] == 1):
            # print(index)
            # print(result)
            i +=1
            f.write("\n")
        else:
            f.write(f"#incorrect answer: {answer} prediction: {max}\n")
    print(f'{i/data.shape[0]}')
    f.write(f'])\n\n # Correct: {i/data.shape[0]}')



if __name__ == '__main__':
    main()
