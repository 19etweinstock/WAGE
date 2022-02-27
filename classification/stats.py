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


def conv2D(var, kernel):
    '''3D convolution by sub-matrix summing.
    Args:
        var (ndarray): 2d or 3d array to convolve along the first 2 dimensions.
        kernel (ndarray): 2d or 3d kernel to convolve. If <var> is 3d and <kernel>
            is 2d, create a dummy dimension to be the 3rd dimension in kernel.
    Keyword Args:
        stride (int): stride along the 1st 2 dimensions. Default to 1.
        pad (int): number of columns/rows to pad at edges.
    Returns:
        result (ndarray): convolution result.
    '''
    var_ndim = np.ndim(var)
    ny, nx = var.shape[:2]
    ky, kx = kernel.shape[:2]
    result = 0
    for ii in range(ky*kx):
        yi, xi = divmod(ii, kx)
        slabii = var[yi:ny-ky+yi+1:1,
                         xi:nx-kx+xi+1:1, ...]*kernel[yi, xi]
        if var_ndim == 3:
            slabii = slabii.sum(axis=-1)
        result += slabii
    return result

def activate(var):
    relued = np.maximum(var, 0)
    max = 0.5
    min = -0.5
    temp = np.clip(relued, min, max)
    return np.round(2 * temp) / 2

def pool(img, factor=2):
    """ Perform max pooling with a (factor x factor) kernel"""
    ds_img = np.full((img.shape[0] // factor, img.shape[1] // factor), -float('inf'), dtype=img.dtype)
    np.maximum.at(ds_img, (np.arange(img.shape[0])[:, None] // factor, np.arange(img.shape[1]) // factor), img)
    return ds_img

def runNetwork(image):

    image = activate(image)
    feature_maps_layer0 = weights.weights_conv0.shape[0]
    feature_maps_layer1 = weights.weights_conv1.shape[0]

    feature_map_conv0 = np.zeros([feature_maps_layer0,12,12])
    feature_map_conv1 = np.zeros([feature_maps_layer1,8,8])

    for i in range(0, feature_maps_layer0):
        feature_map_conv0[i, :, :] = activate(pool(conv2D(image, weights.weights_conv0[i, 0, :, :])))

    for i in range(0, feature_maps_layer1):
        for j in range(0, feature_maps_layer0):
            feature_map_conv1[i, :, :] += conv2D(feature_map_conv0[j,:,:], weights.weights_conv1[i, j, :, :])

    x = np.zeros([16*feature_maps_layer1])

    for i in range(0, feature_maps_layer1):
        x[i*16:i*16+16] = np.reshape(activate(pool(feature_map_conv1[i,:,:])), [16])

    x = np.matmul(x,weights.fc0)
    x = activate(x)
    x = np.matmul(x,weights.fc1)
    x = activate(x)
    x = np.matmul(x,weights.fc2)

    return x

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
