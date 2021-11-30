import os
# from numpy.core.defchararray import array

# from tensorflow.python.ops.gen_array_ops import reshape
# 9670 is better with 0.9651 prediction

import weights_9670_9621 as weights

import sys

import numpy as np

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

def loadData(dataSet,validNum=0):
    pathNPZ = '../dataSet/' + dataSet + '.npz'
    numpyTrainX, numpyTrainY, numpyTestX, numpyTestY, label = loadNPZ(pathNPZ, validNum)
    return numpyTrainX,numpyTrainY,numpyTestX,numpyTestY,label

def loadNPZ(pathNPZ, validNum=0):
  data = np.load(pathNPZ)

  trainX = data['trainX']
  trainY = data['trainY']

  if validNum > 100:
    testX = trainX[-validNum:]
    testY = trainY[-validNum:]
    trainX = trainX[0:-validNum]
    trainY = trainY[0:-validNum]
  else:
    testX = data['testX']
    testY = data['testY']

  label = data['label']
  return trainX, trainY, testX, testY, label

def getAnswer(label):
    for i in range(0,10):
        if (label[i] == 1):
            return i

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
    trainX, trainY, testX, testY, label = loadData('MNIST')

    data = testX / 255.0
    answers = testY

    i = 0
    for index in range(0, data.shape[0]):

    # index = int(sys.argv[1])
    # index = 8

        image = data[index,:,:,0]
        answer = getAnswer(answers[index])

        result = runNetwork(image)

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
    print(i/data.shape[0])



if __name__ == '__main__':
    main()
