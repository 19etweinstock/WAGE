import os
# from numpy.core.defchararray import array

# from tensorflow.python.ops.gen_array_ops import reshape
import weights

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
    return (np.sign(np.abs(var))+np.sign(var))/2

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
    feature_map_0 = activate(pool(conv2D(image, weights.conv0_in0_out0)))
    feature_map_1 = activate(pool(conv2D(image, weights.conv0_in0_out1)))
    feature_map_2 = activate(pool(conv2D(image, weights.conv0_in0_out2)))
    feature_map_3 = activate(pool(conv2D(image, weights.conv0_in0_out3)))
    feature_map_4 = activate(pool(conv2D(image, weights.conv0_in0_out4)))
    feature_map_5 = activate(pool(conv2D(image, weights.conv0_in0_out5)))

    inputs_maps=np.array([feature_map_0, feature_map_1, feature_map_2, feature_map_3, feature_map_4, feature_map_5])
    weights0=np.array([weights.conv1_in0_out0, weights.conv1_in1_out0, weights.conv1_in2_out0, weights.conv1_in3_out0, weights.conv1_in4_out0, weights.conv1_in5_out0])
    weights1=np.array([weights.conv1_in0_out1, weights.conv1_in1_out1, weights.conv1_in2_out1, weights.conv1_in3_out1, weights.conv1_in4_out1, weights.conv1_in5_out1])
    weights2=np.array([weights.conv1_in0_out2, weights.conv1_in1_out2, weights.conv1_in2_out2, weights.conv1_in3_out2, weights.conv1_in4_out2, weights.conv1_in5_out2])
    weights3=np.array([weights.conv1_in0_out3, weights.conv1_in1_out3, weights.conv1_in2_out3, weights.conv1_in3_out3, weights.conv1_in4_out3, weights.conv1_in5_out3])
    weights4=np.array([weights.conv1_in0_out4, weights.conv1_in1_out4, weights.conv1_in2_out4, weights.conv1_in3_out4, weights.conv1_in4_out4, weights.conv1_in5_out4])
    weights5=np.array([weights.conv1_in0_out5, weights.conv1_in1_out5, weights.conv1_in2_out5, weights.conv1_in3_out5, weights.conv1_in4_out5, weights.conv1_in5_out5])
    weights6=np.array([weights.conv1_in0_out6, weights.conv1_in1_out6, weights.conv1_in2_out6, weights.conv1_in3_out6, weights.conv1_in4_out6, weights.conv1_in5_out6])
    weights7=np.array([weights.conv1_in0_out7, weights.conv1_in1_out7, weights.conv1_in2_out7, weights.conv1_in3_out7, weights.conv1_in4_out7, weights.conv1_in5_out7])
    weights_conv1=np.array([weights0,weights1,weights2,weights3,weights4,weights5,weights6,weights7])

    output_maps=np.zeros([8,8,8])
    
    for i in range(0,8):
        weight = weights_conv1[i]
        for j in range(0,6):
            output_maps[i] = output_maps[i] + conv2D(inputs_maps[j], weight[j])

    feature_map_fc_0 = np.reshape(activate(pool(output_maps[0])),[16])
    feature_map_fc_1 = np.reshape(activate(pool(output_maps[1])),[16])
    feature_map_fc_2 = np.reshape(activate(pool(output_maps[2])),[16])
    feature_map_fc_3 = np.reshape(activate(pool(output_maps[3])),[16])
    feature_map_fc_4 = np.reshape(activate(pool(output_maps[4])),[16])
    feature_map_fc_5 = np.reshape(activate(pool(output_maps[5])),[16])
    feature_map_fc_6 = np.reshape(activate(pool(output_maps[6])),[16])
    feature_map_fc_7 = np.reshape(activate(pool(output_maps[7])),[16])

    x = np.reshape(np.array([feature_map_fc_0,feature_map_fc_1,feature_map_fc_2,feature_map_fc_3,
                            feature_map_fc_4,feature_map_fc_5,feature_map_fc_6,feature_map_fc_7]), [128])

    x = np.matmul(np.transpose(weights.fc0),x)
    x = activate(x)
    x = np.matmul(np.transpose(weights.fc1),x)
    # x = np.matmul(x,weights.fc1)
    x = activate(x)
    x = np.matmul(np.transpose(weights.fc2),x)
    # x = np.matmul(x,weights.fc2)

    return x

def main():
    trainX, trainY, testX, testY, label = loadData('MNIST')

    # index = int(sys.argv[1])
    index = 1

    image = testX[index,:,:,0]
    answer = getAnswer(testY[index])

    result = runNetwork(activate(image))

    print(testY[index])
    print(result)



if __name__ == '__main__':
    main()
