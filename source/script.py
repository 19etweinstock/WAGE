import os
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 
# import tensorflow as tf
import numpy as np
import Classifier
import weights
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.disable_v2_behavior()
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# config.allow_soft_placement = True
# config.log_device_placement = False
# sess = tf.compat.v1.InteractiveSession(config=config)
# saver = tf.compat.v1.train.import_meta_graph('../model/2021-10-25 1108(MNIST 11DD 16 [0, 1.1] 100 128 ).tf.meta')
# saver.restore(sess,tf.compat.v1.train.latest_checkpoint('../model/./'))
# graph = tf.compat.v1.get_default_graph() 
# vars = graph.get_collection('variables')


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

trainX, trainY, testX, testY, label = loadData('MNIST')

def display(array):
    for i in range (0, array.shape[0]):
        for j in range (0, array.shape[1]):
            print(np.sign(array[i,j]).astype('uint8'), end='')
        print('')

def process():
  for i in range(0, 10000):
    input = testX[i,:,:,0]
    # count = 0
    # if np.any(Classifier.conv2D(input, weights.conv0_in0_out0) > 0):
    #   count = count + 1
    # if np.any(Classifier.conv2D(input, weights.conv0_in0_out1) > 0):
    #   count = count + 1
    # if np.any(Classifier.conv2D(input, weights.conv0_in0_out3) > 0):
    #   count = count + 1
    # if np.any(Classifier.conv2D(input, weights.conv0_in0_out4) > 0):
    #   count = count + 1
    # if np.any(Classifier.conv2D(input, weights.conv0_in0_out5) > 0):
    #   count = count + 1
    # if count == 5:
    #   print(i)

    if np.any(Classifier.runNetwork(input)):
      print(i)