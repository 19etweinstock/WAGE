import numpy as np

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