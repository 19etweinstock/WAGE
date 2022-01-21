LR    = 1
bitsW = 15  # bit width of weights
bitsA = 15 # bit width of activations
bitsG = 15  # bit width of gradients
bitsE = 15 # bit width of errors

bitsR = 15  # bit width of randomizer

loops = 4
Epoch = 50
batchSize = 50

lr_schedule = [0, 1., 20, 0.7, 35, 0.2]

loadModel = None
# loadModel = '2021-12-03 1640(MNIST 1288 8 [0, 1.0, 20, 0.7, 35, 0.2] 50 50)'
