LR    = 1
bitsW = 1  # bit width of weights
bitsA = 2 # bit width of activations
bitsG = 8  # bit width of gradients
bitsE = 8 # bit width of errors

bitsR = 8  # bit width of randomizer

loops = 8
Epoch = 50
batchSize = 50

lr_schedule = [0, 1., 20, 0.5, 35, 0.25]

# loadModel = None
loadModel = '2022-01-21 1844(MNIST 1288 8 [0, 1.0, 20, 0.5, 35, 0.25] 50 50)'
