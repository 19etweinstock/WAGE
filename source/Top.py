
from __future__ import print_function
from __future__ import division
from builtins import range
from past.utils import old_div
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 
import numpy as np
import time
import tensorflow as tf
import NN
import Option
import Log
import getData
import Quantize
import scipy.io as sio
from tqdm import tqdm
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

# for single GPU quanzation
def quantizeGrads(Grad_and_vars):
  if Quantize.bitsG <= 16:
    grads = []
    for grad_and_vars in Grad_and_vars:
      grads.append( [ Quantize.G(grad_and_vars[0]) , grad_and_vars[1] ])
    return grads
  return Grad_and_vars

def main():
  # get Option
  GPU = Option.GPU
  batchSize = Option.batchSize
  pathLog = '../log/' + Option.Time + '(' + Option.Notes + ')' + '.txt'
  Log.Log(pathLog, 'w+', 1) # set log file
  print(time.strftime('%Y-%m-%d %X', time.localtime()), '\n')
  print(open('Option.py').read())

  # get data
  numThread = 4*len(GPU)
  assert batchSize % len(GPU) == 0, ('batchSize must be divisible by number of GPUs')

  with tf.device('/cpu:0'):
    batchTrainX,batchTrainY,batchTestX,batchTestY,numTrain,numTest,label =\
        getData.loadData(Option.dataSet,batchSize,numThread,Option.validNum)

  batchNumTrain = old_div(numTrain, batchSize)
  batchNumTest = old_div(numTest, 100)

  optimizer = Option.optimizer
  global_step = tf.compat.v1.get_variable('global_step', [], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
  Net = []
  

  # for single GPU
  with tf.device('/gpu:%d' % GPU[0]):
    Net.append(NN.NN(batchTrainX, batchTrainY, training=True, global_step=global_step))
    lossTrainBatch, errorTrainBatch = Net[-1].build_graph()
    update_op = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)  # batchnorm moving average update ops
    update_op += Net[0].W_clip_op

    gradTrainBatch = quantizeGrads(optimizer.compute_gradients(lossTrainBatch))
    with tf.control_dependencies(update_op):
      train_op = optimizer.apply_gradients(gradTrainBatch, global_step=global_step)
    tf.compat.v1.get_variable_scope().reuse_variables()
    Net.append(NN.NN(batchTestX, batchTestY, training=False))
    _, errorTestBatch = Net[-1].build_graph()


  # Build an initialization operation to run below.
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  config.log_device_placement = False
  sess = Option.sess = tf.compat.v1.InteractiveSession(config=config)
  sess.run(tf.compat.v1.global_variables_initializer())
  saver = tf.compat.v1.train.Saver(max_to_keep=None)
  # Start the queue runners.
  tf.compat.v1.train.start_queue_runners(sess=sess)



  def getErrorTest():
    errorTest = 0.
    for i in tqdm(range(batchNumTest),desc = 'Test', leave=False):
      errorTest += sess.run([errorTestBatch])[0]
    errorTest /= batchNumTest
    return errorTest

  Option.eval = True
  if Option.loadModel is not None:
    print('Loading model from %s ...' % Option.loadModel, end=' ')
    saver.restore(sess, Option.loadModel)
    print('Finished', end=' ')
    errorTestBest = getErrorTest()
    print('Test: %.4f ' % (errorTestBest))

  sess.run([Net[0].W_q_op])


  # f = open("weights_top.py", "wt")
  # f.write("import numpy as np\n\n")
  # f.flush()

  # for var in range(0,2):
  #     tensor=Net[-1].W_q[var]
  #     for in_filter in range(0, tensor.shape.as_list()[2]):
  #         for out_filter in range(0,tensor.shape.as_list()[3]):
  #             f.write(f"conv{var}_in{in_filter}_out{out_filter} = np.array([\n")
  #             quant_array = Net[-1].W_q[var].eval(session=sess)
  #             for row in range(0,5):
  #                 f.write("\t[")
  #                 for col in range(0,5):
  #                     f.write(f"{quant_array[row,col,in_filter,out_filter]}{', ' if col != 4 else ''}")
  #                 f.write("]")
  #                 if (row != 4):
  #                     f.write(",\n")
                  
  #             f.write("])\n\n")
  #             f.flush()
  #     for out_filter in range(0, tensor.shape.as_list()[3]):
  #         f.write(f"weights_conv{var}_out{out_filter} = np.array([\n")
  #         for in_filter in range(0, tensor.shape.as_list()[2]):
  #             f.write(f"\tconv{var}_in{in_filter}_out{out_filter}")
  #             if (in_filter != tensor.shape.as_list()[2] - 1):
  #                 f.write(",\n")
  #             else:
  #                 f.write("])\n\n")
      
  #     f.write(f"weights_conv{var} = np.array([\n")
  #     for out_filter in range(0, tensor.shape.as_list()[3]):
  #         f.write(f"\tweights_conv{var}_out{out_filter}")
  #         if (out_filter != tensor.shape.as_list()[3] - 1):
  #             f.write(",\n")
  #         else:
  #             f.write("])\n\n")
  # for var in range(2,5):
  #     tensor=Net[-1].W_q[var]
  #     quant_array = Net[-1].W_q[var].eval(session=sess)
  #     f.write(f"fc{var-2} = np.array([\n")
  #     rows=tensor.shape.as_list()[0]
  #     cols=tensor.shape.as_list()[1]
  #     for row in range(0,rows):
  #         f.write("\t[")
  #         for col in range(0, cols):
  #             f.write(f"{quant_array[row,col]}{', ' if col != (cols-1) else ''}")
  #         f.write("]")
  #         if (row != (rows -1)):
  #             f.write(",\n")
          
  #     f.write("])\n\n")
  #     f.flush()
          
  # f.close()

  print("\nOptimization Start!\n")
  for epoch in range(Option.Epoch):

    # check lr_schedule
    if old_div(len(Option.lr_schedule), 2):
      if epoch == Option.lr_schedule[0]:
        Option.lr_schedule.pop(0)
        lr_new = Option.lr_schedule.pop(0)
        if lr_new == 0:
          print('Optimization Ended!')
          exit(0)
        lr_old = sess.run(Option.lr)
        sess.run(Option.lr.assign(lr_new))
        print('lr: %f -> %f' % (lr_old, lr_new))

    print('Epoch: %03d ' % (epoch), end=' ')

    lossTotal = 0.
    errorTotal = 0
    t0 = time.time()
    for batchNum in tqdm(range(batchNumTrain),desc = 'Epoch: %03d'%epoch, leave=False, smoothing=0.1):
      _, loss_delta, error_delta = sess.run([train_op, lossTrainBatch, errorTrainBatch])
      # _, loss_delta, error_delta, H, W, W_q, gradsH, gradsW, gradW_q=\
      # sess.run([train_op, lossTrainBatch, errorTrainBatch, Net[0].H, Net[0].W, Net[0].W_q, Net[0].gradsH, Net[0].gradsW, gradTrainBatch])

      lossTotal += loss_delta
      errorTotal += error_delta

    lossTotal /= batchNumTrain
    errorTotal /= batchNumTrain

    print('Loss: %.6f Train: %.4f' % (lossTotal, errorTotal), end=' ')

    # get test error
    errorTest = getErrorTest()
    print('Test: %.4f FPS: %d' % (errorTest,old_div(numTrain, (time.time() - t0))), end=' ')


    if epoch == 0:
      errorTestBest = errorTest
      errorTrainBest = errorTotal

    if errorTest < errorTestBest:
      errorTestBest = errorTest
      if Option.saveModel is not None:
        saver.save(sess, Option.saveModel)
        print('S', end=' ')
      print('BEST', end=' ')
    elif (errorTest == errorTestBest and errorTrainBest < errorTotal):
      errorTestBest = errorTest
      errorTrainBest = errorTotal
      if Option.saveModel is not None:
        saver.save(sess, Option.saveModel)
        print('S', end=' ')
      print('BEST', end=' ')

    print('')


if __name__ == '__main__':
  main()

