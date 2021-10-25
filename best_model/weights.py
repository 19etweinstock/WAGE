import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# config.allow_soft_placement = True
# config.log_device_placement = False
# sess = tf.compat.v1.InteractiveSession(config=config)

# saver = tf.compat.v1.train.import_meta_graph('best_model/2021-10-14 2124(MNIST 11DD 16 [0, 1.1] 100 128 ).tf.meta')
# saver.restore(sess,tf.compat.v1.train.latest_checkpoint('best_model/./'))

# graph = tf.compat.v1.get_default_graph() 
# graph.get_collection('variables') 

def main():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.compat.v1.InteractiveSession(config=config)
    
    saver = tf.compat.v1.train.import_meta_graph('2021-10-14 2124(MNIST 11DD 16 [0, 1.1] 100 128 ).tf.meta')
    saver.restore(sess,tf.compat.v1.train.latest_checkpoint('./'))

    graph = tf.compat.v1.get_default_graph() 
    vars = graph.get_collection('variables') 
    var=graph.get_collection('variables')[3]
    tensor = var.value()
    print(tensor[1,1,1,1].eval())

    # print(sess.run())


if __name__ == '__main__':
  main()

