import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

import Quantize


def main():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.compat.v1.InteractiveSession(config=config)

    saver = tf.compat.v1.train.import_meta_graph('../model/2021-10-25 1108(MNIST 11DD 16 [0, 1.1] 100 128 ).tf.meta')
    saver.restore(sess,tf.compat.v1.train.latest_checkpoint('../model/'))

    graph = tf.compat.v1.get_default_graph() 
    vars = graph.get_collection('variables') 

    f = open("weights.py", "wt")
    f.write("import numpy as np\n\n")
    
    tensor=vars[2].value()

    for filter in range(0,6):
        f.write(f"conv_filter_{filter} = np.array([[\n")
        for row in range(0,5):
            f.write("\t[")
            for col in range(0,5):
                f.write(f"[{Quantize.W(tensor[row,col,0,filter].eval())}]{', ' if col != 4 else ''}")
            f.write("],")
            if (row != 4):
                f.write("\n")
        f.write(" ]])\n\n")


if __name__ == '__main__':
  main()

