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
    saver.restore(sess,tf.compat.v1.train.latest_checkpoint('../model/./'))

    graph = tf.compat.v1.get_default_graph() 
    vars = graph.get_collection('variables') 

    f = open("weights.py", "wt")
    f.write("import numpy as np\n\n")
    f.flush()
    
    for var in range(3,5):
        tensor=vars[var].value()
        for in_filter in range(0, tensor.shape.as_list()[2]):
            for out_filter in range(0,tensor.shape.as_list()[3]):
                f.write(f"conv{var-3}_in{in_filter}_out{out_filter} = np.array([\n")
                quant = Quantize.W(tensor)
                quant_array = quant.eval(session=sess)
                for row in range(0,5):
                    f.write("\t[")
                    for col in range(0,5):
                        f.write(f"{quant_array[row,col,in_filter,out_filter]}{', ' if col != 4 else ''}")
                    f.write("]")
                    if (row != 4):
                        f.write(",\n")
                    
                f.write("])\n\n")
                f.flush()
        for out_filter in range(0, tensor.shape.as_list()[3]):
            f.write(f"weights_conv{var-3}_out{out_filter} = np.array([\n")
            for in_filter in range(0, tensor.shape.as_list()[2]):
                f.write(f"\tconv{var-3}_in{in_filter}_out{out_filter}")
                if (in_filter != tensor.shape.as_list()[2] - 1):
                    f.write(",\n")
                else:
                    f.write("])\n\n")
        
        f.write(f"weights_conv{var-3} = np.array([\n")
        for out_filter in range(0, tensor.shape.as_list()[3]):
            f.write(f"\tweights_conv{var-3}_out{out_filter}")
            if (out_filter != tensor.shape.as_list()[3] - 1):
                f.write(",\n")
            else:
                f.write("])\n\n")
    for var in range(5,8):
        tensor=vars[var].value()
        quant = Quantize.W(tensor)
        quant_array = quant.eval(session=sess)
        f.write(f"fc{var-5} = np.array([\n")
        rows=tensor.shape.as_list()[0]
        cols=tensor.shape.as_list()[1]
        for row in range(0,rows):
            f.write("\t[")
            for col in range(0, cols):
                f.write(f"{quant_array[row,col]}{', ' if col != (cols-1) else ''}")
            f.write("]")
            if (row != (rows -1)):
                f.write(",\n")
            
        f.write("])\n\n")
        f.flush()
            
    f.close()


if __name__ == '__main__':
  main()

