import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 
import tensorflow as tf
import Option

LR = Option.LR
bitsW = Option.bitsW
bitsA = Option.bitsA
bitsE = Option.bitsE
bitsG = Option.bitsG

bitsR = Option.bitsR

W_scale = []

def S(bits):
  return 2.0 ** (bits - 1)

def Shift(x):
  return 2 ** tf.round(tf.math.log(x) / tf.math.log(2.0))

def C(x, bits=32):
  if bits > 15 or bits == 1:
    delta = 0.
  else:
    delta = 1. / S(bits)
  MAX = +1 - delta
  MIN = -1 + delta
  x = tf.clip_by_value(x, MIN, MAX, name='saturate')
  return x

def Q(x, bits):
  if bits > 15:
    return x
  elif bits == 1:  # BNN
    return tf.sign(x)
  else:
    SCALE = S(bits)
    return tf.round(x * SCALE) / SCALE

def W(x,scale = 1.0):
    # print("W")
    with tf.name_scope('QW'):
        y = Q(C(x, bitsW), bitsW)
        if scale > 1.8:
            y = y / scale
        return x + tf.stop_gradient(y - x)  # skip derivation of Quantize and Clip

def A(x):
#   print("A")
  with tf.name_scope('QA'):
    x = C(x, bitsA)
    y = Q(x, bitsA)
    return x + tf.stop_gradient(y - x)  # skip derivation of Quantize, but keep Clip

def G(x):
#   print("G")
  with tf.name_scope('QG'):
    if bitsG > 15:
      return x
    else:
      xmax = tf.reduce_max(tf.abs(x))
      x = x / Shift(xmax)

      norm = Q(LR * x , bitsR)
      norm_sign = tf.sign(norm)
      norm_abs = tf.abs(norm)
      norm_int = tf.floor(norm_abs)
      norm_float = norm_abs - norm_int
      rand_float = tf.random.uniform(x.shape, 0, 1)
      norm = norm_sign * ( norm_int + 0.5 * (tf.sign(norm_float - rand_float) + 1) )

      return norm / S(bitsG)

@tf.custom_gradient
def E(x):
  with tf.name_scope('QE'):
    def grad(dy):
      if bitsE > 15:
        return dy
      else:
        ymax = tf.reduce_max(tf.abs(dy))
        ymax_shift = Shift(ymax)
        return Q(C( dy /ymax_shift, bitsE), bitsE)
    return x, grad
