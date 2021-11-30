import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils


import Layers, Quantize

class lenet5(tf.keras.Model):
    def __init__(self, fmaps0=6, fmaps1=8, _fc0 = 120, _fc1 = 84) -> None:
        super().__init__()
        self.input_filter = Layers.qa()
        self.conv0 = Layers.qconv2d(5, fmaps0)
        self.pool0 = Layers.maxpool()
        self.activation0 = Layers.qactivation()

        self.conv1 = Layers.qconv2d(5, fmaps1)
        self.pool1 = Layers.maxpool()
        self.activation1 = Layers.qactivation()

        self.reshape=Layers.reshape()

        self.fc0 = Layers.qfc(_fc0)
        self.activation2 = Layers.qactivation()
        self.fc1 = Layers.qfc(_fc1)
        self.activation3 = Layers.qactivation()
        self.fc2 = Layers.qfc(10)

        self.output_filter = Layers.qsoftmax()

    def call(self, input, training=False):
        x = tf.stop_gradient(self.input_filter(input))
        x = self.conv0(x)
        x = self.pool0(x)
        x = self.activation0(x)

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.activation1(x)

        x = self.reshape(x)

        x = self.fc0(x)
        x = self.activation2(x)
        x = self.fc1(x)
        x = self.activation3(x)
        x = self.fc2(x)

        x = self.output_filter(x)

        return x

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # if our prediction is perfect and loss is 0, gradient is nan
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        for i in range(0,len(gradients)):
            gradients[i] = Quantize.G(gradients[i])

        # Compute gradients
        # Update weights
        # do not update if loss is 0
        # if any gradients are nan do nothing

        # if any gradients are 0, it will be the ones for the first layer (last calculated)
        tf.cond(tf.math.reduce_any(tf.math.is_nan(gradients[0])), 
                false_fn = lambda : self.optimizer.apply_gradients(zip(gradients, trainable_vars)),
                true_fn= tf.no_op)
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}