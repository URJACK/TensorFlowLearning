import keras
import keras.layers as layers
import tensorflow as tf
import os
import numpy as np

import pydot
from keras.utils.vis_utils import model_to_dot

keras.utils.vis_utils.pydot = pydot


class CustomDense(layers.Layer):
    def __init__(self, units=32):
        super(CustomDense, self).__init__()
        self.units = units
        self.w = None
        self.b = None

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        return {"units": self.units}


def main():
    inputs = keras.Input((4,))
    outputs = CustomDense(10)(inputs)
    model = keras.Model(inputs, outputs)

    draw_path = os.path.abspath(os.path.dirname(__file__)) + "\\structure\\custom_dense.png"
    keras.utils.plot_model(model, draw_path, show_shapes=True)
    model.summary()


if __name__ == '__main__':
    main()
