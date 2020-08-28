import keras
import keras.layers as layers
import tensorflow as tf
import os
import numpy as np

import pydot
from keras.utils.vis_utils import model_to_dot

keras.utils.vis_utils.pydot = pydot


class MLP(keras.Model):

    def get_config(self):
        pass

    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.dense_1 = layers.Dense(64, activation='relu')
        self.dense_2 = layers.Dense(10)

    def call(self, inputs):
        x = self.dense_1(inputs)
        return self.dense_2(x)

    def print(self):
        self.summary()
        draw_path = os.path.abspath(os.path.dirname(__file__)) + "\\structure\\custom_model.png"
        keras.utils.plot_model(self, draw_path)


def main():
    # Instantiate the model.
    mlp = MLP()
    # _ = mlp(tf.zeros((1, 32))) # 也是可以的
    _ = mlp(keras.Input(shape=(784,)))
    mlp.print()


if __name__ == '__main__':
    main()
