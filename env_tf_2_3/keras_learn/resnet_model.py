import keras
import keras.layers as layers
import os
import numpy as np

import pydot
from keras.utils.vis_utils import model_to_dot

keras.utils.vis_utils.pydot = pydot


class ResNetModel:
    _model = None

    def __init__(self):
        inputs = keras.Input(shape=(32, 32, 3), name="img")
        x = layers.Conv2D(32, 3, activation="relu")(inputs)
        x = layers.Conv2D(64, 3, activation="relu")(x)
        block_1_output = layers.MaxPooling2D(3)(x)

        x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
        x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        block_2_output = layers.add([x, block_1_output])

        x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
        x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        block_3_output = layers.add([x, block_2_output])

        x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(10)(x)

        self._model = keras.Model(inputs, outputs, name="toy_resnet")

    def print(self):
        draw_path = os.path.abspath(os.path.dirname(__file__)) + "\\structure\\resnet.png"
        keras.utils.plot_model(self._model, draw_path, show_shapes=True)

    def compile(self):
        self._model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                            metrics=["acc"])

    def train(self, x_train, y_train):
        self._model.fit(x_train, y_train, batch_size=64, epochs=1, validation_split=0.2)


def create_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


def main():
    model = ResNetModel()
    model.print()
    model.compile()
    datas = create_data()
    model.train(datas[0][:2000], datas[1][:2000])


if __name__ == '__main__':
    main()
