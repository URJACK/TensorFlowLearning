import keras
import keras.layers as layers
import os
import numpy as np

import pydot
from keras.utils.vis_utils import model_to_dot

keras.utils.vis_utils.pydot = pydot


def main():
    inputs = keras.Input(shape=(784,))

    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)

    model = keras.Model(inputs, outputs, name="mnist_model")
    model.summary()

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=["accuracy"])

    draw_path = os.path.abspath(os.path.dirname(__file__)) + "\\structure\\demo.png"
    keras.utils.plot_model(model, draw_path, show_shapes=True)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255

    history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

    test_scores = model.evaluate(x_test, y_test)
    print("Test Loss ", test_scores[0])
    print("Test Accuracy ", test_scores[1])

    model_path = os.path.abspath(os.path.dirname(__file__)) + "\\model\\demo"
    model.save(model_path)
    print(model.to_json())
    del model

    # Recreate the exact same model purely from the file:
    model = keras.models.load_model(model_path)

    test_scores = model.evaluate(x_test, y_test)
    print("Test Loss ", test_scores[0])
    print("Test Accuracy ", test_scores[1])


if __name__ == '__main__':
    main()
