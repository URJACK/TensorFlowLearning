import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import os


def get_model():
    inputs = keras.Input(shape=(128,))
    outputs = layers.Dense(1)(inputs)
    return keras.Model(inputs, outputs)


def main():
    model1 = get_model()
    model2 = get_model()
    model3 = get_model()

    inputs = keras.Input(shape=(128,))
    y1 = model1(inputs)
    y2 = model2(inputs)
    y3 = model3(inputs)
    outputs = layers.average([y1, y2, y3])
    ensemble_model = keras.Model(inputs=inputs, outputs=outputs, name="function_define_model")
    ensemble_model.summary()


if __name__ == '__main__':
    main()
