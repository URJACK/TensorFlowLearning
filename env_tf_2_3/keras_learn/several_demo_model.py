import keras
import keras.layers as layers
import os
import numpy as np

import pydot
from keras.utils.vis_utils import model_to_dot

keras.utils.vis_utils.pydot = pydot

num_tags = 12
num_words = 10000
num_departments = 4


class SeveralModel:
    _model = None

    def __init__(self):
        title_input = keras.Input(shape=(None,), name="title")
        body_input = keras.Input(shape=(None,), name="body")
        tags_input = keras.Input(shape=(num_tags,), name="tags")

        title_features = layers.Embedding(num_words, 64, name="title_emb")(title_input)
        body_features = layers.Embedding(num_words, 64, name="body_emb")(body_input)

        title_features = layers.LSTM(128)(title_features)
        body_features = layers.LSTM(32)(body_features)

        x = layers.concatenate([title_features, body_features, tags_input])

        priority_pred = layers.Dense(1, name="priority")(x)
        department_pred = layers.Dense(num_departments, name="department")(x)

        model = keras.Model(inputs=[title_input, body_input, tags_input], outputs=[priority_pred, department_pred])

        self._model = model

    def print(self):
        self._model.summary()
        draw_path = os.path.abspath(os.path.dirname(__file__)) + "\\structure\\several_demo_model.png"
        keras.utils.plot_model(self._model, draw_path, show_shapes=True)

    def compile(self):
        self._model.compile(optimizer=keras.optimizers.RMSprop(1e-3), loss=[
            keras.losses.BinaryCrossentropy(from_logits=True),
            keras.losses.CategoricalCrossentropy(from_logits=True)
        ], loss_weights=[1.0, 0.2])

    def train(self, title_data, body_data, tags_data, priority_targets, dept_targets):
        self._model.fit({
            'title': title_data, "body": body_data, "tags": tags_data
        }, {
            "priority": priority_targets, "department": dept_targets
        }, epochs=2, batch_size=32)

    def getModel(self):
        return self._model


def create_data():
    title_data = np.random.randint(num_words, size=(1280, 10))
    body_data = np.random.randint(num_words, size=(1280, 100))
    tags_data = np.random.randint(2, size=(1280, num_tags)).astype("float32")

    # Dummy target data
    priority_targets = np.random.random(size=(1280, 1))
    dept_targets = np.random.randint(2, size=(1280, num_departments))
    return title_data, body_data, tags_data, priority_targets, dept_targets


def main():
    serveral_model = SeveralModel()
    serveral_model.print()
    serveral_model.compile()
    datas = create_data()
    # serveral_model.train(datas[0], datas[1], datas[2], datas[3], datas[4])
    model = serveral_model.getModel()
    print(datas[0].shape)
    print(datas[1].shape)
    print(datas[2].shape)
    print(model([datas[0], datas[1], datas[2]]))


if __name__ == '__main__':
    main()
