import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras import utils as kutils
import re
import csv
import os
import pandas
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from matplotlib import style
from env_tf_2_3.keras_practice.traffic_gan.data_generator import DataGenerator

import env_tf_2_3.keras_practice.traffic_gan.config as tganconfig
import env_tf_2_3.keras_practice.traffic_gan.classifer as classifer
import pydot
from keras.utils.vis_utils import model_to_dot

keras.utils.vis_utils.pydot = pydot
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

FILEPATH = tganconfig.FILEPATH
MODELPATH = tganconfig.TRAFFICGAN_MODELPATH
NOISE_DIM = tganconfig.NOISE_DIM
BATCH_SIZE = tganconfig.BATCH_SIZE


def correlation(x, y):
    x = x - K.mean(x, 1, keepdims=True)
    y = y - K.mean(y, 1, keepdims=True)
    x = K.l2_normalize(x, 1)
    y = K.l2_normalize(y, 1)
    return K.sum(x * y, 1, keepdims=True)


def createTrafficGANGenerator(traffic_dim: int, noise_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(noise_dim,))
    x = keras.layers.BatchNormalization()(inputs)
    x = keras.layers.Dense(512)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(512)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Dense(traffic_dim, activation='tanh')(x)
    generator = keras.Model(inputs, outputs)
    return generator


def createTrafficGANEncoder(traffic_dim: int, noise_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(traffic_dim,))
    x = keras.layers.BatchNormalization()(inputs)
    x = keras.layers.Dense(512)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(512)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Dense(noise_dim)(x)
    decoder = keras.Model(inputs, outputs)
    return decoder


def createTrafficGAN(g_model: keras.Model, e_model: keras.Model, traffic_dim: int, noise_dim: int) -> keras.Model:
    traffic_in = keras.Input(shape=(traffic_dim,))
    noise_in = keras.Input(shape=(noise_dim,))
    traffic_generate = g_model(noise_in)
    traffic_generate_ng = keras.layers.Lambda(K.stop_gradient)(traffic_generate)

    code_traffic = e_model(traffic_in)
    code_generate = e_model(traffic_generate)
    code_generate_ng = e_model(traffic_generate_ng)

    trafficGan = keras.Model([traffic_in, noise_in], [code_traffic, code_generate, code_generate_ng])

    code_traffic_mean = K.mean(code_traffic, 1, keepdims=True)
    code_generate_mean = K.mean(code_generate, 1, keepdims=True)
    code_generate_ng_mean = K.mean(code_generate_ng, 1, keepdims=True)

    t1_loss = code_traffic_mean - code_generate_ng_mean
    t2_loss = code_generate_mean - code_generate_ng_mean
    z_corr = correlation(noise_in, code_generate)
    qp_loss = 0.25 * t1_loss[:, 0] ** 2 / K.mean((traffic_in - traffic_generate_ng) ** 2, axis=1)

    trafficGan.add_loss(K.mean(t1_loss + t2_loss - 0.5 * z_corr) + K.mean(qp_loss))
    trafficGan.compile(optimizer=keras.optimizers.RMSprop(1e-4, 0.99))

    return trafficGan


class Trainer(keras.callbacks.Callback):
    trafficGan: keras.Model
    g_model: keras.Model
    classifer_model: keras.Model
    losses: list

    def __init__(self, trafficGan: keras.Model, g_model: keras.Model, classifer_model: keras.Model, noise_dim: int,
                 batch_size: int = 64, class_num: int = 23):
        super().__init__()
        self.trafficGan = trafficGan
        self.classifer_model = classifer_model
        self.g_model = g_model
        self.losses = []
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.class_num = class_num
        self.counter = 0

    def on_batch_end(self, batch, logs=None):
        LOGTIME = 500
        if self.counter % LOGTIME == 0:
            loss: float = self.testTrain()
            # loss = logs.get('loss')
            self.losses.append(loss)
            self.counter = 0
        self.counter = self.counter + 1

    def on_epoch_end(self, epoch, logs=None):
        self.trafficGan.save_weights(MODELPATH)
        # loss: float = self.testTrain()
        # # loss = logs.get('loss')
        # self.losses.append(loss)
        return

    def on_train_end(self, logs=None):
        losses = np.array(self.losses)
        style.use('ggplot')
        plt.plot(losses, color='b', label='t_loss', linewidth=3)
        plt.title('loss')
        plt.ylabel('Y axis')
        plt.xlabel('X axis')
        plt.legend()  # 会出现每根线的label 在图的右上角
        plt.grid(True, color='k')
        plt.show()

    def testTrain(self) -> float:
        # 生成噪声，并利用噪声生成流量
        noise = np.random.randn(self.batch_size, self.noise_dim)
        generated_traffic = self.g_model.predict(noise)
        # 使用判别器，判别网络所属类别的期望
        g_traffic_type: np.ndarray = self.classifer_model.predict(generated_traffic)
        # 求得所属期望最大的index，并根据该index生成对应的label
        expected_index = np.argmax(g_traffic_type, axis=1)
        # labels = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0....]
        # g_traffic_type = [0.03,0.03,0.03,0.03,0.03,0.6,0.05......]
        labels = to_categorical(expected_index, num_classes=self.class_num)
        g_traffic_type = tf.Variable(g_traffic_type)
        labels = tf.Variable(labels)
        # print(g_traffic_type)
        # print(g_traffic_type.shape)
        # print(labels)
        # print(labels.shape)
        # 使用label 与 判别得到的类别做交叉熵
        loss_function = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=g_traffic_type,
                                                    labels=labels))
        loss = loss_function.numpy()
        loss = loss.tolist()
        # print(loss)
        return loss


def train():
    noise_dim: int = NOISE_DIM
    batch_size: int = BATCH_SIZE
    dataGenerator = DataGenerator(FILEPATH, batch_size, noise_dim)
    traffic_dim: int = dataGenerator.feature_dim

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        g_model = createTrafficGANGenerator(traffic_dim, noise_dim)
        e_model = createTrafficGANEncoder(traffic_dim, noise_dim)

        classifer_model = classifer.createClassifier(traffic_dim, class_num=23)
        classifer_model.load_weights(tganconfig.CLASSIFIER_MODELPATH)

        trafficGAN = createTrafficGAN(g_model, e_model, traffic_dim, noise_dim)
        trafficGAN.load_weights(MODELPATH)
        trainner = Trainer(trafficGAN, g_model, classifer_model, noise_dim)
        trafficGAN.fit_generator(dataGenerator.gan_iter(), steps_per_epoch=len(dataGenerator), epochs=10,
                                 callbacks=[trainner])


def showModel(e_model, g_model, trafficGAN):
    g_model.summary()
    e_model.summary()
    trafficGAN.summary()
    draw_path = os.path.abspath(os.path.dirname(__file__)) + "\\structure\\trafficGan.png"
    keras.utils.plot_model(trafficGAN, draw_path, show_shapes=True)
    draw_path = os.path.abspath(os.path.dirname(__file__)) + "\\structure\\trafficGan_g.png"
    keras.utils.plot_model(g_model, draw_path, show_shapes=True)
    draw_path = os.path.abspath(os.path.dirname(__file__)) + "\\structure\\trafficGan_e.png"
    keras.utils.plot_model(e_model, draw_path, show_shapes=True)


if __name__ == '__main__':
    train()
