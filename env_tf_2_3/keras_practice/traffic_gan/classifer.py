import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from env_tf_2_3.keras_practice.traffic_gan.data_generator import DataGenerator

import pydot
from keras.utils.vis_utils import model_to_dot

import env_tf_2_3.keras_practice.traffic_gan.config as tganconfig

keras.utils.vis_utils.pydot = pydot
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

# 浮点数转整数通用误差
e = 0.000000001


# label
# ['normal.' 'buffer_overflow.' 'loadmodule.' 'perl.' 'neptune.'
# 'smurf.' 'guess_passwd.' 'pod.' 'teardrop.' 'portsweep.'
# 'ipsweep.' 'land.' 'ftp_write.' 'back.' 'imap.'
#  'satan.' 'phf.' 'nmap.' 'multihop.' 'warezmaster.'
#  'warezclient.' 'spy.' 'rootkit.']
def createClassifier(traffic_dim: int, class_num: int = 23) -> keras.Model:
    inputs = keras.Input(shape=(traffic_dim,))
    x = keras.layers.BatchNormalization()(inputs)
    x = keras.layers.Dense(512)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(128)(x)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Dense(class_num, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer=keras.optimizers.RMSprop(1e-4, 0.99),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=["acc"])
    return model


class Trainer(keras.callbacks.Callback):
    classifer: keras.Model
    losses: list
    acces: list
    counter: int

    def __init__(self, classifer: keras.Model):
        super().__init__()
        self.classifer = classifer
        self.losses = []
        self.acces = []
        self.counter = 0

    def on_batch_end(self, batch, logs=None):
        LOGTIME = 100
        if self.counter % LOGTIME == 0:
            self.classifer.save_weights(tganconfig.CLASSIFIER_MODELPATH)
            loss = logs.get('loss')
            acc = logs.get('acc')
            self.losses.append(loss)
            self.acces.append(acc)
            self.counter = 0
        self.counter = self.counter + 1

    def on_train_end(self, logs=None):
        losses = np.array(self.losses)
        acces = np.array(self.acces)
        plt.plot(losses, color='b', label='loss', linewidth=2)
        plt.plot(acces, color='r', label='accuracy', linewidth=2)
        plt.title('分类器训练')
        plt.ylabel('Y axis')
        plt.xlabel('X axis')
        plt.legend()  # 会出现每根线的label 在图的右上角
        plt.grid(True, color='k')
        plt.show()


def train():
    noise_dim: int = tganconfig.NOISE_DIM
    batch_size: int = tganconfig.BATCH_SIZE
    label_class_num: int = 23
    dataGenerator = DataGenerator(tganconfig.FILEPATH, batch_size, noise_dim)
    traffic_dim: int = dataGenerator.feature_dim

    classifer = createClassifier(traffic_dim, label_class_num)
    classifer.load_weights(tganconfig.CLASSIFIER_MODELPATH)
    trainner = Trainer(classifer)
    classifer.fit_generator(dataGenerator.classifier_iter(), steps_per_epoch=len(dataGenerator), epochs=3,
                            callbacks=[trainner])

    # for x, y in dataGenerator.classifier_iter():
    #     print(x)
    #     code = classifer(x)
    #     print(code)
    #     print(type(code))
    #     loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=code, labels=y))
    #     print(loss_function)
    #     data = loss_function.numpy()
    #     print(data)
    #     print(type(data))
    #     break


if __name__ == '__main__':
    train()
