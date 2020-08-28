import numpy as np
import scipy as sp
from scipy import misc
import glob
import imageio
import keras.datasets as datasets
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import RMSprop
from keras.callbacks import Callback
from keras.initializers import RandomNormal
import os
import json
import warnings
import env_sjl.sjl.tool_img as tg

warnings.filterwarnings("ignore")  # 忽略keras带来的满屏警告

if not os.path.exists('samples'):
    os.mkdir('samples')

imgs = glob.glob('./structure/*.png')
print(imgs)
np.random.shuffle(imgs)
img_dim = 28
z_dim = 128
num_layers = int(np.log2(img_dim))
max_num_channels = img_dim * 8
f_size = img_dim // 2 ** (num_layers + 1)
batch_size = 64


def imread(f, mode='gan'):
    data = misc.imread(f, mode='RGB')
    if mode == 'gan':
        data = misc.imresize(data, (img_dim, img_dim))
        data = data.astype(np.float32)
        return data / 255 * 2 - 1
    elif mode == 'fid':
        data = misc.imresize(data, (299, 299))
        return data.astype(np.float32)


class img_generator:
    """图片迭代器，方便重复调用
    """

    def __init__(self, imgs, mode='gan', batch_size=64):
        self.imgs = imgs
        self.batch_size = batch_size
        self.mode = mode
        if len(imgs) % batch_size == 0:
            self.steps = len(imgs) // batch_size
        else:
            self.steps = len(imgs) // batch_size + 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        X = []
        while True:
            np.random.shuffle(self.imgs)
            for i, f in enumerate(self.imgs):
                X.append(imread(f, self.mode))
                if len(X) == self.batch_size or i == len(self.imgs) - 1:
                    X = np.array(X)
                    if self.mode == 'gan':
                        Z = np.random.randn(len(X), z_dim)
                        yield [X, Z], None
                    elif self.mode == 'fid':
                        yield X
                    X = []


# 判别器
x_in = Input(shape=(img_dim * img_dim,))
x = x_in
x = Dense(256, activation="relu")(x)
x = Dense(256, activation="relu")(x)
outputs = Dense(1, activation="tanh")(x)

d_model = Model(x_in, outputs)
d_model.summary()

# 生成器
z_in = Input(shape=(z_dim,), name="noise")

x = Dense(1024, activation="relu")(z_in)
x = Dense(1024, activation="relu")(x)
z_out = Dense(784, activation="tanh")(x)
model = Model(z_in, z_out, name="generator")

g_model = Model(z_in, z_out)
g_model.summary()

# 整合模型
x_in = Input(shape=(img_dim * img_dim,))
z_in = Input(shape=(z_dim,))

x_real = x_in
x_fake = g_model(z_in)
x_fake_ng = Lambda(K.stop_gradient)(x_fake)

x_real_score = d_model(x_real)
x_fake_score = d_model(x_fake)
x_fake_ng_score = d_model(x_fake_ng)

train_model = Model([x_in, z_in],
                    [x_real_score, x_fake_score, x_fake_ng_score])

d_loss = K.relu(1 + x_real_score) + K.relu(1 - x_fake_ng_score)
g_loss = x_fake_score - x_fake_ng_score

train_model.add_loss(K.mean(d_loss + g_loss))
train_model.compile(optimizer=RMSprop(1e-4, 0.99))

# 检查模型结构
train_model.summary()

# def on_batch_end(self, batch, logs=None):
# print(batch)
# if self.batch % self.iters_per_sample == 0:
#     fake_imgs = g_model.predict(batch_size, z_dim)
#     tg.display(fake_imgs)
#     train_model.save_weights('./model/ogan_model.weights')
# self.batch += 1


count = 0


def mnist_generator(trainX, trainY):
    global count
    global batch_size
    while True:
        batch_x = trainX[count * batch_size:(count + 1) * batch_size]
        batch_y = trainY[count * batch_size:(count + 1) * batch_size]
        count = count + 1
        yield count, batch_x, batch_y


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()

    for i, x, y in mnist_generator(train_x, train_y):
        z = np.random.randn(x.shape[0], z_dim)
        x = np.reshape(x, newshape=(x.shape[0], 28 * 28))
        y = []
        train_model.train_on_batch([x, z], y)
        fake_imgs = g_model.predict(z)
        tg.display(fake_imgs)
