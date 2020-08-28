import keras
import keras.layers as layers
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import glob
import pydot
import pickle
import imageio
from keras import utils as kutils
from keras.utils.vis_utils import model_to_dot
from keras import backend as K

keras.utils.vis_utils.pydot = pydot

MODELPATH = './model/vgg_net.weights'


class DataGenerator:
    # 此处要求所有的单个 "数据集文件的长度" 都是 "一样" 的
    def __init__(self, folder_pattern_jpg, batch_size):
        self.batch_size = batch_size
        data_names = glob.glob(folder_pattern_jpg)  # 读取数据集文件名称
        np.random.shuffle(data_names)  # 打乱数据集文件名称的顺序
        self.data_names = data_names  # 将获取到的数据集文件名称中放入成员变量，方便后续的迭代器调用
        obj = unpickle(data_names[0])  # （关键）从单个文件中，解析出数据
        self.per_set_len = len(obj[b'labels'])  # 提取出labels，因为labels是python-list类型，可以直接用len获取到样本个数
        self.sets_num = len(self.data_names)  # 获取到数据集一共有sets_num个
        self.all_sets_len = self.per_set_len * self.sets_num  # 得到总的数据集个数
        # 规划训练的总步数
        self.steps = self.per_set_len // batch_size
        # 如果没有整除，需要额外加上一步
        if self.per_set_len % batch_size != 0:
            self.steps = self.steps + 1
        # 单个数据集需要走的步数，变为所有数据集需要的步数
        self.steps = self.steps * self.sets_num

    def __len__(self):
        return self.steps

    @staticmethod
    def x_preprocess(x):
        # x = np.reshape(x, (-1, 3, 1024))
        # x = x.T  # 1024 x 3
        # x = np.reshape(x, (-1, 32, 32, 3))

        # 非常重要，必须采用下面这种预处理的办法
        x = np.reshape(x, (-1, 3, 32, 32))
        x = x.transpose(0, 2, 3, 1)
        return x / 255 * 2 - 1

    @staticmethod
    def y_preprocess(y):
        return kutils.to_categorical(y)

    def __iter__(self):
        while True:
            for (i, f) in enumerate(self.data_names):
                obj = unpickle(f)
                labels = obj[b'labels']  # 获取到b'obj'对象
                datas = obj[b'data']  # 获取到b'data'对象
                labels = np.array(labels)  # 将 labels 转为 ndarray 类型
                datas = self.x_preprocess(datas)  # 对datas 进行归一化
                labels = self.y_preprocess(labels)  # 将labels 进行 one-hot 编码
                start = 0
                end = self.batch_size
                while end <= self.per_set_len:
                    X = datas[start:end]
                    Y = labels[start:end]
                    yield X, Y
                    start = start + self.batch_size
                    end = end + self.batch_size
                # 调整越界的end
                if start < self.per_set_len:
                    yield datas[start:self.per_set_len], labels[start:self.per_set_len]


def unpickle(file):
    with open(file, 'rb') as fo:
        obj = pickle.load(fo, encoding='bytes')
    return obj


def createVGGBlock(inputs: keras.Input, num_convs, out_channel):
    x: keras.Model
    x = keras.layers.Conv2D(out_channel, (3, 3), (1, 1), padding="SAME")(inputs)
    for i in range(num_convs - 1):
        x = keras.layers.Conv2D(out_channel, (3, 3), (1, 1), padding="SAME")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)
    outputs = keras.layers.BatchNormalization()(x)
    return outputs


def createVGGStack(inputs: keras.Input):
    x = createVGGBlock(inputs, 2, 64)
    x = createVGGBlock(x, 2, 128)
    x = createVGGBlock(x, 3, 256)
    x = createVGGBlock(x, 3, 512)
    x = createVGGBlock(x, 3, 512)
    return x


def fc(inputs: keras.Input, dense):
    x = keras.layers.Dense(dense, kernel_initializer=keras.initializers.RandomNormal(0, 0.02))(inputs)
    x = keras.layers.BatchNormalization()(x)
    return x


def createVGG(height, width, channel) -> keras.Model:
    inputs = keras.Input(shape=(height, width, channel))
    x = createVGGStack(inputs)
    x = keras.layers.Flatten()(x)
    x = fc(x, 4096)
    x = fc(x, 100)
    outputs = fc(x, 10)
    model = keras.Model(inputs, outputs)
    return model


def compileVGG(vggNet: keras.Model):
    vggNet.compile(optimizer=keras.optimizers.RMSprop(1e-4, 0.99),
                   loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                   metrics=["acc"])


class Trainner(keras.callbacks.Callback):
    vggNet: keras.Model
    losses: list
    accs: list

    def __init__(self, vggNet):
        print("Trainner Init")
        self.vggNet = vggNet
        self.losses = []
        self.accs = []

    def on_epoch_end(self, epoch, logs=None):
        print("Epoch END")
        self.vggNet.save_weights(MODELPATH)
        loss = logs.get('loss')
        acc = logs.get('acc')
        self.losses.append(loss)
        self.accs.append(acc)

    def on_train_end(self, logs=None):
        losses = np.array(self.losses)
        accs = np.array(self.accs)
        style.use('ggplot')
        plt.plot(losses, color='b', label='losses', linewidth=3)
        plt.plot(accs, color='r', label='accuracy', linewidth=4)
        plt.title('losses and accuracy')
        plt.ylabel('values')
        plt.xlabel('trainning times')
        plt.legend()  # 会出现每根线的label 在图的右上角
        plt.grid(True, color='k')
        plt.show()


def train():
    dataGenerator = DataGenerator('D:\\Storage\\datasets\\cifar-10-batches-py\\*.data', 64)
    vggNet: keras.Model = createVGG(32, 32, 3)
    vggNet.summary()
    compileVGG(vggNet)

    trainner = Trainner(vggNet)
    vggNet.load_weights(MODELPATH)
    vggNet.fit_generator(dataGenerator.__iter__(), steps_per_epoch=len(dataGenerator), epochs=2, callbacks=[trainner])


def mapping(index):
    arr = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
    return arr[index]


def use():
    dim = 32
    height = 8
    width = 8
    vggNet: keras.Model = createVGG(32, 32, 3)
    vggNet.load_weights(MODELPATH)
    dataGenerator = DataGenerator('D:\\Storage\\datasets\\cifar-10-batches-py\\*.data', height * width)
    for (datas, labels) in dataGenerator:
        figure = np.zeros((height * dim, width * dim, 3))
        ans_list: list[str] = []
        for i in range(height):
            for j in range(width):
                data = datas[i * width + j]
                figure[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim] = (data + 1) / 2 * 255
                value = vggNet.predict(data.reshape(1, dim, dim, 3))
                value = np.argmax(value, axis=1)[0]
                value = mapping(value)
                ans_list.append(value)
        index: int = 0
        printValue: str = ""
        for value in ans_list:
            index = index + 1
            printValue = printValue + value + "\t"
            if index == width:
                print(printValue)
                printValue = ""
                index = 0
        imageio.imwrite('samples/a.png', figure)
        break


def main():
    # train()
    use()


if __name__ == '__main__':
    main()
