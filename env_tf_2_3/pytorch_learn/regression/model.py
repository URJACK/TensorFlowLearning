import pandas as pd
import numpy as np
import math

TRAIN_DATAPATH = './data/train.csv'
MODELPATH = './model/weight.npz'


def data_process_x_y(month_data):
    # 9个小时的数据(包括PM2.5)作为输入，第10个小时的PM2.5为输出
    # 类似于卷积的操作，对480个小时的数据均进行这样的操作，最终可凑够471组数据
    x = np.empty([12 * 471, 18 * 9], dtype=float)
    y = np.empty([12 * 471, 1], dtype=float)
    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    break
                beginIndex = day * 24 + hour
                x[month * 471 + beginIndex, :] = month_data[month][:, beginIndex:beginIndex + 9].reshape(1, -1)
                y[month * 471 + beginIndex, 0] = month_data[month][9, beginIndex + 9]
    return x, y


def data_process_monthdata(raw_data):
    month_data = {}
    DAY_DIM = 24
    DAY_SPAN = 18
    MONTH_DAYS = 20
    MONTH_SPAN = DAY_SPAN * MONTH_DAYS  # 360
    for month in range(12):
        # 一个月的数据
        # 18 x 24 为一天的数据，这里是 18 x 480 ，一个月一共是20天的数据
        sample = np.empty([DAY_SPAN, MONTH_DAYS * DAY_DIM])
        for day in range(MONTH_DAYS):
            beginIndex = month * MONTH_SPAN + day * DAY_SPAN
            sample[:, day * DAY_DIM: (day + 1) * DAY_DIM] = raw_data[beginIndex:beginIndex + DAY_SPAN, :]
        month_data[month] = sample
    return month_data


def data_process_normalize(x: np.ndarray):
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    print(mean_x.shape)
    print(std_x.shape)
    for i in range(len(x)):
        for j in range(len(x[0])):
            if std_x[j] != 0:
                x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
    return x


def data_process_split(data: np.ndarray):
    train_data = data[: math.floor(len(data) * 0.8), :]
    validation_data = data[math.floor(len(data) * 0.8):, :]
    return train_data, validation_data


def data_train(x, y, itertime: int, init=False):
    w: np.ndarray
    # 163 ( 162 + 1 ) 这里是为了提供额外的bias项目
    dim = 18 * 9 + 1
    N = x.shape[0]
    if N != y.shape[0]:
        print("sample numbers of x,y are not equal")
        return
    if init:
        w = np.zeros((dim, 1))
        adagrad = np.zeros([dim, 1])
    else:
        model = np.load(MODELPATH)
        w = model['w']
        adagrad = model['adagrad']
    x = np.concatenate((x, np.ones((N, 1))), axis=1).astype(float)
    learning_rate = 100
    iter_time = itertime
    eps = 0.0000000001
    for t in range(iter_time):
        # 这里权重矩阵，因为前期的预处理，已经包括了bias
        loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)
        if (t % 100) == 0:
            print(str(t) + "times, loss is :" + str(loss))
        gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)
        adagrad += gradient ** 2
        w = w - learning_rate * gradient / np.sort(adagrad + eps)
    np.savez(MODELPATH, w=w, adagrad=adagrad)


def data_process_traindata(data):
    data = data.iloc[:, 3:]
    data[data == 'NR'] = 0
    raw_data = data.to_numpy()
    print(raw_data.shape)
    print(raw_data[0])
    return raw_data


def main():
    data = pd.read_csv(TRAIN_DATAPATH, encoding='big5')
    raw_data = data_process_traindata(data)
    # 一共十二个月的数据，类型是dict
    # 每个月的数据的形状均为18 x 480   -> 480 代表 480个小时（只记录了20天，而不是30天）
    month_data = data_process_monthdata(raw_data)
    # 转化为162维度的输入 （18个类型，9个小时：相当于不同小时的同类型的数据，也看成是不同的输入特征）
    x, y = data_process_x_y(month_data)
    # 对数据进行预处理
    x = data_process_normalize(x)
    # 将数据分成训练集与验证集
    train_x, validation_x = data_process_split(x)
    train_y, validation_y = data_process_split(y)
    # data_train(train_x, train_y, 1000, True)
    data_train(train_x, train_y, 10000)
    data_train(train_x, train_y, 10000)


if __name__ == '__main__':
    main()
