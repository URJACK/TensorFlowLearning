import pandas as pd
import numpy as np
import math
from env_tf_2_3.pytorch_learn.regression import model

TEST_DATAPATH = './data/test.csv'


def data_process_testdata(data):
    data = pd.read_csv(TEST_DATAPATH, header=None, encoding='big5')
    data = data.iloc[:, 2:]
    data[data == 'NR'] = 0
    raw_data = data.to_numpy()
    return raw_data


def data_process_x(data: np.ndarray) -> np.ndarray:
    N = data.shape[0]
    DATA_SPAN = 18
    DATA_SINGLE_DIM = 9
    beginIndex = 0
    sample_num = N // DATA_SPAN  # 样本个数
    # 偏移量项，需要额外补足一个维度
    x = np.empty([sample_num, DATA_SPAN * DATA_SINGLE_DIM])
    for cursor in range(sample_num):
        for i in range(DATA_SPAN):
            x[cursor, i * DATA_SINGLE_DIM: (i + 1) * DATA_SINGLE_DIM] = data[beginIndex + i, :]
        beginIndex = beginIndex + 18
    x = np.concatenate([x, np.ones((sample_num, 1))], axis=1)
    return x


def main():
    data = pd.read_csv(TEST_DATAPATH, header=None, encoding="big5")
    raw_data = data_process_testdata(data)
    x = raw_data
    x = data_process_x(x)
    x = model.data_process_normalize(x)
    m = np.load(model.MODELPATH)
    w = m['w']
    y_pred = np.dot(x, w)
    print(y_pred)


if __name__ == '__main__':
    main()
