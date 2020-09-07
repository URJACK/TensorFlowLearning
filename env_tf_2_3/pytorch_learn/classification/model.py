import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X_train_fpath = './data/X_train'
Y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'
output_fpath = './output_{}.csv'

data_dim: int


def main():
    # Parse csv files to numpy array
    global data_dim
    with open(X_train_fpath) as file:
        next(file)
        x_train = np.array([line.strip('\n').split(',')[1:] for line in file], dtype=float)
    with open(Y_train_fpath) as file:
        next(file)
        y_train = np.array([line.strip('\n').split(',')[1] for line in file], dtype=float)
    with open(X_test_fpath) as file:
        next(file)
        x_test = np.array([line.strip('\n').split(',')[1:] for line in file], dtype=float)

    x_train, x_mean, x_std = normalize(x_train, isTrain=True)
    x_test, _, _ = normalize(x_test, isTrain=False, specified_columns=None, x_mean=x_mean, x_std=x_std)

    dev_ratio = 0.1
    x_train, y_train, x_dev, y_dev = train_dev_split(x_train, y_train, dev_ratio)

    train_size = x_train.shape[0]
    dev_size = x_dev.shape[0]
    test_size = x_test.shape[0]
    data_dim = x_train.shape[1]
    print('Size of training set: {}'.format(train_size))
    print('Size of development set: {}'.format(dev_size))
    print('Size of testing set: {}'.format(test_size))
    print('Dimension of data: {}'.format(data_dim))
    train(x_train, y_train, x_dev, y_dev, train_size, dev_size)


def train(x_train, y_train, x_dev, y_dev, train_size, dev_size):
    global data_dim
    # Zero initialization for weights ans bias
    w = np.zeros((data_dim,))
    b = np.zeros((1,))

    # Some parameters for training
    max_iter = 10
    batch_size = 8
    learning_rate = 0.2

    # Keep the loss and accuracy at every iteration for plotting
    train_loss = []
    dev_loss = []
    train_acc = []
    dev_acc = []

    # Calcuate the number of parameter updates
    step = 1

    # Iterative training
    for epoch in range(max_iter):
        # Random shuffle at the begging of each epoch
        x_train, y_train = shuffle(x_train, y_train)

        # Mini-batch training
        for idx in range(int(np.floor(train_size / batch_size))):
            X = x_train[idx * batch_size:(idx + 1) * batch_size]
            Y = y_train[idx * batch_size:(idx + 1) * batch_size]

            # Compute the gradient
            w_grad, b_grad = gradient(X, Y, w, b)

            # gradient descent update
            # learning rate decay with time
            w = w - learning_rate / np.sqrt(step) * w_grad
            b = b - learning_rate / np.sqrt(step) * b_grad

            step = step + 1

        # Compute loss and accuracy of training set and development set
        y_train_pred = f(x_train, w, b)
        Y_train_pred = np.round(y_train_pred)
        train_acc.append(accuracy(Y_train_pred, y_train))
        train_loss.append(cross_entropy_loss(y_train_pred, y_train) / train_size)

        y_dev_pred = f(x_dev, w, b)
        Y_dev_pred = np.round(y_dev_pred)
        dev_acc.append(accuracy(Y_dev_pred, y_dev))
        dev_loss.append(cross_entropy_loss(y_dev_pred, y_dev) / dev_size)

    # 打印最后一次的数据
    print('Training loss: {}'.format(train_loss[-1]))
    print('Development loss: {}'.format(dev_loss[-1]))
    print('Training accuracy: {}'.format(train_acc[-1]))
    print('Development accuracy: {}'.format(dev_acc[-1]))

    # Loss curve
    plt.plot(train_loss)
    plt.plot(dev_loss)
    plt.title('Loss')
    plt.legend(['train', 'dev'])
    plt.savefig('loss.png')
    plt.show()

    # Accuracy curve
    plt.plot(train_acc)
    plt.plot(dev_acc)
    plt.title('Accuracy')
    plt.legend(['train', 'dev'])
    plt.savefig('acc.png')
    plt.show()


def shuffle(x, y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    return x[randomize], y[randomize]


def sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - 1e-8)


def f(x, w, b):
    return sigmoid(np.matmul(x, w) + b)


def accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc


def cross_entropy_loss(y_pred, Y_label):
    cross_entropy = -np.dot(Y_label, np.log(y_pred + 1e-8)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy


def gradient(X, Y_label, w, b):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad


def normalize(x: np.ndarray, isTrain=True, specified_columns: np.ndarray = None, x_mean=None, x_std=None):
    if specified_columns is None:
        # 如果没有指定columns，那么所有的都将包括在这个范围内
        specified_columns = np.arange(x.shape[1])
    if isTrain:
        x_mean = np.mean(x[:, specified_columns], 0).reshape(1, -1)
        x_std = np.std(x[:, specified_columns], 0).reshape(1, -1)
    x[:, specified_columns] = (x[:, specified_columns] - x_mean) / (x_std + 1e-8)
    return x, x_mean, x_std


def train_dev_split(X, Y, dev_ratio=0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


if __name__ == '__main__':
    main()
