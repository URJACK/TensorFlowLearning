import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(2017)


# Module show Image

def plot_decision_boundary(model, x, y):
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    # 绘图网格的单位长度
    h = 0.01
    # 将生成的x向量与y向量 扩充为x向量矩阵xx、y向量矩阵yy
    # xx 与 yy具有相同的维度(m x n) 其中xx中，x向量自身是行向量，竖直堆叠
    # yy中，y向量是列向量，横向堆叠
    # 只需要将xx 中每一单元 与 yy 中每一单元 依次对应，就可以得到每一个单元点
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    # 计算得到的单元点的数值之后，再重新定义矩阵维度
    Z = Z.reshape(xx.shape)
    # 这样，每一个点(xx,yy)，及其点的颜色（函数值Z）都绘制完毕
    plt.contourf(xx, yy, Z)
    plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y))
    plt.show()

    # End Module


# Module: Create Image
# x y 就是训练集的 输入与label
np.random.seed(1)
m = 400  # samples size
N = int(m / 2)  # 每一类点的个数
D = 2  # 维度
x = np.zeros((m, D))  # 输入特征(2个维度) 第一个维度是横坐标，第二个维度是纵坐标
y = np.zeros((m, 1), dtype='uint8')  # label 有两种取值（0，1）分别代表红色 和 蓝色
a = 4

for j in range(2):
    ix = range(N * j, N * (j + 1))  # range(0, 200)、range(200, 400) numpy可以使用这种range对象作为索引
t = np.linspace(j * 3.12, (j + 1) * 3.12, N)  # 生成(0, π) 的数据   t(200,) 是一维向量
t2 = np.random.randn(N) * 0.2  # 生成标准正态分布数值 因为N==200，所以生成的 t2(200,) 是一个一维向量
t = t + t2  # theta
r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]  # np.c_代表将两个矩阵横向拼接，可见一维向量非常类似于列向量
y[ix] = j

# End Module

# plt.plot(x[:200, 0], x[:200, 1], 'ro', label='red')
# plt.plot(x[200:, 0], x[200:, 1], 'bo', label='blue')
# plt.show()

# 将numpy对象放入tensor中
x = tf.constant(x, dtype=tf.float32, name='x')
y = tf.constant(y, dtype=tf.float32, name='y')

# 使用初始化构造器 获取tensor变量
w = tf.get_variable(initializer=tf.random_normal_initializer(), shape=(2, 1), dtype=tf.float32, name='weights')
b = tf.get_variable(initializer=tf.zeros_initializer(), shape=1, dtype=tf.float32, name='biase')


def logistic_model(inputs):
    logit = tf.matmul(inputs, w) + b
    return tf.sigmoid(logit)


y_ = logistic_model(x)  # 使用这个，搭建神经网络

loss = tf.losses.log_loss(predictions=y_, labels=y)

lr = 1e-1
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

TARINING_TIME = 1000
for e in range(TARINING_TIME):
    sess.run(train_op)
    if e % 100 == 0:
        loss_numpy = sess.run(loss)
        print("训练 %d 次， 损失 %.12f " % (e + 1, loss_numpy))

data_input = tf.placeholder(shape=(None, 2), dtype=tf.float32, name="logistic_input")
logistic_output = logistic_model(data_input)


def plot_logistic(x_data):
    y_pred_numpy = sess.run(logistic_output, feed_dict={data_input: x_data})
    out = np.greater(y_pred_numpy, 0.5).astype(np.float32)
    return np.squeeze(out)


# 调用显示分布图的函数
plot_decision_boundary(plot_logistic, sess.run(x), sess.run(y))
