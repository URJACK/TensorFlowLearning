import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

tf.disable_eager_execution()

# 模拟设定 w 与 b 的数值、看最后训练出来的w 与 b 能不能接近这两个数值
w_target = np.array([0.5, 3, 2.4])
b_target = np.array([0.9])

# 调试一下表达式
f_des = 'y = {:.2f} + {:.2f} * x + {:.2f} * x^2 + {:.2f} * x^3'.format(b_target[0], w_target[0], w_target[1],
                                                                       w_target[2])
print(f_des)

# 模拟数据集
x_sample = np.arange(-3, 3.1, 0.1)
y_sample = b_target + w_target[0] * x_sample + w_target[1] * x_sample ** 2 + w_target[2] * x_sample ** 3
# plt.plot(x_sample, y_sample, label='real curve')
# plt.show()

# 构建初始数据
x_train = np.stack([x_sample ** i for i in range(1, 4)], axis=1)
x_train = tf.constant(x_train, dtype=tf.float32, name='x_train')
y_train = tf.constant(y_sample, dtype=tf.float32, name='y_train')

w = tf.Variable(initial_value=tf.random_normal(shape=(3, 1), name="random_weight"), dtype=tf.float32, name='weights')
b = tf.Variable(initial_value=0, dtype=tf.float32, name='biase')

# 开启会话与初始化参数
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# tf.squeeze之后，默认的，会变成列向量
with tf.name_scope("Linear_Model"):
    y_expected = tf.squeeze(tf.matmul(x_train, w) + b)

# 要使用plot查看一下图也太麻烦了....
# 必须得从tensor类型转化为numpy数组--->  numpy.Array == sess.run(tensor)

y_expected_numpy = sess.run(y_expected)
x_train_numpy = sess.run(x_train)
y_train_numpy = sess.run(y_train)
# x_train_numpy[:,0] 是一个列向量(n x 1) ，数值刚好就是输入的x的数值
plt.plot(x_train_numpy[:, 0], y_train_numpy, 'bo', label='real')
# x_train_numpy[:,1] 与 x_train_numpy[:,2] ，数值刚好是输入的x^2 与 x^3 的数值，这些没有必要作为横坐标
# plt.plot(x_train_numpy[:, 1], y_train_numpy, 'ro', label='real')
# plt.plot(x_train_numpy[:, 2], y_train_numpy, 'go', label='real')
plt.plot(x_train_numpy[:, 0], y_expected_numpy, 'go', label='estimated')
plt.show()

# 求梯度
loss = tf.reduce_mean(tf.square(y_expected - y_train))
w_grad, b_grad = tf.gradients(loss, [w, b])
print("loss", sess.run(loss))
print("w_grad", sess.run(w_grad))
print("b_grad", sess.run(b_grad))

# 学习
with tf.name_scope("Learning_Model"):
    learningRate = 1e-3
    w_update = w.assign_sub(learningRate * w_grad)
    b_update = b.assign_sub(learningRate * b_grad)

for i in range(10):
    print(sess.run([w_update, b_update]))

y_expected_numpy = sess.run(y_expected)
plt.plot(x_train_numpy[:, 0], y_train_numpy, 'bo', label='real')
plt.plot(x_train_numpy[:, 0], y_expected_numpy, 'go', label='estimated')
plt.show()
