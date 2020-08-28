import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.disable_eager_execution()

# 导入数据
x_train = np.array(
    [[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], [9.779], [6.182], [7.59], [2.167], [7.042], [10.791], [5.313],
     [7.997], [3.1]], dtype=np.float32)
y_train = np.array(
    [[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], [3.366], [2.596], [2.53], [1.221], [2.827], [3.465], [1.65],
     [2.904], [1.3]], dtype=np.float32)

# 显示初始数据集

# 定义x，y两个变量
x = tf.Variable(initial_value=x_train, dtype=tf.float32, name='x')
y = tf.Variable(initial_value=y_train, dtype=tf.float32, name='y')

# weight权重设置为随机数，但是biase偏差量我们设置为0
w = tf.Variable(initial_value=tf.random_normal(shape=(), seed=2019, name="randomWeight"), dtype=tf.float32,
                name='weight')
b = tf.Variable(initial_value=0, dtype=tf.float32, name='biase')

# 声明一个线性模型作用域，因为这里有太多的运算，这样整体可以看成是一个模块
with tf.variable_scope('Linear_Model'):
    y_pred = w * x + b

# 开启会话与初始化参数
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# 可以在这里查看一下最开始的y_pred
y_pred_numpy = y_pred.eval(session=sess)
print(y_pred_numpy.shape)
plt.plot(x_train, y_train, 'bo', label='real')
plt.plot(x_train, y_pred_numpy, 'ro', label='estimated')
plt.show()

# 为什么这里要调用一次reduce_mean呢？
# reduce_mean:计算向量的各个维度上的元素的平均值.
# 因为y可能不是单独的一个数值，而是一个向量，同样x也可能是一个向量
# 虽然 w 与 b 是两个变量，即单独的数。但(w * x)也会是一个向量，并且y_pred = (w * x) + b 也会是一个向量（数与向量相加，变成向量）
# 所以(y - y_pred)也是一个向量，平方运算会对向量上的每一个维度都做平方运算。
# 不过我们最后需要的loss是一个变量值，而不是一个向量，于是调用reduce_mean，将向量的每一个维度，求平均值
loss = tf.reduce_mean(tf.square(y - y_pred))

# 让平均值loss对w与b求导数，得到w和b的梯度
w_grad, b_grad = tf.gradients(loss, [w, b])

# 定义学习率
with tf.variable_scope('Learning_Model'):
    lr = 1e-2
    w_update = w.assign_sub(lr * w_grad)
    b_update = b.assign_sub(lr * b_grad)

# 打印图
tf.summary.FileWriter('D:\\Storage\\pycharmProjects\\demo\\structure', tf.get_default_graph())

# 运行图
for i in range(10):
    print(sess.run([w_update, b_update]))

# 可以在这里查看一下更新的y_pred
y_pred_numpy = y_pred.eval(session=sess)
plt.plot(x_train, y_train, 'bo', label='real')
plt.plot(x_train, y_pred_numpy, 'ro', label='estimated')
plt.show()

sess.close()
