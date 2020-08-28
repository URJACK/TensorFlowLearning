from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tensorflow.examples.tutorials.mnist.input_data as input_data

tf.set_random_seed(2019)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_set = mnist.train_model
test_set = mnist.test

fig, axes = plt.subplots(ncols=6, nrows=2)
plt.tight_layout(w_pad=-2.0, h_pad=-8.0)

images, labels = train_set.next_batch(12, shuffle=False)

for ind, (image, label) in enumerate(zip(images, labels)):
    image = image.reshape((28, 28))
    label = label.argmax()

    row = ind // 6
    col = ind % 6

    # 填充子图元素
    axes[row][col].imshow(image, cmap='gray')
    axes[row][col].axis('off')
    axes[row][col].set_title('%d' % label)

plt.show()


def hidden_layer(layer_input, output_depth, scope='hidden_layer', reuse=None):
    input_depth = layer_input.get_shape()[-1]
    with tf.variable_scope(scope, reuse=reuse):
        w = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1), shape=(input_depth, output_depth),
                            name="weights")
        b = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=output_depth, name='biase')
        net = tf.matmul(layer_input, w) + b
        return net


def DNN(x, output_depths, scope='DNN', reuse=None):
    net = x
    for i, output_depth in enumerate(output_depths):
        net = hidden_layer(net, output_depth, scope='layer%d' % i, reuse=reuse)
        net = tf.nn.relu(net)
    net = hidden_layer(net, 10, scope='classification', reuse=reuse)
    # 注意，这里先不加上softmax，就用一个10维输出即可
    return net


# 设置占位符
input_ph = tf.placeholder(shape=(None, 784), dtype=tf.float32)
label_ph = tf.placeholder(shape=(None, 10), dtype=tf.int64)

# 构建一个4层神经网络(400,200,100,10)
dnn = DNN(input_ph, [400, 200, 100])
# 交叉熵计算损失函数
loss = tf.losses.softmax_cross_entropy(logits=dnn, onehot_labels=label_ph)

# 下面定义的是正确率, 注意理解它为什么是这么定义的
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(dnn, axis=-1), tf.argmax(label_ph, axis=-1)), dtype=tf.float32))

# 定义学习率
lr = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)

sess = tf.InteractiveSession()

batch_size = 64

sess.run(tf.global_variables_initializer())

for e in range(20000):
    images, labels = train_set.next_batch(batch_size)
    # print(labels.shape)
    sess.run(train_op, feed_dict={input_ph: images, label_ph: labels})
    if e % 1000 == 999:
        # 获取 batch_size 个测试样本
        test_imgs, test_labels = test_set.next_batch(batch_size)
        # 计算在当前样本上的训练以及测试样本的损失值和正确率
        loss_train, acc_train = sess.run([loss, acc], feed_dict={input_ph: images, label_ph: labels})
        loss_test, acc_test = sess.run([loss, acc], feed_dict={input_ph: test_imgs, label_ph: test_labels})
        print(
            'STEP {}: train_loss: {:.6f} train_acc: {:.6f} test_loss: {:.6f}test_acc: {:.6f}'.format(e + 1, loss_train,
                                                                                                     acc_train,
                                                                                                     loss_test,
                                                                                                     acc_test))

print('Train Done!')
print('-' * 30)

test_loss_array = []
test_acc_array = []
for _ in range(test_set.num_examples // 100):
    image, label = test_set.next_batch(100)
    loss_test, acc_test = sess.run([loss, acc], feed_dict={input_ph: image, label_ph: label})
    test_loss_array.append(loss_test)
    test_acc_array.append(acc_test)

print('test loss: {:.6f}'.format(np.array(test_loss_array).mean()))
print('test accuracy: {:.6f}'.format(np.array(test_acc_array).mean()))
