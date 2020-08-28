from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.examples.tutorials.mnist.input_data as input_data
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

tf.set_random_seed(2017)

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # 设置画图的尺寸
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def show_images(images):  # 定义画图工具
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))

    return


def deprocess_img(x):
    return (x + 1.0) / 2.0


mnist = input_data.read_data_sets('MINST_data')
train_set = mnist.train_model
test_set = mnist.test

input_ph = tf.placeholder(tf.float32, shape=[None, 784], name="inputdata")
inputs = tf.divide(input_ph - 0.5, 0.5)

noise_dim = 96


def discriminator(data, scope="discriminator", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=None):
            net = slim.conv2d(data, 32, 5, stride=1, scope='conv1')
            net = tf.nn.leaky_relu(net, alpha=0.2, name='act1')
            net = slim.max_pool2d(net, 2, stride=2, scope='maxpool1')
            net = slim.conv2d(net, 64, 5, stride=1, scope='conv2')
            net = tf.nn.leaky_relu(net, alpha=0.2, name='act2')
            net = slim.max_pool2d(net, 2, stride=2, scope='maxpool2')
            net = slim.flatten(net, scope='flatten')
            net = slim.fully_connected(net, 1024, scope='fc3')
            net1 = tf.nn.leaky_relu(net, alpha=0.01, name='act3')
            net1 = slim.fully_connected(net1, 1, scope='fc4')
            net2 = tf.nn.tanh(net)
            net2 = slim.fully_connected(net2, noise_dim, scope='fc5')
            return net1, net2


def generator(noise, scope='generator', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.fully_connected, slim.conv2d_transpose], activation_fn=None):
            net = slim.fully_connected(noise, 1024, scope='fc1')
            net = tf.nn.relu(net, name='act1')
            net = slim.batch_norm(net, scope='bn1')
            net = slim.fully_connected(net, 7 * 7 * 128, scope='fc2')
            net = tf.nn.relu(net, name='act2')
            net = slim.batch_norm(net, scope='bn2')
            net = tf.reshape(net, (-1, 7, 7, 128))
            net = slim.conv2d_transpose(net, 64, 4, stride=2, scope='convT3')
            net = tf.nn.relu(net, name='act3')
            net = slim.batch_norm(net, scope='bn3')
            net = slim.conv2d_transpose(net, 1, 4, stride=2, scope='convT4')
            net = tf.tanh(net, name='tanh')
            return net


batch_size = tf.shape(input_ph)[0]
true_labels = tf.ones((batch_size, 1), dtype=tf.int64, name='true_labels')
fake_labels = tf.zeros((batch_size, 1), dtype=tf.int64, name='fake_labels')

noise_lumbda = 0.25


def discriminator_loss(logics_real, logics_fake, noise_loss_1, scope='D_Loss'):
    with tf.variable_scope(scope):
        loss = tf.reduce_mean(logics_real) + tf.reduce_mean(logics_fake) - noise_lumbda * noise_loss_1
        return loss


def generator_loss(fake, noise_loss_1, scope='G_loss'):  # 生成网络的`loss`
    with tf.variable_scope(scope):
        loss = tf.losses.log_loss(true_labels, tf.sigmoid(fake)) - noise_lumbda * noise_loss_1
        return loss


def std(z):
    return tf.reduce_mean(tf.square(z - tf.reduce_mean(z)))


def noise_loss(noise, sample):
    loss = tf.reduce_mean((noise - tf.reduce_mean(noise, axis=0)) * (sample - tf.reduce_mean(sample, axis=0))) / (
            std(noise) * std(sample))
    # loss = tf.losses.softmax_cross_entropy(logits=noise, onehot_labels=sample)
    return loss


sample_noise = tf.random_uniform([batch_size, noise_dim], dtype=tf.float32, minval=-1.0, maxval=1.0,
                                 name='sample_noise')

dc_inputs = tf.reshape(inputs, (-1, 28, 28, 1))

inputs_fake = generator(sample_noise)
logits_real, noise_dis1 = discriminator(dc_inputs)
logits_fake, noise_dis2 = discriminator(inputs_fake, reuse=True)

noise_total_error = noise_loss(noise_dis1, sample_noise) + noise_loss(noise_dis2, sample_noise)
d_total_error = discriminator_loss(logits_real, logits_fake, noise_total_error)
g_total_error = generator_loss(logits_fake, noise_total_error)

opt = tf.train.AdamOptimizer(3e-4, beta1=0.5, beta2=0.999)
discriminator_params = tf.trainable_variables('discriminator')
dc_train_discriminator = opt.minimize(d_total_error, var_list=discriminator_params)
generator_params = tf.trainable_variables('generator')
with tf.control_dependencies([dc_train_discriminator]):
    dc_train_generator = opt.minimize(g_total_error, var_list=generator_params)

# with tf.control_dependencies([dc_train_generator]):
#     dc_train_noise_discriminator = opt.minimize(noise_total_error, var_list=discriminator_params)
#
# with tf.control_dependencies([dc_train_noise_discriminator]):
#     dc_train_noise_generator = opt.minimize(noise_total_error, var_list=generator_params)

print(input_ph)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(inputs_fake)

iter_count = 0
show_every = 100
for e in range(5):
    num_examples = 0
    while num_examples < train_set.num_examples:
        if num_examples + 128 < train_set.num_examples:
            batch = 128
        else:
            batch = train_set.num_examples - num_examples
        num_examples += batch
        train_imgs, _ = train_set.next_batch(batch)
        _ = sess.run([dc_train_generator], feed_dict={input_ph: train_imgs})
        if iter_count % show_every == 0:
            loss_d, loss_g, loss_n, fake_imgs = sess.run(
                [d_total_error, g_total_error, noise_total_error, inputs_fake],
                feed_dict={input_ph: train_imgs})
            print(loss_g)
            print(loss_n)
            print('Iter: {},D: {:.4f}, G:{:.4f} , N:{:.4f}'.format(iter_count, loss_d, loss_g, loss_n))
            imgs_numpy = deprocess_img(fake_imgs)
            show_images(imgs_numpy[:16])
            plt.show()
            imgs_numpy = deprocess_img(train_imgs)
            show_images(imgs_numpy[:16])
            plt.show()
        iter_count += 1

saver = tf.train.Saver()
saver.save(sess, os.path.abspath(os.path.dirname(__file__)) + "\\model\\ogan.ckpt")
g = tf.get_default_graph()
# 获取图文件的命令 这里我使用了绝对路径
tf.summary.FileWriter(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '\\structure', g)
