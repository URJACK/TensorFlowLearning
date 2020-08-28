from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.examples.tutorials.mnist.input_data as input_data

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
print(mnist)
train_set = mnist.train_model
print(train_set)
test_set = mnist.test
print(test_set)

input_ph = tf.placeholder(tf.float32, shape=[None, 784])
inputs = tf.divide(input_ph - 0.5, 0.5)


def discriminator(data, scope="discriminator", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.fully_connected], activation_fn=None):
            net = slim.fully_connected(data, 256, scope='fc1')
            net = tf.nn.leaky_relu(net, alpha=0.2, name='act1')
            net = slim.fully_connected(net, 256, scope='fc2')
            net = tf.nn.leaky_relu(net, alpha=0.2, name='act2')
            net = slim.fully_connected(net, 1, scope='fc3')
            return net


def generator(noise, scope='generator', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
            net = slim.fully_connected(noise, 1024, scope='fc1')
            net = slim.fully_connected(net, 1024, scope='fc2')
            net = slim.fully_connected(net, 784, activation_fn=tf.tanh, scope='fc3')
            return net


batch_size = tf.shape(input_ph)[0]
true_labels = tf.ones((batch_size, 1), dtype=tf.int64, name='true_labels')
fake_labels = tf.zeros((batch_size, 1), dtype=tf.int64, name='fake_labels')


def ls_discriminator_loss(logics_real, logics_fake, scope='D_Loss'):
    with tf.variable_scope(scope):
        loss = 0.5 * tf.reduce_mean(tf.square(logics_real - 1)) + 0.5 * tf.reduce_mean(tf.square(logics_fake))
    return loss


def ls_generator_loss(fake, scope='G_loss'):  # 生成网络的`loss`
    with tf.variable_scope(scope):
        loss = 0.5 * tf.reduce_mean(tf.square(logits_fake - 1))
    return loss


noise_dim = 96
sample_noise = tf.random_uniform([batch_size, noise_dim], dtype=tf.float32, minval=-1.0, maxval=1.0,
                                 name='sample_noise')
inputs_fake = generator(sample_noise)
logits_real = discriminator(inputs)
logits_fake = discriminator(inputs_fake, reuse=True)

ls_d_total_error = ls_discriminator_loss(logits_real, logits_fake)
ls_g_total_error = ls_generator_loss(logits_fake)

opt = tf.train.AdamOptimizer(3e-4, beta1=0.5, beta2=0.999)
discriminator_params = tf.trainable_variables('discriminator')
train_ls_discriminator = opt.minimize(ls_d_total_error, var_list=discriminator_params)
generator_params = tf.trainable_variables('generator')
with tf.control_dependencies([train_ls_discriminator]):
    train_ls_generator = opt.minimize(ls_g_total_error, var_list=generator_params)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

iter_count = 0
show_every = 1000
for e in range(10):
    num_examples = 0
    while num_examples < train_set.num_examples:
        if num_examples + 128 < train_set.num_examples:
            batch = 128
        else:
            batch = train_set.num_examples - num_examples
        num_examples += batch
        train_imgs, _ = train_set.next_batch(batch)
        loss_d, loss_g, fake_imgs, _ = sess.run([ls_d_total_error, ls_g_total_error, inputs_fake, train_ls_generator],
                                                feed_dict={input_ph: train_imgs})
        if iter_count % show_every == 0:
            print('Iter: {},D: {:.4f},G:{:.4f}'.format(iter_count, loss_d, loss_g))
            imgs_numpy = deprocess_img(fake_imgs)
            show_images(imgs_numpy[:16])
            plt.show()
            print()
            imgs_numpy = deprocess_img(train_imgs)
            show_images(imgs_numpy[:16])
            plt.show()
            print()
        iter_count += 1
