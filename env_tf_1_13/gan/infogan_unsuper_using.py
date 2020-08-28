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

tf.set_random_seed(2022)

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


sess = tf.Session()
saver = tf.train.import_meta_graph(os.path.abspath(os.path.dirname(__file__)) + '\\model\\infogan_unsuper.ckpt.meta')
saver.restore(sess, os.path.abspath(os.path.dirname(__file__)) + "\\model\\infogan_unsuper.ckpt")

g = tf.get_default_graph()
inputs_fake = g.get_tensor_by_name("generator/tanh:0")
input_ph = g.get_tensor_by_name("inputdata:0")
label_ph = g.get_tensor_by_name("inputlabel:0")
latent_ph = g.get_tensor_by_name("inputlatent:0")
labels_fake_dis = g.get_tensor_by_name("discriminator_1/fc5/BiasAdd:0")
latent_fake_dis = g.get_tensor_by_name("discriminator_1/fc7/BiasAdd:0")

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_set = mnist.train_model
test_set = mnist.test

train_imgs, train_labels = test_set.next_batch(8)

labelHolder = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

latent_placeHolder = np.random.rand(8, 2)
latent_placeHolder = np.array(
    [[0.12, 0.12], [0.12, 0.12], [0.22, 0.22], [0.22, 0.22], [0.32, 0.32], [0.32, 0.32], [0.42, 0.42], [0.42, 0.42]])
inputHolder = np.random.rand(labelHolder.shape[0], 784)
fake_imgs, labels_fake_dis_numpy, latent_fake_dis_numpy = sess.run([inputs_fake, labels_fake_dis, latent_fake_dis],
                                                                   feed_dict={input_ph: inputHolder,
                                                                              label_ph: train_labels,
                                                                              latent_ph: latent_placeHolder})
imgs_numpy = deprocess_img(fake_imgs)
show_images(imgs_numpy[:16])
plt.show()
fake_imgs, labels_fake_dis_numpy, latent_fake_dis_numpy = sess.run([inputs_fake, labels_fake_dis, latent_fake_dis],
                                                                   feed_dict={input_ph: inputHolder,
                                                                              label_ph: labelHolder,
                                                                              latent_ph: latent_placeHolder})

imgs_numpy = deprocess_img(fake_imgs)
show_images(imgs_numpy[:16])
plt.show()
print(train_labels)
print(labelHolder)
print(labelHolder.dtype)
print(np.argmax(labels_fake_dis_numpy, axis=1))
print(latent_fake_dis_numpy)
