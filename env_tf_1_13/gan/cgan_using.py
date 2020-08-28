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


sess = tf.Session()
saver = tf.train.import_meta_graph(os.path.abspath(os.path.dirname(__file__)) + '\\model\\cgan.ckpt.meta')
saver.restore(sess, os.path.abspath(os.path.dirname(__file__)) + "\\model\\cgan.ckpt")

g = tf.get_default_graph()
inputs_fake = g.get_tensor_by_name("generator/fc3/Tanh:0")
input_ph = g.get_tensor_by_name("inputdata:0")
label_ph = g.get_tensor_by_name("inputlabel:0")

labelHolder = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
inputHolder = np.random.rand(labelHolder.shape[0], 784)

fake_imgs = sess.run(inputs_fake, feed_dict={input_ph: inputHolder, label_ph: labelHolder})

imgs_numpy = deprocess_img(fake_imgs)
show_images(imgs_numpy[:16])
plt.show()
