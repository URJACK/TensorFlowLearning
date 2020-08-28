import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import os

im = Image.open(os.path.abspath(os.path.dirname(__file__)) + '\\a.jpg').convert('L')
# im = Image.open('D:\\Storage\\pycharmProjects\\demo\\cnn\\a.jpg')
# im = Image.open(ulb.urlopen(im_url)).convert('L')
# 这里类型指定为浮点类型很重要
im = np.array(im, dtype='float32')
# 查看图片的shape
print(im.shape)
# 图片的数据类型因为之前被指定为了浮点类型，所以这里要使用astype，暂时转化为整型
plt.imshow(im.astype('uint8'), cmap='gray')
# plt.imshow(im.astype('uint8'))
plt.show()

# 卷积使用的图片需要进行形状的预处理
im = tf.constant(im.reshape((1, im.shape[0], im.shape[1], 1)), name='input')
# 定义卷积核
sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
sobel_kernel = tf.constant(sobel_kernel, shape=(3, 3, 1, 1))

# 进行卷积
edge1 = tf.nn.conv2d(im, sobel_kernel, [1, 1, 1, 1], 'SAME', name='same_conv')
edge2 = tf.nn.conv2d(im, sobel_kernel, [1, 1, 1, 1], 'VALID', name='valid_conv')

sess = tf.Session()
edge1_np, edge2_np = sess.run([edge1, edge2])
print(edge1_np.shape)
print(edge2_np.shape)

plt.imshow(np.squeeze(edge1_np), cmap='gray')
plt.title("edge_same")
plt.show()

plt.imshow(np.squeeze(edge2_np), cmap='gray')
plt.title("edge_valid")
plt.show()

pool1 = tf.nn.max_pool(im, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='same_max_pool')
pool2 = tf.nn.max_pool(im, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name='valid_max_pool')

pool1_np, pool2_np = sess.run([pool1, pool2])
print(pool1_np.shape)
print(pool2_np.shape)
print(np.squeeze(pool2_np).shape)

plt.imshow(np.squeeze(pool1_np), cmap='gray')
plt.title("pool_same")
plt.show()

plt.imshow(np.squeeze(pool2_np), cmap='gray')
plt.title("pool_valid")
plt.show()
