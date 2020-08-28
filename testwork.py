import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4], dtype=np.float)
x = np.reshape(x, (4, 1))
y1 = x
x = np.array([2, 3, 4, 5], dtype=np.float)
x = np.reshape(x, (4, 1))
y2 = x
y_pred_np = np.c_[y1, y2]

x = np.array([4, 4, 4, 4], dtype=np.float)
x = np.reshape(x, (4, 1))
y1 = x
x = np.array([5, 5, 6, 6], dtype=np.float)
# x = np.array([5, 5, 5, 5], dtype=np.float)
x = np.reshape(x, (4, 1))
y2 = x
y_np = np.c_[y1, y2]

y_pred = tf.constant(y_pred_np, dtype=tf.float32, name="y_pred")
y = tf.constant(y_np, dtype=tf.float32, name="y_np")

t1 = tf.square(y_pred - y)
t2 = tf.reduce_mean(t1)
t1_ave = tf.reduce_mean(t1, axis=0)
t2_ave = tf.reduce_mean(t2)

sess = tf.Session()
t1_np = sess.run(t1)
t1_ave_np = sess.run(t1_ave)

print(t1_np)
# print(t2_np)
print(t1_ave_np)

# noise = np.random.rand(10, 2)
# print(noise)
