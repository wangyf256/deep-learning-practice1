#tensorflow的全连接层实现(只有softmax)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 处理数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
batch_size = 128
n_batch = mnist.train.num_examples // batch_size

#构建tensorflow图

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

 #create a simple neutral network
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#定义需要的变量
prediction = tf.nn.softmax(tf.matmul(x, W)+b)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))                 

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss = tf.reduce_mean(tf.square(y-prediction))

train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

#运行构建好的图
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
                acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
                print('Iter' + str(epoch) + ",Testing Accuracy" + str(acc))
                plt.plot(range(len(loss)) , loss)
                plt.show()