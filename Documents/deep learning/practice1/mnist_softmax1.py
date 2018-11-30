#!/usr/bin/env python3  
# -*- coding: utf-8 -*- 
 
import input_data
import tensorflow as tf
 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
 
#x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。
#我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。
#我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。
#（这里的None表示此张量的第一个维度可以是任何长度的。）
x = tf.placeholder(tf.float32, [None, 784])
#权重值，初始值全为0
W = tf.Variable(tf.zeros([784,10]))
#偏置量，初始值全为0
b = tf.Variable(tf.zeros([10]))
 
#建立模型，y是匹配的概率
#tf.matmul(x,W)表示x乘以W
#y是预测，y_是实际
y = tf.nn.softmax(tf.matmul(x,W) + b)
 
#为计算交叉熵，添加的placeholder
y_ = tf.placeholder("float", [None,10])
 
#交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
 
#初始化我们创建的变量
init = tf.global_variables_initializer()
 
#在Session里面启动模型
sess = tf.Session()
sess.run(init)
 
#训练模型
#循环的每个步骤中，都会随机抓取训练数据中的100个批处理数据点，然后用这些数据点作为参数替换之前的占位符来运行train_step
#即：使用的随机梯度下降训练方法
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
 
#-------------------模型评估----------------------
#判断预测标签和实际标签是否匹配 
#tf.argmax 找出某个tensor对象在某一维上的其数据最大值所在的索引值
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
 
#计算所学习到的模型在测试数据集上面的正确率 
print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
