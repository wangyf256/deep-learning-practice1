#tensorflow的FC实现
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

#参数设置
INPUT_NODE = 784                # 输入节点数
OUTPUT_NODE = 10                # 输出节点数

LAYER1_NODE = 500              #隐层节点数

BATCH_NODE = 128                # batch的大小
LEARNING_RATE = 0.1             # 基础的学习率
LEARNING_RATE_DECAY = 0.99      # 学习率的衰减率
REGULARIZATION_RATE = 0.0001    # l2正则化项在损失函数中的系数
TRAINING_STEPS = 10000          # 训练轮数
MOVING_RATE_DEACY = 0.99        #滑动模型那个衰减率

i_c = []
loss_c = []


#前向传播，variable_average平均滑动模型参数
def inference(x,variable_average,w1,b1,w2,b2):
    
    if variable_average == None:    
        layer1 = tf.nn.relu(tf.matmul(x,w1)+b1)
        return tf.matmul(layer1,w2)+b2
    
    else:
        layer1 = tf.nn.relu(tf.matmul(x,variable_average.average(w1))+variable_average.average(b1))
        return tf.matmul(layer1,variable_average.average(w2))+variable_average.average(b2)


def train(mnist): 
    
    # 首先定义输入数据，使用了placeholder
    x = tf.placeholder(tf.float32,[None,784])
    y_ = tf.placeholder(tf.float32,[None,10])
    
    #参数初始化
    w1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    w2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    
    #不使用平均化滑动模型的前向传播结果
    y = inference(x,None,w1,b1,w2,b2)
    
    #平均滑动模型
    global_step = tf.Variable(0,trainable=False)
    #定义一个平均滑动模型的类
    variable_average = tf.train.ExponentialMovingAverage(MOVING_RATE_DEACY,global_step)
    #定义一个平均华东模型操作，应用给所有可训练变量
    variable_average_op = variable_average.apply(tf.trainable_variables())
    #使用平均化滑动模型的前向传播结果
    average_y = inference(x,variable_average,w1,b1,w2,b2)
    
    #交叉熵损失函数
    cost_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cost_entropy_mean = tf.reduce_mean(cost_entropy)
    
    #l2正则化
    regulations = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    l2_regulation = regulations(w1) + regulations(w2)
    
    #带有正则化的损失函数作为最终的损失函数
    loss = cost_entropy_mean + l2_regulation
    
    #学习率衰减
    learning_rate_deacy = tf.train.exponential_decay(learning_rate=0.1,global_step=global_step,decay_steps=100,decay_rate=0.99)
    #训练
    train_step = tf.train.GradientDescentOptimizer(learning_rate_deacy).minimize(loss,global_step=global_step)
    
    #tf.group函数保证再一次迭代中，参数的train和参数的平均滑动都被执行
    train_op = tf.group(train_step,variable_average_op)
    
    #准确率
    correct_predict = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict,tf.float32))
    
    #定义一个初始化的操作
    init_op = tf.global_variables_initializer()
 
   # 开启一个会话，开始计算过程
    def calculate():
        with tf.Session() as sess: 
             init_op.run()
             validation_feed = {x:mnist.validation.images,y_:mnist.validation.labels} #验证数据
             test_feed = {x:mnist.test.images,y_:mnist.test.labels} #测试数据
             for i in range(TRAINING_STEPS):
                 if i % 1000 == 0:
                     validation_acc = sess.run(accuracy,feed_dict=validation_feed)
                     print('After %d training step(s), validation accuracy using average model is %g'%(i,validation_acc))
                
                 #训练数据
                 x_data,y_data = mnist.train.next_batch(BATCH_NODE)
                 train_feed = {x:x_data,y_:y_data}
                 sess.run(train_op,feed_dict=train_feed)
                 
                 #绘图
                 if i % 100 == 0:
                    l = sess.run(loss,feed_dict = train_feed)
                    i_c.append(i)
                    loss_c.append(l)
             
             #测试精度
             test_acc = sess.run(accuracy,feed_dict=test_feed)
             print("After %d training step(s), test accuracy using average model is %g" % (i, test_acc))
             return True

    flag = True
    while flag:
        flag = not calculate()

# 主程序入口
def main(args=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train(mnist)

    #绘图
    plt.plot(i_c,loss_c)
    plt.show()



# TensorFlow提供的一个主程序入口，tf.app.run()会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()
