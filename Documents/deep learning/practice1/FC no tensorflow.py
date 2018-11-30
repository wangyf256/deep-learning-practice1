#不使用tensorflow的全连接层（输入+隐藏+输出）

#引入
import numpy as np
import matplotlib.pyplot as plt
 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
 
 
def forward(X, w1, w2, b1, b2):
    z1 = np.dot(w1, X) + b1  # w1=h*n     X=n*m      z1=h*m
    A1 = sigmoid(z1)  # A1=h*m
    z2 = np.dot(w2, A1) + b2  # w2=1*h   z2=1*m
    A2 = sigmoid(z2)  # A2=1*m
    return z1, z2, A1, A2
 
 
def backward(y, X, A2, A1, z2, z1, w2, w1):
    n, m = np.shape(X)
    dz2 = A2 - y  # A2=1*m y=1*m
    dw2 = 1 / m * np.dot(dz2, A1.T)  # dz2=1*m A1.T=m*h dw2=1*h
    db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.dot(w2.T, dz2) * A1 * (1 - A1)  # w2.T=h*1 dz2=1*m z1=h*m A1=h*m dz1=h*m
    dw1 = 1 / m * np.dot(dz1, X.T)  # z1=h*m X'=m*n dw1=h*n
    db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)
    return dw1, dw2, db1, db2
 
 
def costfunction(A2, y):
    m, n = np.shape(y)
    J = np.sum(y * np.log(A2) + (1 - y) * np.log(1 - A2)) / m
    # J = (np.dot(y, np.log(A2.T)) + np.dot((1 - y).T, np.log(1 - A2))) / m
    return -J
 
 
# Data = np.loadtxt("gua2.txt")
# X = Data[:, 0:-1]
# X = X.T
# y = Data[:, -1]

#输入x和拟合数据y
X=np.random.rand(100,200)
n, m = np.shape(X)
y=np.random.rand(1,m)
#y = y.reshape(1, m)
 
#定义各参数
n_x = n  # size of the input layer
n_y = 1  # size of the output layer
n_h = 5  # size of the hidden layer
w1 = np.random.randn(n_h, n_x) * 0.01  # h*n
b1 = np.zeros((n_h, 1))  # h*1
w2 = np.random.randn(n_y, n_h) * 0.01  # 1*h
b2 = np.zeros((n_y, 1))
alpha = 0.1
number = 10000

#训练，用梯度下降
for i in range(0, number):
    z1, z2, A1, A2 = forward(X, w1, w2, b1, b2)
    dw1, dw2, db1, db2 = backward(y, X, A2, A1, z2, z1, w2, w1)
    w1 = w1 - alpha * dw1
    w2 = w2 - alpha * dw2
    b1 = b1 - alpha * db1
    b2 = b2 - alpha * db2
    J = costfunction(A2, y)
    if (i % 100 == 0):
        print(i)
    plt.plot(i, J, 'ro')
plt.show()
