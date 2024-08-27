import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

from lr_utils import load_dataset

train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
'''
 # 显示图片
 plt.imshow(train_set_x_orig[56])
 plt.show()
 # 生成随机图像
 image = np.random.rand(30, 30)
plt.imshow(image)
plt.show()
'''

print(train_set_x_orig.shape)  # (209, 64, 64, 3)
print("the shape of y:" + str(train_set_y_orig.shape))  # (1, 209)

# 将训练集进行平铺，即每一张图片变成一个列数据
'''
# 使用如下代码转换是不对的
train_X_flatten_o = train_set_x_orig.reshape(64 * 64 * 3, 209)  # (12288, 209)
test_X_flatten_o = test_set_x_orig.reshape(64 * 64 * 3, -1) # (12288, 50)
正确写法应该是：
train_X_flatten_o = train_set_x_orig.reshape(209, 64 * 64 * 3).T  # (12288, 209)
test_X_flatten_o = test_set_x_orig.reshape(50, 64 * 64 * 3).T # (12288, 50)
更通用的方法如下
'''
train_X_flatten_o = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_X_flatten_o = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# 数据标准化，在RGB中不存在比255大的数据。
train_X_flatten = train_X_flatten_o / 255
print("the shape of X_train:" + str(train_X_flatten.shape))
# 表示将数组转换成64 * 64 * 3行的矩阵，具体多少列我们不知道，所以参数设为-1
test_X_flatten = test_X_flatten_o / 255
print("the shape of X_test:" + str(test_X_flatten.shape))

# 定义sigmoid函数
def sigmoid(z):
    '''
    # 容易出现极大的数据,导致np.exp运算溢出
    return 1 / (1 + np.exp(-z))
    '''
   # return .5 * (1 + np.tanh(.5 * z))
    return 1 / (1 + np.exp(-z))

# 定义全局变量
m = train_set_y_orig.shape[1] # 样本数
iteration = 5000 # 迭代次数
learning_rate = 0.02 # 学习率
print ("训练集的数量: " + str(m))
print ("测试集的数量: " + str(test_set_y_orig.shape[1]))

# w = np.random.randn(train_X_flatten.shape[0], 1);
w = np.zeros((train_X_flatten.shape[0], 1))
print ("参数的维度: " + str(w.shape))
b = 0

def forward_propagation():
    z = np.dot(w.T, train_X_flatten) + b
    a = sigmoid(z)
    # loss = np.negative(train_set_y_orig * (np.log(a + 1e-8)) + (1 - train_set_y_orig) * np.log(1 - a + 1e-8))
    # cost = 1 / m * np.sum(loss)
    cost = (-1 / m) * np.sum(train_set_y_orig * np.log(a) + (1 - train_set_y_orig) * np.log(1 - a))
    return cost,a


def back_propagation(a):
    global w, b # 若要在函数内修改全局变量，需要global关键字
    dz = a - train_set_y_orig
    dw = (1 / m) * np.dot(train_X_flatten, dz.T)
    db = (1 / m) * np.sum(dz)
    w = w - learning_rate * dw
    b = b - learning_rate * db

def train():
    costs = []
    for i in range(iteration):
        cost, a = forward_propagation()
        # 从数组中移除长度为 1 的维度。语法为 np.squeeze(arr, axis=None)，axis可选，指定要移除的轴
        cost = np.squeeze(cost)
        assert (cost.shape == ())
        back_propagation(a)
        if (i % 50 == 0):
            print(i, ":", cost)
            costs.append(cost)
    return costs
def predict(X):
    y_predict = np.zeros((1, X.shape[1]))
    z = np.dot(w.T, X) + b
    A = sigmoid(z)
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            y_predict[0, i] = 1
        else:
            y_predict[0, i] = 0
    return y_predict

def test():
    costs = train()
    y_predict_train = predict(train_X_flatten)
    y_predict_test = predict(test_X_flatten)
    # 若值相同，则相减一定为0，否则取1
    # 100 - np.mean(np.abs(y_predict_test - test_set_y_orig)) * 100
    y_predict_train_ratio = (1 - np.mean(np.abs(y_predict_train - train_set_y_orig))) * 100
    y_predict_test_ratio = (1 - np.mean(np.abs(y_predict_test - test_set_y_orig))) * 100
    print("Accuracy on train_set: " + str(y_predict_train_ratio))
    print("Accuracy on test_set: " + str(y_predict_test_ratio))
    # 描述costs的曲线
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per ten)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

test()
