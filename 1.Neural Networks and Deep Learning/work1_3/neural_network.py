import numpy as np

import sklearn  # sklearn:免费软件机器学习库,具有各种分类，回归和聚类算法，包括支持向量机，随机森林，梯度提升，k均值和DBSCAN
from planar_utils import load_planar_dataset, plot_decision_boundary, sigmoid, tanh, back_sigmoid, back_tanh
import matplotlib.pyplot as plt  # 用于创建各种数据可视化图表,提供了丰富的函数和方法，用于绘制折线图、柱状图等多种类型的图表

# 随机种子，可以是任意整数。确保随机数的可重复性，有助于可重复性的实验或调试。只要在相同的位置设置相同的种子值，后续生成的随机数序列就会是相同的。
np.random.seed(1)

# 加载一个数据集，这个数据集是由代码生成的两个序列，自定义了其样本数量、构成特征等。
X, Y = load_planar_dataset()
print('X的维度：', str(X.shape), '，Y的维度：', str(Y.shape))  # X的维度： (2, 400) ，Y的维度： (1, 400)

'''
    --------------- 神经网络对工具类中的数据集进行分类 -------------
'''


def layer_sizes(X, Y):
    n_x = X.shape[0]  # size of input layer
    n_h = 4
    n_y = Y.shape[0]  # size of output layer
    return n_x, n_h, n_y


# 初始化参数
def initialize_parameters(n_x, n_h, n_y):
    """
    n_x:输入层的节点数
    n_h:隐藏层的节点数
    n_y:输出层的节点数
    """
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# 前向传播
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache


# 计算代价函数(交叉熵)
def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    loss = Y * np.log(A2) + (1 - Y) * np.log(1 - A2)
    cost = (-1 / m) * np.sum(loss)
    # cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.

    assert (isinstance(cost, float))
    return cost


# 反向传播
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    # np.power用于计算数组元素的幂次方
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
#
# grads = backward_propagation(parameters, cache, X_assess, Y_assess)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("db2 = "+ str(grads["db2"]))

# 参数更新
def update_parameters(parameters, grads, learning_rate=1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# 模型整合
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):

        A2, cache = forward_propagation(X, parameters)

        cost = compute_cost(A2, Y, parameters)

        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    # 如果该元素大于 0.5 ，则对应的predictions中元素为 True ；否则为 False 。
    predictions = (A2 > 0.5)
    return predictions


# 调用单隐藏层神经网络
parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0, :])
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()

# 更改隐藏层的节点数量
hidden_layer_sizes = np.array([1, 2, 3, 4, 5, 20, 50])
for i in range(len(hidden_layer_sizes)):
    # 绘制分界线
    plt.subplot(5, 2, i + 1)
    n_h = hidden_layer_sizes[i]
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0, :])
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print(n_h, '个隐藏节点的准确率为：', accuracy, '%')
    plt.show()
