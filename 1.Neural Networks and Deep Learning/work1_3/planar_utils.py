'''
planar_utils provide various useful functions used in this assignment
'''

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)


'''
    sigmoid函数的导数：
        g’(z) = g(z)（1-g(z)）

        tanh函数的导数：
        g’(z) = 1- ( g(z) )2
'''


def sigmoid(Z):
    """
    Compute the sigmoid of Z

    Arguments:
    Z -- A scalar or numpy array of any size.

    Return:
     -- sigmoid(x)
    """
    return 1 / (1 + np.exp(-Z))


def back_sigmoid(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))


# 激活函数tanh
def tanh(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))


def back_tanh(Z):
    return 1 - np.square(tanh(Z))


def load_planar_dataset():
    np.random.seed(1)
    m = 400  # 样本数量
    N = int(m / 2)  # 每类的样本数量
    D = 2  # 维度：2维
    X = np.zeros((m, D))  # 数据矩阵，其中每行是一个样本
    Y = np.zeros((m, 1), dtype='uint8')  # 标签向量，0表示红色，1表示蓝色
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))  # 生成两个序列。j取0和1时，分别生成（0~N-1）和（N~2N-1）的序列，即[0,1,...,N-1]和[N...]
        # 下面是样本的两个feature
        # np.linspace 用于创建一个在指定范围内均匀分布的数字序列。三个参数表示起始值、结束值和生成数量。
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        '''
          # np.c_用于按行连接（即左右相加）两个或多个数组，要求这些数组的行数相等。
          比如： a = np.array([(1, 2, 3), (7, 8, 9)])
                b = np.array([(4, 5, 6), (1, 2, 3)])
                c = np.c_[a, b] # 结果：[[1 2 3 4 5 6]
                                        [7 8 9 1 2 3]]
        '''
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j  # 第一个序列的标签全部取0，第二个全部取1

    X = X.T
    Y = Y.T

    return X, Y


def load_extra_datasets():
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2,
                                                                  n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure
