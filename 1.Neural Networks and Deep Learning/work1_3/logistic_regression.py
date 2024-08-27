'''
    逻辑回归对工具类中的数据集进行分类
'''

import numpy as np

import sklearn  # sklearn:免费软件机器学习库,具有各种分类，回归和聚类算法，包括支持向量机，随机森林，梯度提升，k均值和DBSCAN
from planar_utils import load_planar_dataset, plot_decision_boundary, sigmoid, tanh, back_sigmoid, back_tanh
import matplotlib.pyplot as plt  # 用于创建各种数据可视化图表,提供了丰富的函数和方法，用于绘制折线图、柱状图等多种类型的图表

# 随机种子，可以是任意整数。确保随机数的可重复性，有助于可重复性的实验或调试。只要在相同的位置设置相同的种子值，后续生成的随机数序列就会是相同的。
np.random.seed(42)

# 加载一个数据集，这个数据集是由代码生成的两个序列，自定义了其样本数量、构成特征等。
X, Y = load_planar_dataset()
print('X的维度：', str(X.shape), '，Y的维度：', str(Y.shape))  # X的维度： (2, 400) ，Y的维度： (1, 400)

# 绘制图像
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
plt.show()  # 可以看出，数据的分布不是线性可分的

# 至此我们拥有了一个特征为（x1,x2）的矩阵X，一个标签（红色：0，蓝色：1））的Y
'''
    sklearn.linear_model.LogisticRegressionCV() 函数用于逻辑回归的交叉验证，
    它通过自动进行交叉验证来选择合适的正则化参数（例如 C 值），以优化逻辑回归模型的性能。
'''
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T[:, 0])
# #绘制分界线
plot_decision_boundary(lambda x: clf.predict(x), X, Y[0, :])
plt.title("Logistic Regression")
plt.show()

LR_predictions = clf.predict(X.T)
# #正确率只有47%，分类效果较差，并不能很好的进行分类。
print(clf.score(X.T, Y.T))