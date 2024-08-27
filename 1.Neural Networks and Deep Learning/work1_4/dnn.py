import numpy as np
from lr_utils import load_dataset, ReLU, sigmoid
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import cv2 as cv

train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
# 数据集标准化
train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
train_x = train_set_x / 255
test_x = test_set_x / 255
train_y = train_set_y_orig
test_y = test_set_y_orig

print("训练集维度: ", train_x.shape)
print("测试集维度: ", test_x.shape)
print("训练集标签维度: ", train_y.shape)
print("测试集标签维度: ", test_y.shape)
# 样本数
train_m = train_y.shape[1]
test_m = test_y.shape[1]
print("训练集数量: ", train_m)
print("测试集数量: ", test_m)

# 设置随机种子
np.random.seed(42)
# 学习率 todo learning_rate = 0.05
learning_rate = 0.0075
iterations = 2500
# 层级和隐藏节点
layer_dims = [train_x.shape[0], 20, 7, 5, 1]
# 各隐藏层激活函数
activate_function_arr = ['relu', 'relu', 'relu', 'sigmoid']


# 初始化参数
def init_param():
    '''
        参数初始化
    :return:
    '''
    # 使用数组存储参数，便于区分层次，第0层为输入层，不需要设置参数，故设为-1
    W = [-1]
    b = [-1]
    # 前layer-1层使用ReLU激活函数，第layer层使用sigmoid激活函数
    for i in range(1, len(layer_dims)):
        # 通常把权重矩阵初始化成非常小的随机数
        # todo  W.append(np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01)
        W.append(np.random.randn(layer_dims[i], layer_dims[i - 1]) / np.sqrt(layer_dims[i - 1]))
        b.append(np.zeros((layer_dims[i], 1)))
    return W, b


def forward_propagation(W, b, X, Y):
    '''
    前向传播，Z=Wx+b, A=g(Z)
    :param W: 权重参数
    :param b: 偏移量参数
    :param X: 输入
    :param Y: 输出
    :return:
    '''
    # 初始： A0 = X
    A_cur = X
    # 缓存A0
    cache_A = [X]
    # 第0层不需要Z，设为-1
    cache_Z = [-1]
    # 前layer-1层使用ReLU函数，最后一层使用sigmoid函数
    for i in range(1, len(W)):  # 1,2,3,4
        A_prev = A_cur
        # 线性部分
        Z = np.dot(W[i], A_prev) + b[i]
        cache_Z.append(Z)
        if activate_function_arr[i - 1] == 'relu':
            A_cur = ReLU(Z)
        if activate_function_arr[i - 1] == 'sigmoid':
            A_cur = sigmoid(Z)
        # 缓存A
        cache_A.append(A_cur)
    Y_hat = A_cur

    assert (Y_hat.shape == Y.shape)

    # 计算成本
    cost = compute_cost(Y_hat, Y)
    cost = np.squeeze(cost)
    return cost, cache_A, cache_Z


def compute_cost(y_hat, y):
    '''
    代价函数
    :param y_hat:
    :param y: 训练集标签
    :return:
    '''
    return (- 1 / y.shape[1]) * np.sum(np.multiply(y, np.log(y_hat)) + np.multiply(1 - y, np.log(1 - y_hat)))


def ReLU_backward(x):
    # return np.where(x > 0, 1, 0)
    return 1.0 * (x > 0)


def backward_propagation(W, b, cache_Z, cache_A):
    '''
    反向传播
    :param W: 权重参数
    :param b: 偏移值
    :param cache_Z: 线性值缓存
    :param cache_A: 非线性值缓存
    :return:
    '''
    dA_prev = 0
    L = len(W) - 1
    AL = cache_A[len(cache_A) - 1]
    for i in range(L, 0, -1):  # 4,3,2,1
        if activate_function_arr[i - 1] == 'sigmoid':
            dZ = AL - train_y
        if activate_function_arr[i - 1] == 'relu':
            # dZ[l] = dA[l] * g[l]'(Z[l])
            dZ = np.multiply(dA_prev, ReLU_backward(cache_Z[i]))

        dW = (1 / train_m) * np.dot(dZ, cache_A[i - 1].T)
        assert (dW.shape == W[i].shape)

        db = (1 / train_m) * np.sum(dZ, axis=1, keepdims=True)
        assert (db.shape == b[i].shape)

        W[i] = W[i] - learning_rate * dW
        b[i] = b[i] - learning_rate * db
        dA_prev = np.dot(W[i].T, dZ)
    return W, b


def model():
    '''
    数据训练
    :return:
    '''
    np.random.seed(1)

    W, b = init_param()

    costs = []
    for i in range(iterations):
        cost, cache_A, cache_Z = forward_propagation(W, b, train_x, train_y)
        if (i % 100 == 0):
            costs.append(cost)
            print("第", i, "次迭代，成本值为：", np.squeeze(cost))
        W, b = backward_propagation(W, b, cache_Z, cache_A)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return W, b


def predict(W, b, X, Y):
    """
    训练结果预测
    :param W:
    :param b:
    :param X:
    :param Y:
    :return:
    """
    y_predict = np.zeros((1, X.shape[1]))
    cost, cache_A, cache_Z = forward_propagation(W, b, X, Y)
    Y_hat = cache_A[len(cache_A) - 1]
    for i in range(Y.shape[1]):
        if Y_hat[0, i] > 0.5:
            y_predict[0, i] = 1
        else:
            y_predict[0, i] = 0
    return y_predict


W, b = model()
Y_predict_train = predict(W, b, train_x, train_y)
Y_predict_test = predict(W, b, test_x, test_y)

# 若值相同，则相减一定为0，否则取1
Y_predict_train_ratio = (1 - np.mean(np.abs(Y_predict_train - train_y))) * 100
Y_predict_test_ratio = (1 - np.mean(np.abs(Y_predict_test - test_y))) * 100
print("Accuracy on train_set: " + str(Y_predict_train_ratio))
print("Accuracy on test_set: " + str(Y_predict_test_ratio))


def print_mislabeled_images():
    """
	展现预测和实际不同的图像。
	    X - 数据集
	    y - 实际的标签
	    p - 预测
    """
    y = test_y
    X = test_x
    a = Y_predict_test + y
    # 如果结果相同，p + y的结果应为0或2，否则为1。`where` 函数用于根据指定的条件返回满足条件的元素的索引
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(Y_predict_test[0, index])].decode("utf-8") + " \n Class: " + classes[
                y[0, index]].decode(
                "utf-8"))
        plt.show()


# print_mislabeled_images()

def test_my_cat(my_image, lable):
    '''
    自定义图片识别
    :param my_image:
    :param lable:
    :return:
    '''
    ## START CODE HERE ##
    my_label_y = lable  # the true class of your image (1 -> cat, 0 -> non-cat)
    ## END CODE HERE ##

    fname = "images/" + my_image
    # 使用 imageio 库读取一个图像文件，并将其转换为一个 NumPy 数组。
    image = np.array(imageio.imread(fname))
    # resize 函数将图像 image 的大小调整为 64×64。.reshape((64 * 64 * 3, 1))：对调整大小后的图像进行形状重塑
    my_image_x = cv.resize(image, (64, 64)).reshape((64 * 64 * 3, 1))
    my_image_predicted = predict(W, b, my_image_x, my_label_y)

    #  plt.imshow(image)
    print("y = " + str(np.squeeze(my_image_predicted)) + ", your L-layer model predicts a \"" + classes[
        int(np.squeeze(my_image_predicted)),].decode("utf-8") + "\" picture.")

    # plt.show()

# 测试结果总是1，尚不知原因
my_image1 = "no_1.jpg"
my_label_y_1 = np.zeros((1, 1))
test_my_cat(my_image1, my_label_y_1)

my_image2 = "no_2.png"
my_label_y_2 = np.zeros((1, 1))
test_my_cat(my_image2, my_label_y_2)

my_image3 = "yes_1.png"
my_label_y_3 = np.array([[1]])
test_my_cat(my_image3, my_label_y_3)

my_image4 = "yes_2.png"
my_label_y_4 = np.array([[1]])
test_my_cat(my_image4, my_label_y_4)
