import numpy as np
import h5py


def ReLU(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    # return .5 * (1 + np.tanh(.5 * x))


def load_dataset():
    # h5py.File函数用于创建或打开 HDF5 文件，文件模式常见的有 'r' 表示只读，'w' 表示写（如果文件已存在则覆盖），'a' 表示读写，如果文件不存在则创建。
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")  # "r"：制度
    # 将 train_dataset 对象中键为 train_set_x 的全部数据转换为 NumPy 数组。[:] 表示获取该数据的全部内容。
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
