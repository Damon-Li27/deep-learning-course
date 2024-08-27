"""
正常梯度下降（未使用任何优化算法），对散点进行分类，计算精确度。

"""
from utils.opt_utils import plot_decision_boundary, load_dataset, initialize_parameters, forward_propagation, \
    backward_propagation, compute_cost, update_parameters, predict_dec
import numpy as np
import matplotlib.pyplot as plt


train_X, train_Y = load_dataset(is_plot=True)

print("训练集维度：", train_X.shape)

layer_dims = [train_X.shape[0], 20, 5, 1]

iterations = 1000
learning_rate = 0.05


def model():
    parameters = initialize_parameters(layer_dims)
    costs = []
    for i in range(iterations):

        a3, cache = forward_propagation(train_X, parameters)

        cost = compute_cost(a3, train_Y)
        if i % 20 == 0:
            costs.append(cost)
            print("第" + str(i) + "次迭代后，成本值为：", cost)

        gradients = backward_propagation(train_X, train_Y, cache)
        update_parameters(parameters, gradients, layer_dims, learning_rate)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    predictions = predict_dec(parameters, train_X)
    accuracy = (1 - np.mean(np.abs(predictions - train_Y))) * 100
    print(accuracy)
    plt.title("Model with Gradients Descent")
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


model()
