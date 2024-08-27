"""
小批量梯度下降实现散点分类
"""
import utils.opt_utils as opt
import numpy as np
import matplotlib.pyplot as plt

train_X, train_Y = opt.load_dataset(is_plot=True)

print("训练集维度：", train_X.shape)

iterations = 1000
learning_rate = 0.001
# learning_rate = 0.00001


def model():
    seed = 42
    layer_dims = [train_X.shape[0], 20, 5, 1]
    parameters = opt.initialize_parameters(layer_dims)
    v = opt.initialize_velocity_momentum(parameters)
    s = opt.initialize_params_RMSprop(parameters)
    costs = []
    t = 0
    for i in range(iterations):
        seed += 1
        mini_batch = opt.random_mini_batch(train_X, train_Y, seed, 128)
        cost = 0
        for minibatches in mini_batch:
            (X, Y) = minibatches
            a3, cache = opt.forward_propagation(X, parameters)
            cost = opt.compute_cost(a3, Y)
            gradients = opt.backward_propagation(X, Y, cache)
            t += 1
            parameters, v, s = opt.update_parameters_with_adam(parameters, gradients, v, s, layer_dims, learning_rate, t)
        if i % 10 == 0:
            costs.append(cost)
            print("第" + str(i) + "次迭代后，成本值为：", cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per 10)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters

parameters = model()
predictions = opt.predict(train_X, train_Y, parameters)
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
opt.plot_decision_boundary(lambda x: opt.predict_dec(parameters, x.T), train_X, train_Y)



