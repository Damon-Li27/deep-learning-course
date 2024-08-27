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

def model():
    layer_dims = [train_X.shape[0], 20, 5, 1]
    parameters = opt.initialize_parameters(layer_dims)
    costs = []
    seed = 10
    for i in range(iterations):
        # 每次打乱顺序，随机种子不同，随机规则不同
        seed += 1
        mini_batch = opt.random_mini_batch(train_X, train_Y, seed, 128)
        cost = 0
        for minibatches in mini_batch:
            (X, Y) = minibatches
            a3, cache = opt.forward_propagation(X, parameters)
            cost = opt.compute_cost(a3, Y)
            gradients = opt.backward_propagation(X, Y, cache)
            opt.update_parameters(parameters, gradients, layer_dims, learning_rate)
        if i % 10 == 0:
            costs.append(cost)
            print("第" + str(i) + "次迭代后，成本值为：", cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per 10)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters

parameters = model()
predictions = opt.predict(train_X, train_Y, parameters)
# Plot decision boundary
plt.title("Model with mini batches Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
opt.plot_decision_boundary(lambda x: opt.predict_dec(parameters, x.T), train_X, train_Y)
