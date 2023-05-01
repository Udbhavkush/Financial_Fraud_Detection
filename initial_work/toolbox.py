import pandas as pd
import numpy as np


def activation_func(f, x):
    dict = {
        'logsigmoid': logsigmoid,
        'purelin': purelin,
        'tanh': tanh,
        'compet': compet
            }
    if f in dict:
        return dict[f](x)
    else:
        print('Function does not exist')


def logsigmoid(x):
    return 1 / (1 + np.exp(-x))


def purelin(x):
    return x


def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


def compet(x):
    max_val = np.max(x)
    res_arr = np.zeros_like(x)
    res_arr[x == max_val] = 1
    return res_arr


def derivative(f, x):
    match f:
        case 'logsigmoid':
            mat = logsigmoid(x) * (1 - logsigmoid(x))
            if x.shape[1] > 1:
                np.fill_diagonal(np.fliplr(mat), 0)
            return mat

        case 'purelin':
            mat = np.zeros(x.shape)
            return np.diag(np.ones(mat.shape[0]))

        case 'tanh':
            mat = 1 - (tanh(x) ** 2)
            if x.shape[0] > 1:
                np.fill_diagonal(np.fliplr(mat), 0)
            return mat


def distance(v):
    v = v.reshape(1, len(v))
    sum_squares = np.sum(np.square(v))

    return sum_squares**0.5


# def LVQ(X):
#     # since, there are two classes i.e. fraud and not fraud
#     # so, second layer will have two neurons.
#     w2 = [[1, 0],
#           [0, 1]]
#     w2 = np.array(w2)
#
#     # for trying, I am taking just two neurons in the first layer
#     w1 = np.random.rand(2, len(X.iloc[0, :]))
#
#     epochs = 1
#
#     for epoch in range(epochs):
#         print('epoch = ', epoch)
#         for i in range(len(X)):
#             p_temp = np.array(X.iloc[i, :]).reshape(-1, 1)
#             temp_arr = np.zeros((w1.shape[0]))
#             for j in range(w1.shape[0]):
#                 temp_arr[j] = -distance(w1[j].reshape(-1, 1) - p_temp)
#
#             a1 = activation_func('compet', temp_arr)
#             a2 = w2 @ a1.reshape(-1, 1)
#             winner_neuron = np.argmax(a1)  # ref: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
#             w1[winner_neuron] += 0.5 * (p_temp.ravel() - w1[winner_neuron])
#
#     print(w1)
#

