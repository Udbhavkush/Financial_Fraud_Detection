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

