import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from toolbox import *
from sklearn.metrics import precision_score
np.random.seed(6202)


# p = np.array([[-1, 1, 1, -1],
#               [-1, 1, -1, 1]])
# t = np.array([[1, 1, 0, 0],
#               [0, 0, 1, 1]])
#
# color = ['red' if t1 == 1 else 'blue' for t1 in t[0]]
# # ref: https://stackoverflow.com/questions/56088121/scatter-plot-with-conditions
#
# # suppose we have two subclasses for each class, and we have two classes
#
# # that means
# w2 = [[1, 1, 0, 0],
#       [0, 0, 1, 1]]
# w2 = np.array(w2)
#
# # for trying, I am taking just two neurons in the first layer
# w1 = np.random.rand(2, p.shape[0])
# color2 = ['black', 'green', 'purple', 'orange']
# plt.scatter(p[0], p[1], color=color)
# plt.scatter(w1[:, 0], w1[:, 1], color=color2)
# plt.grid()
# plt.show()
# print(w1)
# epochs = 50
#
# for epoch in range(epochs):
#     for i in range(p.shape[1]):
#         p_temp = p[:, i]
#         temp_arr = np.zeros((w1.shape[0]))
#         for j in range(w1.shape[0]):
#             temp_arr[j] = -distance(w1[j] - p_temp)
#
#         a1 = activation_func('compet', temp_arr)
#         a2 = w2 @ a1.reshape(-1, 1)
#         winner_neuron = np.argmax(a1)  #ref: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
#         w1[winner_neuron] += 0.5 * (p_temp - w1[winner_neuron])
#
# print(w1)
# color2 = ['black', 'green', 'purple', 'orange']
# plt.scatter(p[0], p[1], color=color)
# plt.scatter(w1[:, 0], w1[:, 1], color=color2)
# plt.grid()
# plt.show()

class LVQ(object):
    def __init__(self, alpha=0.01, epochs=20):
        self.alpha = alpha
        self.epochs = epochs
        w2 = [[1, 0],
              [0, 1]]

        self.w2 = np.array(w2)
        # for trying, I am taking just two neurons in the first layer
        # Tried different s1 values but s1=2 worked best for this problem

    def fit(self, X, y):
        X = np.array(X)
        self.w1 = np.random.rand(2, len(X[0, :]))
        for epoch in range(self.epochs):
            print('epoch = ', epoch+1)
            for i in range(len(X)):
                p_temp = np.array(X[i, :]).reshape(-1, 1)
                temp_arr = np.zeros((self.w1.shape[0]))
                for j in range(self.w1.shape[0]):
                    # temp_arr[j] = -distance(self.w1[j].reshape(-1, 1) - p_temp)
                    temp_arr[j] = -np.linalg.norm(self.w1[j].reshape(-1, 1) - p_temp)
                a1 = activation_func('compet', temp_arr)
                a2 = self.w2 @ a1.reshape(-1, 1)
                winner_neuron = np.argmax(a1)  # ref: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
                out = np.argmax(a2)
                # self.w1[winner_neuron] += self.alpha * (p_temp.ravel() - self.w1[winner_neuron])
                if out == y.iloc[i]:
                    self.w1[winner_neuron] += self.alpha * (p_temp.ravel() - self.w1[winner_neuron])
                else:
                    self.w1[winner_neuron] -= self.alpha * (p_temp.ravel() - self.w1[winner_neuron])
            print(self.w1)
        return self

    def predict(self, X):
        X = np.array(X)
        y_pred = []
        for i in range(len(X)):
            p_temp = np.array(X[i, :]).reshape(-1, 1)
            temp_arr = np.zeros((self.w1.shape[0]))
            for j in range(self.w1.shape[0]):
                temp_arr[j] = -distance(self.w1[j].reshape(-1, 1) - p_temp)

            a1 = activation_func('compet', temp_arr)
            a2 = self.w2 @ a1.reshape(-1, 1)
            out = np.argmax(a2)
            y_pred.append(out)

        y_pred = np.array(y_pred)

        return y_pred

    def score(self, y_true, y_pred):
        return precision_score(y_true, y_pred)









