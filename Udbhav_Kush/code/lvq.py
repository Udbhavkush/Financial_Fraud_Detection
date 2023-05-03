import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from toolbox import *
from sklearn.metrics import precision_score
np.random.seed(6202)

class LVQ(object):
    def __init__(self, alpha=0.01, epochs=20):
        self.alpha = alpha
        self.epochs = epochs
        # w2 = [[1, 0, 0], # need to uncomment this for iris dataset
        #       [0, 1, 0],
        #       [0, 0, 1]]
        w2 = [[1, 0],
              [0, 1]]

        self.w2 = np.array(w2)
        # for trying, I am taking just two neurons in the first layer
        # Tried different s1 values but s1=2 worked best for this problem

    def fit(self, X, y):
        print('LVQ RUNNING!')
        y = pd.Series(y)
        X = np.array(X)
        self.w1 = np.random.rand(self.w2.shape[0], len(X[0, :]))
        for epoch in range(self.epochs):
            print('epoch = ', epoch+1)
            for i in range(len(X)):
                p_temp = np.array(X[i, :]).reshape(-1, 1)
                n1 = np.zeros((self.w1.shape[0]))
                for j in range(self.w1.shape[0]):
                    n1[j] = -np.linalg.norm(self.w1[j].reshape(-1, 1) - p_temp)
                a1 = activation_func('compet', n1)
                a2 = self.w2 @ a1.reshape(-1, 1)
                winner_neuron = np.argmax(a1)  # ref: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
                out = np.argmax(a2)
                if out == y.iloc[i]:
                    self.w1[winner_neuron] += self.alpha * (p_temp.ravel() - self.w1[winner_neuron])
                else:
                    self.w1[winner_neuron] -= self.alpha * (p_temp.ravel() - self.w1[winner_neuron])
                    k = len(n1)-1
                    while True:
                        a1 = np.zeros(n1.shape)
                        ele = n1.argsort()[k]  # ref: https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
                        a1[ele] = 1
                        a2 = self.w2 @ a1.reshape(-1, 1)
                        out = np.argmax(a2)
                        if out == y.iloc[i]:
                            self.w1[ele] += self.alpha * (p_temp.ravel() - self.w1[ele])
                            break
                        else:
                            k -= 1
        return self

    def predict(self, X):
        X = np.array(X)
        y_pred = []
        for i in range(len(X)):
            p_temp = np.array(X[i, :]).reshape(-1, 1)
            n1 = np.zeros((self.w1.shape[0]))
            for j in range(self.w1.shape[0]):
                n1[j] = -distance(self.w1[j].reshape(-1, 1) - p_temp)

            a1 = activation_func('compet', n1)
            a2 = self.w2 @ a1.reshape(-1, 1)
            out = np.argmax(a2)
            y_pred.append(out)

        y_pred = np.array(y_pred)

        return y_pred

    def score(self, y_true, y_pred):
        return precision_score(y_true, y_pred)









