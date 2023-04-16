import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from toolbox import *
np.random.seed(6202)


p = np.array([[-1, 1, 1, -1],
              [-1, 1, -1, 1]])
t = np.array([[1, 1, 0, 0],
              [0, 0, 1, 1]])

color = ['red' if t1 == 1 else 'blue' for t1 in t[0]]
# ref: https://stackoverflow.com/questions/56088121/scatter-plot-with-conditions

# suppose we have two subclasses for each class, and we have two classes

# that means
w2 = [[1, 1, 0, 0],
      [0, 0, 1, 1]]
w2 = np.array(w2)

w1 = np.random.rand(4, p.shape[0])
color2 = ['black', 'green', 'purple', 'orange']
plt.scatter(p[0], p[1], color=color)
plt.scatter(w1[:, 0], w1[:, 1], color=color2)
plt.grid()
plt.show()
# print(w1)
epochs = 1

for epoch in range(epochs):
    for i in range(p.shape[1]):
        p_temp = np.array([p[0][i], p[1][i]])
        temp_arr = np.zeros((w1.shape[0]))
        for j in range(w1.shape[0]):
            temp_arr[j] = -distance(w1[j] - p_temp)

        a1 = activation_func('compet', temp_arr)
        a2 = w2 @ a1.reshape(-1, 1)
        winner_neuron = np.argmax(a1)  #ref: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
        w1[winner_neuron] += 0.5 * (p_temp - w1[winner_neuron])

print(w1)
color2 = ['black', 'green', 'purple', 'orange']
plt.scatter(p[0], p[1], color=color)
plt.scatter(w1[:, 0], w1[:, 1], color=color2)
plt.grid()
plt.show()


