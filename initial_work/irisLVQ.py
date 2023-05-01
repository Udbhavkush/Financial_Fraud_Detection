from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from toolbox import *
from lvq import LVQ
from sklearn.metrics import classification_report, confusion_matrix, precision_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

print('LVQ RUNNING!')
lv = LVQ(0.001, 1000)
lv.fit(X_train, y_train)
y_pred = lv.predict(X_test)
print('LVQ classification:')
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)
a = [1, 2, 3, 4]
a = np.array(a)
a = a.reshape(-1, 1)

# import numpy as np

# a = np.array([5, 2, 7, 1, 8])
# idx_second_largest = a.argsort()[-2]
# print(idx_second_largest)

