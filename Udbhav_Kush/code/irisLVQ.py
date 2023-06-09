from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from toolbox import *
from lvq import LVQ
from sklearn.metrics import classification_report, confusion_matrix

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

print('LVQ RUNNING!')
lv = LVQ(0.05, 5)
lv.fit(X_train, y_train)
y_pred = lv.predict(X_test)
print('LVQ classification:')
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True)
plt.title('LVQ on IRIS')
plt.show()
