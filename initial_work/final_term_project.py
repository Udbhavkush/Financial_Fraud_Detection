import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from toolbox import *
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score

pd.set_option('display.expand_frame_repr', False)

data = pd.read_csv('credit_card_fraud_updated.csv')
df = data.drop(['Unnamed: 0', 'nameOrig', 'nameDest'], axis=1)
print(df.head())
print(len(df))
print(df.columns)
# print(df.isnull().sum())  # no null values in any of the column

# df_temp = df[df['isFraud'] == 1]
# df_temp = df_temp[df_temp['isFlaggedFraud'] == 1]
# len(df_temp)  # 16

# basically this piece of code tells us that whatever values are flagged fraud
# and which are actually fraud is 16. So, the column isFlaggedFraud is of no
# use for our current analysis. Basically, the rows with isFlaggedFraud value 1
# tells us that they can be potential fraudulent activities they are needed
# for further investigation. But, for this model, we are not concerned with that aspect.

df = df.drop(['isFlaggedFraud'], axis=1)

df_fraud = df[df['isFraud'] == 1]
len(df_fraud)
print('types observed in the fraudulent activities with their counts:')
print(df_fraud['type'].value_counts())

# df = pd.get_dummies(df, columns=["type"]) # one hot encoding for the 'type' column

le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])


X = df.drop(columns=["isFraud"])
y = df["isFraud"]

# Instantiate SMOTE
smote = SMOTE(random_state=4)
# reference: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html#imblearn.over_sampling.SMOTE.fit_resample

# Upsample the minority class
X_resampled, y_resampled = smote.fit_resample(X, y)

df_resampled = X_resampled.copy()
df_resampled['isFraud'] = y_resampled
print(df_resampled['isFraud'].value_counts())
# 1    91787
# 0    91787

# so upsampling of the data is done using SMOTE

# applying logistic regression
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=4)

# Fit a logistic regression model
lr = LogisticRegression(random_state=4)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Print the classification report and confusion matrix
print('Results for logistic regression:')

cm = confusion_matrix(y_test, y_pred)
print(cm)
cf = classification_report(y_test, y_pred)
print(cf)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
#
# # Fit a logistic regression model
# lr = LogisticRegression(random_state=4)
# lr.fit(X_train, y_train)
#
# # Predict on the test set
# y_pred = lr.predict(X_test)
#
# # Print the classification report and confusion matrix
# print('Results for logistic regression before upsampling:')
# cf2 = classification_report(y_test, y_pred)
# print(cf2)
# cm2 = confusion_matrix(y_test, y_pred)
# print(cm2)

# Precision: the ratio of true positives to the total number of predicted positives.
# Recall: the ratio of true positives to the total number of actual positives.


clf = DecisionTreeClassifier(random_state=4)
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the performance of the model
print('Results for Decision Tree:')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
clf = DecisionTreeClassifier(random_state=4)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

k_values = list(range(1, 31))

# Train a KNN model for each value of k and store the precision score
precisions = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    precision = precision_score(y_test, y_pred)
    precisions.append(precision)

# Plot the results
plt.plot(k_values, precisions)
plt.xlabel('k')
plt.ylabel('Precision')
plt.title('KNN Precision vs. k')
plt.show()
# we get the maximum precision at k=2
