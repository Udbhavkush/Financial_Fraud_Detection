import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from toolbox import *
from lvq import LVQ
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, roc_curve

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

df = df.drop(['isFlaggedFraud', 'step'], axis=1)

df_fraud = df[df['isFraud'] == 1]
len(df_fraud)
print('types observed in the fraudulent activities with their counts:')
print(df_fraud['type'].value_counts())

sns.countplot(x="type", hue='isFraud', data=df)
plt.title("histogram of different types with frequency of fraudulent activities")
plt.tight_layout()
plt.show()

le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

X = df.drop(columns=["isFraud"])
y = df["isFraud"]

sns.countplot(x='isFraud', data=df)
plt.title("An estimate of Imbalance in the dataset")
plt.tight_layout()
plt.show()

categorical_features = ['type']
numeric_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

ohe = OneHotEncoder(sparse=False)
scaler = StandardScaler()
ct = ColumnTransformer([
    ('ohe', ohe, categorical_features),
    ('scaler', scaler, numeric_features)
], remainder='passthrough')
# reference: chatgpt

X_scaled = ct.fit_transform(X)

# applying logistic regression
X_train1, X_test_val, y_train1, y_test_val = train_test_split(X_scaled, y, test_size=0.3, random_state=4)

X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=4)
# Splitting the dataset into train, validation, and test in ratio of 70:15:15

# Instantiate SMOTE
smote = SMOTE(random_state=4)
# reference: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html#imblearn.over_sampling.SMOTE.fit_resample

# Upsample the minority class of train set
X_train, y_train = smote.fit_resample(X_train1, y_train1)


# Fit a logistic regression model
lr = LogisticRegression(random_state=4)
lr.fit(X_train, y_train)

# Results on the train set
y_pred_train = lr.predict(X_train)
print('Results for logistic regression on train set:')
cm = confusion_matrix(y_train, y_pred_train)
print(cm)
cf = classification_report(y_train, y_pred_train)
print(cf)
sns.heatmap(cm, annot=True)
plt.title('Logistic regression on train set')
plt.show()

# Results on the validation set
y_pred_val = lr.predict(X_val)
print('Results for logistic regression on validation set:')
cm = confusion_matrix(y_val, y_pred_val)
print(cm)
cf = classification_report(y_val, y_pred_val)
print(cf)
sns.heatmap(cm, annot=True)
plt.title('Logistic regression on validation set')
plt.show()

# AUC Curves for train and validation sets of logistic regression
auc_lr_train = roc_auc_score(y_train, y_pred_train)
fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)

auc_lr_val = roc_auc_score(y_val, y_pred_val)
fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_val)

plt.plot(fpr_val, tpr_val, label=f"Val set = {auc_lr_val:.2f}")
plt.plot(fpr_train, tpr_train, label=f"Train set = {auc_lr_train:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.title('Comparing Logistic Regression on train and validation')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()


# comparable results on both train set and validation set for logistic regression
# and results are decent.

# Precision: the ratio of true positives to the total number of predicted positives.
# Recall: the ratio of true positives to the total number of actual positives.


# applying decision tree
clf = DecisionTreeClassifier(random_state=4)
clf.fit(X_train, y_train)

print('Results of decision tree on the train set:')
y_pred_train = clf.predict(X_train)
cm = confusion_matrix(y_train, y_pred_train)
print(cm)
cf = classification_report(y_train, y_pred_train)
print(cf)
sns.heatmap(cm, annot=True)
plt.title('Decision Tree on train set')
plt.show()

print('Results of decision tree on the validation set:')
y_pred_val = clf.predict(X_val)
cm = confusion_matrix(y_val, y_pred_val)
print(cm)
cf = classification_report(y_val, y_pred_val)
print(cf)
sns.heatmap(cm, annot=True)
plt.title('Decision Tree on validation set')
plt.show()

# AUC Curves for train and validation sets
auc_dt_train = roc_auc_score(y_train, y_pred_train)
fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)

auc_dt_val = roc_auc_score(y_val, y_pred_val)
fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_val)

plt.plot(fpr_val, tpr_val, label=f"Val set = {auc_dt_val:.2f}")
plt.plot(fpr_train, tpr_train, label=f"Train set = {auc_dt_train:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.title('Comparing Decision Tree on train and validation')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()


# applying KNN

k_values = list(range(1, 15))

precisions = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_val = knn.predict(X_val)
    precision = precision_score(y_val, y_pred_val)
    precisions.append(precision)

# Plot the results
plt.plot(k_values, precisions)
plt.xlabel('k')
plt.ylabel('Precision')
plt.title('KNN Precision vs. k')
plt.show()
print('Best precision using KNN when k = 2:', precisions[1])
# we get the maximum precision at k=2
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

print('Results of KNN on the train set:')
y_pred_train = knn.predict(X_train)
print(classification_report(y_train, y_pred_train))
cm = confusion_matrix(y_train, y_pred_train)
print(cm)
sns.heatmap(cm, annot=True)
plt.title('KNN (k=2) on train set')
plt.show()

print('Results of KNN on the validation set:')
y_pred_val = knn.predict(X_val)
print(classification_report(y_val, y_pred_val))
cm = confusion_matrix(y_val, y_pred_val)
print(cm)
sns.heatmap(cm, annot=True)
plt.title('KNN (k=2) on validation set')
plt.show()


# AUC Curves for train and validation sets
auc_knn_train = roc_auc_score(y_train, y_pred_train)
fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)

auc_knn_val = roc_auc_score(y_val, y_pred_val)
fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_val)

plt.plot(fpr_val, tpr_val, label=f"Val set = {auc_knn_val:.2f}")
plt.plot(fpr_train, tpr_train, label=f"Train set = {auc_knn_train:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.title('Comparing KNN on train and validation')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# applying LVQ
print('LVQ RUNNING!')
lv = LVQ(0.00001, 20)
lv.fit(X_train, y_train)
y_pred_val = lv.predict(X_val)
y_pred_train = lv.predict(X_train)
# precision = lv.score(y_val, y_pred_val)

print('LVQ classification on train set:')
print(classification_report(y_train, y_pred_train))
cm = confusion_matrix(y_train, y_pred_train)
print(cm)
sns.heatmap(cm, annot=True)
plt.title('LVQ on train set')
plt.show()

print('LVQ classification on validation set:')
print(classification_report(y_val, y_pred_val))
cm = confusion_matrix(y_val, y_pred_val)
print(cm)
sns.heatmap(cm, annot=True)
plt.title('LVQ on validation set')
plt.show()

# AUC Curves for train and validation sets
auc_lvq_train = roc_auc_score(y_train, y_pred_train)
fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)

auc_lvq_val = roc_auc_score(y_val, y_pred_val)
fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_val)

plt.plot(fpr_val, tpr_val, label=f"Val set = {auc_lvq_val:.2f}")
plt.plot(fpr_train, tpr_train, label=f"Train set = {auc_lvq_train:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.title('Comparing LVQ on train and validation')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()


# Comparing all the models on the test set now.

# LR
y_pred = lr.predict(X_test)
print('Results for logistic regression on test set:')
cm = confusion_matrix(y_test, y_pred)
print(cm)
cf = classification_report(y_test, y_pred)
print(cf)
sns.heatmap(cm, annot=True)
plt.title('Logistic regression on test set')
plt.show()

# DT
y_pred = clf.predict(X_test)
print('Results for decision tree on test set:')
cm = confusion_matrix(y_test, y_pred)
print(cm)
cf = classification_report(y_test, y_pred)
print(cf)
sns.heatmap(cm, annot=True)
plt.title('Decision Tree on test set')
plt.show()

# KNN
y_pred = knn.predict(X_test)
print('Results for KNN on test set:')
cm = confusion_matrix(y_test, y_pred)
print(cm)
cf = classification_report(y_test, y_pred)
print(cf)
sns.heatmap(cm, annot=True)
plt.title('KNN on test set')
plt.show()

# LVQ
y_pred = lv.predict(X_test)
print('Results for LVQ on test set:')
cm = confusion_matrix(y_test, y_pred)
print(cm)
cf = classification_report(y_test, y_pred)
print(cf)
sns.heatmap(cm, annot=True)
plt.title('LVQ on test set')
plt.show()


# AUC curves for all models on test set
y_pred = lv.predict(X_test)
auc_lvq_test = roc_auc_score(y_test, y_pred)
fpr_lvq, tpr_lvq, _ = roc_curve(y_test, y_pred)

y_pred = lr.predict(X_test)
auc_lr_test = roc_auc_score(y_test, y_pred)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred)

y_pred = clf.predict(X_test)
auc_dt_test = roc_auc_score(y_test, y_pred)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred)

y_pred = knn.predict(X_test)
auc_knn_test = roc_auc_score(y_test, y_pred)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred)

plt.plot(fpr_lvq, tpr_lvq, label=f"LVQ = {auc_lvq_val:.2f}")
plt.plot(fpr_lr, tpr_lr, label=f"LR  = {auc_lr_test:.2f}")
plt.plot(fpr_dt, tpr_dt, label=f"DT = {auc_dt_test:.2f}")
plt.plot(fpr_knn, tpr_knn, label=f"KNN = {auc_knn_test:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.title('Comparing all models on test set')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()


# applying LVQ on the non standardized dataset

X_train1, X_test_val, y_train1, y_test_val = train_test_split(X, y, test_size=0.3, random_state=4)

X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=4)

smote = SMOTE(random_state=4)

# Upsample the minority class of train set
X_train, y_train = smote.fit_resample(X_train1, y_train1)

# applying LVQ
#
lv = LVQ(0.00001, 20)
lv.fit(X_train, y_train)
y_pred_val = lv.predict(X_val)
y_pred_train = lv.predict(X_train)
# precision = lv.score(y_val, y_pred_val)

print('LVQ classification on train set:')
print(classification_report(y_train, y_pred_train))
cm = confusion_matrix(y_train, y_pred_train)
print(cm)
sns.heatmap(cm, annot=True)
plt.title('LVQ on train set')
plt.show()

print('LVQ classification on validation set:')
print(classification_report(y_val, y_pred_val))
cm = confusion_matrix(y_val, y_pred_val)
print(cm)
sns.heatmap(cm, annot=True)
plt.title('LVQ on validation set')
plt.show()

# AUC Curves for train and validation sets
auc_lvq_train = roc_auc_score(y_train, y_pred_train)
fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)

auc_lvq_val = roc_auc_score(y_val, y_pred_val)
fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_val)

plt.plot(fpr_val, tpr_val, label=f"Val set = {auc_lvq_val:.2f}")
plt.plot(fpr_train, tpr_train, label=f"Train set = {auc_lvq_train:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.title('Comparing LVQ on train and validation non standardized')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# LVQ
y_pred = lv.predict(X_test)
print('Results for LVQ on test set non standardized:')
cm = confusion_matrix(y_test, y_pred)
print(cm)
cf = classification_report(y_test, y_pred)
print(cf)
sns.heatmap(cm, annot=True)
plt.title('LVQ on test set')
plt.show()

auc_lvq_val = roc_auc_score(y_test, y_pred)
fpr_val, tpr_val, _ = roc_curve(y_test, y_pred)

plt.plot(fpr_val, tpr_val, label=f"Val set = {auc_lvq_val:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.title('LVQ on Test set non standardized')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()