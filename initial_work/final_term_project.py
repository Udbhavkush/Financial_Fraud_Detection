import numpy as np
import pandas as pd
from toolbox import *
from imblearn.over_sampling import SMOTE
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

df = pd.get_dummies(df, columns=["type"])

X = df.drop(columns=["isFraud"])
y = df["isFraud"]

# Instantiate SMOTE
smote = SMOTE(random_state=42)

# Upsample the minority class
X_resampled, y_resampled = smote.fit_resample(X, y)


df_resampled = X_resampled.copy()
df_resampled['isFraud'] = y_resampled
print(df_resampled['isFraud'].value_counts())
# 1    91787
# 0    91787

# so upsampling of the data is done using SMOTE
