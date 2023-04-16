import numpy as np
import pandas as pd
from toolbox import *

df = pd.read_csv('credit_card_fraud_updated.csv')
df = df.drop(['Unnamed: 0'], axis=1)
print(df.head())
print(len(df))
print(df.columns)
