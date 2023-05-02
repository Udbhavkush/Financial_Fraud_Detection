import pandas as pd

df = pd.read_csv('credit_card_fraud.csv')
# reading of this line will give error as the initial data file was very large
# hence I could not upload it on GitHub.

df_is_fraud = df[df['isFraud'] == 1]
print(len(df_is_fraud))
df_not_fraud = df[df['isFraud'] == 0]
print(len(df_not_fraud))

df_new = df_is_fraud.copy()
df_new = pd.concat([df_new, df_not_fraud.sample(n=91787)])

print(len(df_new))

df_new.to_csv('credit_card_fraud_updated.csv')

#
# [[1.10349220e+00 3.99979312e+04 6.59972549e+06 6.62150877e+06
#   6.82245773e+05 6.51355420e+05]
#  [2.28987732e+00 5.69415202e+04 2.12361966e+04 9.26221689e+03
#   1.24939714e+04 5.75114767e+04]]