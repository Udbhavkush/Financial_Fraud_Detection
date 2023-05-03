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
