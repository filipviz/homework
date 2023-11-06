import pandas as pd

df = pd.read_csv('training_data_cleaned.csv')
print(df.head())
print(df.isnull().sum())
print(df.info())
