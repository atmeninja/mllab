import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('/content/housing.csv')
df.head()
df.shape
df.info()
df.nunique()
df.isnull().sum()
df.duplicated().sum()
df['total_bedrooms'].median()
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

for i in df.iloc[:,2:7]:
  df[i] = df[i].astype('int')
df.head()

df.describe().T

Numerical = df.select_dtypes(include=[np.number]).columns
print(Numerical)

for col in Numerical:
  plt.figure(figsize=(10, 6))
  df[col].plot(kind='hist', title=col, bins=60, edgecolor='black')
  plt.ylabel('Frequency')
  plt.show()

for col in Numerical:
  plt.figure(figsize=(6, 6))
  sns.boxplot(df[col], color='blue')
  plt.title(col)
  plt.ylabel(col)
  plt.show()