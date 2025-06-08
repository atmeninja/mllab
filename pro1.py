import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 import seaborn as sns
 from sklearn.datasets import fetch_california_housing

df = fetch_california_housing(as_frame=True).frame
numerical_features = df.select_dtypes(include=[np.number]).columns

for i, feature in enumerate(numerical_features):
  plt.figure(figsize=(5, 3))
  sns.histplot(df[feature], kde=True, bins=30, color='blue')
  plt.title(f'Distribution of {feature}')
  plt.show()

for i, feature in enumerate(numerical_features):
  plt.figure(figsize=(6, 3))
  sns.boxplot(x=df[feature], color='orange')
  plt.title(f'Box Plot of {feature}')
  plt.show()

print("Outliers Detection:")
outliers_summary = {}
for feature in numerical_features:
  Q1 = df[feature].quantile(0.25)
  Q3 = df[feature].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
  outliers_summary[feature] = len(outliers)
  print(f"{feature}: {len(outliers)} outliers")
print("\nDataset Summary:")
print(df.describe())