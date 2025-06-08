import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)
values = np.random.rand(100)
labels = np.where(values[:50] <= 0.5, 'Class1', 'Class2')
labels = np.concatenate([labels, [None]*50])

df = pd.DataFrame({
  "Value": values,
  "Label": labels
})

X_train = df.loc[:49, ["Value"]]
y_train = df.loc[:49, "Label"]
X_test = df.loc[50:, ["Value"]]
true_labels = np.where(values[50:] <= 0.5, 'Class1', 'Class2')

k_values = [1, 2, 3, 4, 5, 20, 30]
for k in k_values:
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train, y_train)
  preds = knn.predict(X_test)
  acc = accuracy_score(true_labels, preds) * 100
  print(f"Accuracy for k={k}: {acc:.2f}%")
  print(preds)