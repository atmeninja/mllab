import pandas as pd
data = pd.read_csv("/content/dataset.csv")

print(data)

def find_s_algorithm(data):
  """Implements the Find-S algorithm to find the most specific hypothesis."""
  attributes = data.iloc[:, :-1].values
  target = data.iloc[:,-1].values
  for i in range(len(target)):
    if target[i] == "Yes":
      hypothesis = attributes[i].copy()
    break
  for i in range(len(target)):
    if target[i] == "Yes":
      for j in range(len(hypothesis)):
        if hypothesis[j] != attributes[i][j]:
          hypothesis[j] = '?'
  return hypothesis
final_hypothesis = find_s_algorithm(data)
print("Most Specific Hypothesis:", final_hypothesis)