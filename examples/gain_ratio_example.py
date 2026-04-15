import pandas as pd
from sklearn_nominal import TreeClassifier
import os

# Load the dataset
# Using absolute path for the example to work in the dev environment
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/classification/alsol.csv"))
df = pd.read_csv(dataset_path)

# Features and target
X = df.drop(columns=["Quemado"])
y = df["Quemado"]

print("Dataset features:")
print(X.head())
print("\nUnique values per feature:")
print(X.nunique())

# 1. No penalization (importance = 0.0)
# This favors high-cardinality attributes like 'id', leading to overfitting
model_none = TreeClassifier(criterion="gain_ratio", attribute_penalization_importance=0.0, min_error_decrease=0)
model_none.fit(X, y)
print("\n--- Tree with Importance = 0.0 (Regular Entropy: Overfitted to 'id') ---")
print(model_none.pretty_print())

# 2. Balanced penalization (importance = 0.3)
# This penalizes 'id' enough to favor more generalizable features like 'Protector'
model_std = TreeClassifier(
    criterion="gain_ratio",
    attribute_penalization_importance=0.3,
    min_error_decrease=0,
)
model_std.fit(X, y)
print("\n--- Tree with Importance = 0.3 (Balanced: Uses 'Protector' and 'Pelo') ---")
print(model_std.pretty_print())

# 3. Standard penalization (importance = 1.0)
# Note: At higher importance, the penalization can be so strong that
# it discourages any split, resulting in a single leaf (root-only tree).
model_high = TreeClassifier(
    criterion="gain_ratio",
    attribute_penalization_importance=1.0,
    min_error_decrease=0,
)
model_high.fit(X, y)
print("\n--- Tree with Importance = 1.0 (Strongly Regularized: Root only) ---")
print(model_high.pretty_print())
