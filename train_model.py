"""
train_model.py
Trains a simple churn prediction model on gold_churn_data.csv and saves:
  - app/transformer.pkl  (the preprocessing pipeline)
  - app/model.pkl        (the trained classifier)

Run once, from the project root:
    python train_model.py
"""

import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---- 1. Load data ----
df = pd.read_csv("gold_churn_data.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Creating X and y
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Step 1: Drop the 'customerID' column
X = X.drop(columns=["customerID"])

# Step 2: Convert 'TotalCharges' to numeric (handles spaces or non-numeric values)
X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

# Step 3: Convert target column 'y' to binary values
y = y.map({"Yes": 1, "No": 0})

# Step 4: Identify column types
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Step 5: Define preprocessing pipeline (no model yet)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="mean"), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ]
)

# ---- 2. Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ---- 3. Fit preprocessing on train, transform both ----
X_train_t = preprocessor.fit_transform(X_train)
X_test_t = preprocessor.transform(X_test)

# ---- 4. Train model ----
model = RandomForestClassifier(max_depth=6, random_state=42)
model.fit(X_train_t, y_train)

# ---- 5. Quick evaluation ----
y_pred = model.predict(X_test_t)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

# ---- 6. Save transformer + model ----
with open("app/transformer.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

with open("app/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nSaved app/transformer.pkl and app/model.pkl")
