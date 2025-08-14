# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Step 1: Load data
df = pd.read_csv("insurance.csv")

# Step 2: Separate features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# Step 3: Preprocessing for categorical columns
categorical_features = ["sex", "smoker", "region"]
numeric_features = ["age", "bmi", "children"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(), categorical_features)
], remainder="passthrough")

# Step 4: Create pipeline
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LinearRegression())
])

# Step 5: Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Step 6: Save the model
joblib.dump(pipeline, "insurance_model.pkl")

print("✅ Model trained and saved as insurance_model.pkl")
