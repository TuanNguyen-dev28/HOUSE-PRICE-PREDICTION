import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("data/house_data.csv")

# Target
y = df["SalePrice"]
X = df.drop(columns=["SalePrice"])

# All numeric features
numeric_features = X.columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num",
         Pipeline([
             ("imputer", SimpleImputer(strategy="median")),
             ("scaler", StandardScaler())
         ]),
         numeric_features)
    ]
)

# Base pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42))
    ]
)

# 🔥 Hyperparameter grid
param_grid = {
    "regressor__n_estimators": [200, 300, 500],
    "regressor__max_depth": [None, 10, 20],
    "regressor__min_samples_split": [2, 5],
}

# GridSearch
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train with tuning
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV R2:", grid_search.best_score_)

# Best model
best_model = grid_search.best_estimator_

# Predict
preds = best_model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("MAE:", mae)
print("RMSE:", rmse)
print("Test R2:", r2)

# Save model
joblib.dump(best_model, "models/best_model.pkl")

print("Tuned model saved successfully.")