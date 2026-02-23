import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from data_processing import load_data, preprocess_data, split_features_target

# Load data
df = load_data("data/house_data.csv")
df = preprocess_data(df)
X, y = split_features_target(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(),
    "XGBoost": XGBRegressor()
}

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"{name} - MAE: {mae:.2f}, R2: {r2:.3f}")

    if r2 > best_score:
        best_score = r2
        best_model = model

joblib.dump(best_model, "models/best_model.pkl")
print("Best model saved.")