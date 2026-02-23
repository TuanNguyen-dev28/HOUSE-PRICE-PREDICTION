from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="House Price Prediction API")

model = joblib.load("models/best_model.pkl")

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"predicted_price": float(prediction[0])}