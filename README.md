# HOUSE-PRICE-PREDICTION
# 🏠 House Price Prediction (AI Engineer Project)

## 📌 Overview
This project builds a machine learning pipeline to predict house prices using the Kaggle House Prices dataset.

## 🧠 Key Features
- Full ML pipeline using scikit-learn
- ColumnTransformer for preprocessing
- Cross-validation (5-fold)
- Hyperparameter tuning with GridSearchCV
- RandomForest Regressor
- Production-ready FastAPI deployment
- Dockerized

## 📊 Model Performance
- Cross-validated R²: 0.86
- Test R²: 0.88
- RMSE: 0.14 (log scale)

## ⚙️ Tech Stack
- Python
- scikit-learn
- FastAPI
- Docker

## 🚀 Run Locally

```bash
pip install -r requirements.txt
python src/train.py
uvicorn app.main:app --reload