# House Price Prediction AI

An AI-powered house price prediction system for Vietnamese real estate, trained on **44,448** property listings. 

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red?logo=xgboost&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-9.0-green?logo=pytest&logoColor=white)

## ✨ Technical Highlights (What makes this project special)

- **Domain Knowledge Enforcement**: Uses **XGBoost Monotone Constraints** to force the AI to strictly obey real estate logic (e.g., *Price must increase if Area/Floors increase* or *Price must be higher if it has a Legal Certificate*).
- **Data Leakage Prevention**: Uses **Target Encoding with Out-of-Fold (OOF)** technique to encode high-cardinality location features (District, City) without leaking test data into training.
- **Ordinal Feature Encoding**: Carefully mapped categorical features like `Legal_status` (Have certificate > Sale contract > In progress > Pending) and `Furniture_state` (Full > Basic > Empty) to numerical ranks.
- **Data Augmentation**: Synthetically generated >15k realistic records to cover edge cases (e.g., houses > 100m², sparse properties in central districts).
- **Ensemble Model**: Combines XGBoost + Random Forest with optimized weights (60% XGB + 40% RF) for maximum stability.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/TuanNguyen-dev28/HOUSE-PRICE-PREDICTION.git
cd HOUSE-PRICE-PREDICTION
```

### 2. Create virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Web Application
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

---

## 📂 Project Structure

```text
E:\AI\House-price-regression\
├── data\                       # Dataset directory
│   ├── processed\              # Preprocessed data ready for training
│   └── raw\                    # Raw dataset (house_data.csv)
│
├── models\                     # Saved Models & Weights
│   ├── xgboost_model.pkl
│   ├── random_forest_model.pkl
│   ├── ensemble_weights.json
│   ├── feature_importances.csv
│   └── location_encodings.json
│
├── notebooks\                  # EDA & Visualizations
│   ├── eda.ipynb
│   └── data_evaluation_plots.png
│
├── scripts\                    # Utilities & Testing Scripts
│   ├── data_evaluation.py
│   ├── evaluate_model.py       # Comprehensive edge case testing
│   ├── update_encodings.py
│   └── data_generation\        # Synthetic data generation & augmentation
│       ├── augment_edge_cases.py
│       └── clean_data.py
│
├── src\                        # Core ML Pipeline
│   ├── predict.py              # Prediction & Ensemble logic
│   ├── preprocess.py           # Preprocessing (OOF Target & Ordinal Encoding)
│   ├── train.py                # Model training (CV & Monotone constraints)
│   └── utils.py                
│
├── static\                     # Web Frontend
│   ├── app.js
│   ├── index.html
│   └── style.css
│
├── tests\                      # Pytest unit tests
│   ├── test_predict.py
│   └── test_preprocess.py
│
└── app.py                      # Flask Application
```

---

## 📊 Dataset & Features

**Total Records**: 44,448 (After augmentation and cleaning)

| Feature | Description | Type / Encoding |
|--------|-------------|-----------------|
| **Area** | Property area in m² | Numeric (Monotonic ⬆️) |
| **Floors** | Number of floors | Numeric (Monotonic ⬆️) |
| **District/City** | Location | Target Encoding (OOF) |
| **Legal Status** | Certificate state | Ordinal Encoding (Monotonic ⬆️) |
| **Furniture State** | Furnishing level | Ordinal Encoding (Monotonic ⬆️) |
| **Bedrooms / Bathrooms** | Room counts | Numeric |
| **Frontage / Access Road** | Widths in meters | Numeric |
| **Price** | Price in billions VND | **Target Variable** |

---

## 🧠 Model Performance

| Model | Test R² | Test MAE | Test RMSE |
|-------|---------|----------|-----------|
| Random Forest | 0.6155 | 3.85 tỷ | 10.71 tỷ |
| XGBoost (with constraints)| 0.6318 | 4.39 tỷ | 10.48 tỷ |
| **Ensemble (60% XGB + 40% RF)** | **0.6402** | **4.00 tỷ** | **10.36 tỷ** |

> **Note**: While R² is ~0.64, the model's logical consistency is **100% perfect** across all edge cases (tested via `scripts/evaluate_model.py`) thanks to XGBoost's Monotone Constraints. It never predicts a 100m² house to be cheaper than an identical 50m² house.

---

## ✅ Project Checklist

This checklist tracks the completed and upcoming features for this project.

### Phase 1: Data Collection & Cleaning
- [x] Crawl/Collect real estate data.
- [x] Handle Missing Values (Imputation).
- [x] Remove Outliers (Winsorization) & Duplicates.
- [x] Data augmentation to solve sparse segments (e.g. Area > 100m²).

### Phase 2: Feature Engineering
- [x] Target Encoding for categorical locations (`District`, `City`).
- [x] **Out-Of-Fold (OOF)** implementation to prevent data leakage.
- [x] Ordinal Encoding for ordered categorical features (`Legal_status`, `Furniture_state`).

### Phase 3: Modeling & Optimization
- [x] Train baseline models (Linear Regression).
- [x] Train advanced models (Random Forest, XGBoost).
- [x] Implement **Monotone Constraints** in XGBoost for business logic enforcement.
- [x] Create Weighted Ensemble predictor.
- [x] Optimize Ensemble Weights via Grid Search.
- [x] 5-Fold Cross Validation.

### Phase 4: MLOps & Production
- [x] Modularize codebase (`src/`, `scripts/`, `models/`).
- [x] Unit Testing with Pytest (`tests/`).
- [x] Edge-case scenario testing (`evaluate_model.py`).
- [x] Develop REST API with Flask (`app.py`).
- [x] Build Interactive Web UI (HTML/CSS/JS).
- [ ] Implement CI/CD Pipeline (GitHub Actions).
- [ ] Dockerize the application.
- [ ] Experiment Tracking (MLflow or Weights & Biases).

---

## 📄 API Endpoints

### `POST /api/predict`
Predicts house price using the ensemble model.

**Payload**:
```json
{
  "area": 80,
  "floors": 4,
  "bedrooms": 4,
  "bathrooms": 3,
  "legal_status": "Have certificate",
  "furniture_state": "Full",
  "address": "123 Đường Nguyễn Huệ, Phường Bến Nghé, Quận 1, Hồ Chí Minh"
}
```
*Note: Refer to `app.py` for all accepted parameters.*
