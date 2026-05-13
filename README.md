# House Price Prediction AI - TP.HCM

Hệ thống AI dự đoán giá bất động sản tại TP.HCM, sử dụng mô hình Ensemble (XGBoost + Random Forest) với các kỹ thuật tiên tiến như SHAP explainability, Optuna hyperparameter tuning, và Geo Intelligence features.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red?logo=xgboost&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-brightgreen)

---

## Tính năng chính

### 🤖 Mô hình Ensemble thông minh
- **XGBoost** + **Random Forest** kết hợp theo trọng số tối ưu (10% XGB + 90% RF)
- **Monotone Constraints**: Đảm bảo diện tích tăng → giá tăng (không bao giờ giảm)
- **Log-Target Transformation**: Xử lý tốt cả nhà giá thấp và nhà cao cấp

### 🗺️ Geo Intelligence
- KMeans location clusters
- Distance to CBD, Metro, Mall, Hospital, School
- Premium location detection (Lê Lợi, Nguyễn Huệ, Đồng Khởi...)
- Urban Development Index

### 🔒 Anti-Leakage Architecture
- **Smoothed Target Encoding**: Giảm overfitting cho Street/Ward encoding
- **K-Fold OOF Encoding**: Ngăn data leakage trong training
- **Geographic Hierarchy**: Giảm trọng số Street, tăng trọng số Land_Value

### 📊 Model Explainability
- **SHAP Analysis**: Giải thích từng dự đoán
- **Feature Importance**: Trọng số thực của từng feature
- **Confidence Interval**: Độ tin cậy của dự đoán

### ⚡ Auto Optimization
- **Optuna**: Tự động tìm hyperparameter tốt nhất
- **Cross-Validation**: Đánh giá model robustness

---

## Cấu trúc dự án

```
House-price-regression/
├── app.py                    # Flask web server
├── src/
│   ├── train.py             # Training pipeline
│   ├── predict.py           # Prediction module
│   ├── preprocess.py        # Data preprocessing
│   ├── geo_intelligence.py  # Geo features
│   ├── location_features.py # Location encoding
│   ├── smooth_encoding.py   # Anti-leakage encoding
│   ├── optuna_tuning.py     # Hyperparameter tuning
│   ├── shap_analysis.py     # SHAP explainability
│   └── utils.py, logger.py  # Utilities
├── models/                  # Trained models
├── data/
│   ├── raw/                 # Raw data
│   └── processed/           # Preprocessed data
├── static/                  # Web frontend
└── tests/                   # Unit tests
```

---

## Dataset

| Metric | Value |
|--------|-------|
| Tổng số bản ghi | ~40,000 |
| Khu vực | TP.HCM |
| Features | 25+ |

### Features chính

| Feature | Mô tả | Tầm quan trọng |
|---------|--------|----------------|
| Land_Value | Diện tích × Giá đất/m² | ⭐⭐⭐ Core |
| Land_Price_Per_M2 | Giá đất theo quận | ⭐⭐⭐ Core |
| Area | Diện tích (m²) | ⭐⭐⭐ Core |
| Floors | Số tầng | ⭐⭐ High |
| Frontage | Mặt tiền (m) | ⭐⭐ High |
| Legal_Status | Pháp lý | ⭐⭐ High |
| Furniture_State | Tình trạng nội thất | ⭐⭐ High |
| Distance_From_CBD | Khoảng cách đến trung tâm | ⭐ Medium |
| Premium_Location_Score | Điểm vị trí cao cấp | ⭐ Medium |
| Urban_Development_Index | Chỉ số đô thị hóa | ⭐ Medium |

---

## Cài đặt & Chạy

```bash
# 1. Tạo virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Training (tùy chọn - đã có pretrained models)
python -m src.train

# 4. Chạy web app
python app.py
```

Truy cập **http://localhost:5000** để sử dụng.

---

## API Endpoints

### `POST /api/predict`
Dự đoán giá nhà

```json
{
  "area": 80,
  "frontage": 5,
  "access_road": 6,
  "floors": 3,
  "bedrooms": 3,
  "bathrooms": 2,
  "house_direction": "Đông",
  "balcony_direction": "Nam",
  "legal_status": "Have certificate",
  "furniture_state": "Full",
  "address": "100 Lê Lợi, Bến Nghé, Quận 1, TP.HCM"
}
```

### `POST /api/compare`
So sánh XGBoost vs Random Forest vs Ensemble

### `GET /api/stats`
Thống kê dataset

### `GET /api/locations`
Danh sách quận/huyện

---

## Ví dụ prediction

```
Input: 80m², 3 tầng, 3 phòng ngủ, Lê Lợi Q1
Output: ~12-15 tỷ VNĐ
```

---

## Tech Stack

- **Python 3.10+**
- **XGBoost 2.0** - Gradient Boosting
- **scikit-learn** - ML utilities
- **Optuna** - Hyperparameter optimization
- **SHAP** - Model explainability
- **Flask 3.0** - Web framework

---

*Developed with ❤️ for TP.HCM real estate market*
