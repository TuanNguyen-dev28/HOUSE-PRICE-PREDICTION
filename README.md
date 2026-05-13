# House Price Prediction AI (Ensemble Model)

Hệ thống AI dự đoán giá bất động sản tại Việt Nam, sử dụng mô hình kết hợp (Ensemble) giữa **XGBoost** và **Random Forest** được tối ưu hóa, huấn luyện trên tập dữ liệu hơn **44,000** bản ghi.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red?logo=xgboost&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter--Tuning-blueviolet)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-brightgreen)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask&logoColor=white)

---

## ✨ Điểm nhấn kỹ thuật (Technical Highlights)

Dự án này vượt xa các mô hình hồi quy thông thường bằng cách tích hợp các kỹ thuật tiên tiến nhất để đảm bảo tính **chính xác** và **logic kinh doanh**:

- **Weighted Ensemble Architecture**: Kết hợp dự đoán từ **XGBoost** và **Random Forest** theo trọng số tối ưu (ví dụ: 60% XGB + 40% RF). Việc kết hợp này giúp giảm variance và tăng tính ổn định cho các phân khúc giá khác nhau.
- **Strict Monotone Constraints**: Áp dụng ràng buộc đơn điệu cho **cả hai mô hình**. 
    - **XGBoost**: Sử dụng `monotone_constraints` nguyên bản.
    - **Random Forest**: Chuyển sang sử dụng `HistGradientBoostingRegressor` để hỗ trợ ràng buộc đơn điệu.
    - *Kết quả*: Đảm bảo các quy luật bất biến (Ví dụ: Diện tích tăng thì giá không được giảm).
- **Log-Target Transformation**: Áp dụng phép biến đổi $y' = \log(1+y)$ cho biến mục tiêu (Giá) để xử lý dữ liệu bị lệch (skewed data) và cải thiện hiệu suất dự đoán cho các bất động sản giá trị cực cao hoặc cực thấp.
- **SHAP Explainability**: Tích hợp phân tích **SHAP (SHapley Additive exPlanations)** để giải thích tại sao mô hình đưa ra mức giá đó. Người dùng có thể hiểu được trọng số của từng yếu tố (Diện tích, Vị trí, Pháp lý) tác động thế nào đến kết quả cuối cùng.
- **Optuna Hyperparameter Optimization**: Sử dụng framework **Optuna** để tự động tìm kiếm bộ tham số tốt nhất cho cả XGBoost và Random Forest thông qua quá trình thử nghiệm hàng trăm phiên bản khác nhau.
- **Target Encoding with OOF (Out-Of-Fold)**: Mã hóa các đặc trưng vị trí (Quận, Thành phố) có độ đa dạng cao mà không gây hiện tượng rò rỉ dữ liệu (data leakage).

---

## 📂 Cấu trúc dự án (Project Structure)

```text
E:\AI\House-price-regression\
├── data\                       # Thư mục dữ liệu
│   ├── processed\              # Dữ liệu đã tiền xử lý, sẵn sàng training
│   └── raw\                    # Dữ liệu thô (house_data.csv)
│
├── models\                     # Model & Trọng số đã lưu
│   ├── xgboost_model.pkl       # Mô hình XGBoost (với constraints)
│   ├── random_forest_model.pkl  # Mô hình RF (HistGradientBoosting)
│   ├── ensemble_weights.json   # Trọng số tối ưu cho Ensemble
│   ├── optuna_best_params.json # Tham số tốt nhất tìm được bởi Optuna
│   ├── location_encodings.json # Encodings cho Quận/Huyện/Thành phố
│   └── shap_plots\             # Các biểu đồ giải thích mô hình SHAP
│
├── src\                        # ML Pipeline cốt lõi
│   ├── train.py                # Pipeline huấn luyện, CV & Ensemble logic
│   ├── predict.py              # Module dự đoán & Confidence Interval
│   ├── preprocess.py           # Tiền xử lý (OOF Target & Ordinal Encoding)
│   ├── shap_analysis.py        # Phân tích độ quan trọng bằng SHAP
│   ├── optuna_tuning.py        # Tự động tối ưu tham số (Hyperparameter Tuning)
│   ├── logger.py               # Hệ thống ghi log tập trung
│   └── utils.py                # Các hàm hỗ trợ chung
│
├── static\                     # Web Frontend (UI/UX cao cấp)
│   ├── app.js                  # Logic tương tác & gọi API
│   ├── index.html              # Giao diện chính
│   └── style.css               # Styling (Modern Glassmorphism)
│
├── tests\                      # Kiểm thử tự động (Pytest)
│   ├── test_predict.py
│   └── test_preprocess.py
│
└── app.py                      # Flask Application (Production Server)
```

---

## 📊 Dataset & Features

**Tổng số bản ghi**: ~44,400 (Sau khi làm sạch và tăng cường dữ liệu)

| Đặc trưng | Mô tả | Kiểu / Encoding | Ràng buộc |
|:--- |:--- |:--- |:--- |
| **Area** | Diện tích (m²) | Numeric | **Đơn điệu tăng** (⬆️) |
| **Floors** | Số tầng | Numeric | **Đơn điệu tăng** (⬆️) |
| **District/City** | Vị trí địa lý | Target Encoding (OOF) | **Đơn điệu tăng** (⬆️) |
| **Legal Status** | Trình trạng pháp lý | Ordinal Encoding | **Đơn điệu tăng** (⬆️) |
| **Furniture** | Tình trạng nội thất | Ordinal Encoding | **Đơn điệu tăng** (⬆️) |
| **Beds/Baths** | Số phòng ngủ/vệ sinh | Numeric | Tự do |
| **Price** | Giá (Tỷ VNĐ) | **Biến mục tiêu** | (Log Transformed) |

---

## 🧠 Hiệu suất & Độ tin cậy (Performance & Reliability)

Mô hình không chỉ tối ưu về mặt sai số mà còn được thiết kế để cung cấp **Khoảng tin cậy (Confidence Interval)**:

- **Ensemble Result**: Kết hợp sức mạnh của Gradient Boosting (XGBoost) và Histogram-based Trees (Random Forest).
- **Agreement Metric**: Hệ thống đo lường độ đồng thuận giữa 2 mô hình. Nếu 2 mô hình đưa ra kết quả gần nhau, độ tin cậy của dự đoán sẽ cao hơn.
- **Outlier Handling**: Loại bỏ các bất động sản có giá trị phi thực tế (>250 tỷ) hoặc diện tích quá lớn (>500m²) để tránh nhiễu mô hình.

---

## ✅ Lộ trình phát triển (Project Checklist)

### Phase 1: Data Infrastructure
- [x] Crawl/Thu thập dữ liệu BĐS thực tế.
- [x] Xử lý giá trị thiếu (Imputation) & Outliers.
- [x] Data Augmentation (Tăng cường dữ liệu cho các phân khúc hiếm).

### Phase 2: Advanced Feature Engineering
- [x] Target Encoding (OOF) cho Quận/Huyện/Thành phố.
- [x] Phân loại Ordinal cho Pháp lý & Nội thất.
- [x] **Log Transformation** cho biến mục tiêu.

### Phase 3: Modeling & AI Science
- [x] Triển khai **XGBoost Monotone Constraints**.
- [x] Triển khai **Random Forest Monotone** (via HistGBR).
- [x] Tối ưu hóa tham số bằng **Optuna**.
- [x] Tích hợp giải thích mô hình bằng **SHAP**.
- [x] Xây dựng Weighted Ensemble tự động tối ưu trọng số.

### Phase 4: MLOps & Deployment
- [x] Xây dựng REST API bằng Flask.
- [x] Giao diện Web Interactive cao cấp.
- [x] Ước lượng khoảng tin cậy (Confidence Interval) cho dự đoán.
- [ ] Dockerize ứng dụng.
- [ ] CI/CD Pipeline với GitHub Actions.

---

## 🚀 Khởi chạy nhanh

1. **Cài đặt môi trường**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Hoặc .venv\Scripts\activate trên Windows
   pip install -r requirements.txt
   ```

2. **Huấn luyện mô hình**:
   ```bash
   python -m src.train
   ```

3. **Chạy Web App**:
   ```bash
   python app.py
   ```
   Truy cập **http://localhost:5000** để sử dụng.

---

## 📄 API Endpoints

### `POST /api/predict`
Dự đoán giá nhà sử dụng Ensemble model. Trả về giá dự đoán, khoảng tin cậy và độ đồng thuận giữa các mô hình.

**Dữ liệu mẫu**:
```json
{
  "area": 75,
  "floors": 3,
  "bedrooms": 3,
  "legal_status": "Have certificate",
  "address": "Phường Bến Nghé, Quận 1, Hồ Chí Minh"
}
```

---
*Phát triển bởi [TuanNguyen-dev28](https://github.com/TuanNguyen-dev28)*
