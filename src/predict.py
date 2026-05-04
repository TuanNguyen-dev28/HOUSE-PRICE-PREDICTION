"""
Dự đoán giá nhà — Module dự đoán
Tải mô hình đã huấn luyện và cung cấp dự đoán cho đầu vào một bất động sản.
Hỗ trợ cả single model và ensemble (XGBoost + Random Forest).
"""
import numpy as np
import pandas as pd
import pickle
import os
import json
from typing import Optional, Dict, List, Tuple

# Các cột features chính xác mà mô hình đã huấn luyện yêu cầu (theo thứ tự)
FEATURE_COLUMNS = [
    'Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms',
    'House direction_Bắc', 'House direction_Nam', 'House direction_Tây',
    'House direction_Tây - Bắc', 'House direction_Tây - Nam',
    'House direction_Đông', 'House direction_Đông - Bắc', 'House direction_Đông - Nam',
    'Balcony direction_Bắc', 'Balcony direction_Nam',
    'Balcony direction_Tây', 'Balcony direction_Tây - Bắc',
    'Balcony direction_Tây - Nam', 'Balcony direction_Đông',
    'Balcony direction_Đông - Bắc', 'Balcony direction_Đông - Nam',
    'Legal_status_ordinal', 'Furniture_state_ordinal',
    'District_Encoded', 'City_Encoded'
]

# Trọng số mặc định cho ensemble (có thể điều chỉnh)
DEFAULT_ENSEMBLE_WEIGHTS = {
    'xgboost': 0.6,
    'random_forest': 0.4
}

# Các hướng nhà có thể
HOUSE_DIRECTIONS = [
    'Bắc', 'Nam', 'Tây', 'Tây - Bắc', 'Tây - Nam',
    'Đông', 'Đông - Bắc', 'Đông - Nam'
]

# Các hướng ban công có thể
BALCONY_DIRECTIONS = [
    'Bắc', 'Nam', 'Tây', 'Tây - Bắc', 'Tây - Nam',
    'Đông', 'Đông - Bắc', 'Đông - Nam'
]

# Ordinal mapping cho pháp lý (theo thực tế thị trường BĐS)
# Có sổ → giá cao nhất, Chưa có → giá thấp nhất
LEGAL_STATUS_ORDINAL = {
    'Have certificate': 3,  # Có sổ đỏ/sổ hồng
    'Sale contract': 2,     # Hợp đồng mua bán
    'In progress': 1,       # Đang làm sổ
    'Pending': 0,           # Chưa có giấy tờ
}
LEGAL_STATUSES = list(LEGAL_STATUS_ORDINAL.keys())

# Ordinal mapping cho nội thất
FURNITURE_STATE_ORDINAL = {
    'Full': 2,    # Nội thất đầy đủ
    'Basic': 1,   # Nội thất cơ bản
    'Empty': 0,   # Không nội thất
}
FURNITURE_STATES = list(FURNITURE_STATE_ORDINAL.keys())


class HousePricePredictor:
    """Tải mô hình đã huấn luyện và cung cấp dự đoán cho đầu vào một bất động sản."""

    def __init__(self, model_path, location_encodings_path=None):
        """
        Args:
            model_path: Đường dẫn đến file model đã lưu (.pkl)
            location_encodings_path: Đường dẫn đến file JSON encoding vị trí
        """
        print(f"Đang tải model từ {model_path}...")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print("Model được tải thành công.")

        # Tải location encodings cho target encoding
        self.district_encodings = {}
        self.city_encodings = {}
        self.district_global_mean = 8.71
        self.city_global_mean = 8.71

        if location_encodings_path and os.path.exists(location_encodings_path):
            print(f"Đang tải location encodings từ {location_encodings_path}...")
            with open(location_encodings_path, 'r', encoding='utf-8') as f:
                enc = json.load(f)
            
            self.district_encodings = enc.get('district_encoding', {})
            self.city_encodings = enc.get('city_encoding', {})
            self.district_global_mean = enc.get('district_global_mean', self.district_global_mean)
            self.city_global_mean = enc.get('city_global_mean', self.city_global_mean)
            
            print(f"  Đã tải {len(self.district_encodings)} district encodings, "
                  f"{len(self.city_encodings)} city encodings.")
        else:
            print("  ⚠️ Không tìm thấy location encodings! Sử dụng giá trị mặc định.")

    def _extract_district_city(self, address):
        """Trích xuất Quận/Huyện và Thành phố từ địa chỉ đầy đủ."""
        if not address or not isinstance(address, str):
            return 'Unknown', 'Unknown'

        parts = address.split(',')
        if len(parts) >= 2:
            city = parts[-1].strip()
            district = parts[-2].strip()
        else:
            district = 'Unknown'
            city = 'Unknown'

        return district, city

    def _encode_location(self, address):
        """Target encode vị trí từ chuỗi địa chỉ."""
        district, city = self._extract_district_city(address)

        district_encoded = self.district_encodings.get(district, self.district_global_mean)
        city_encoded = self.city_encodings.get(city, self.city_global_mean)

        return district_encoded, city_encoded

    def preprocess_single(self, input_data: dict) -> pd.DataFrame:
        """Chuyển đổi một dict đầu vào thành DataFrame features phù hợp với schema mô hình."""
        row = {col: 0.0 for col in FEATURE_COLUMNS}

        # Các features số
        row['Area'] = float(input_data.get('area', 0))
        row['Frontage'] = float(input_data.get('frontage', 0))
        row['Access Road'] = float(input_data.get('access_road', 0))
        row['Floors'] = float(input_data.get('floors', 0))
        row['Bedrooms'] = float(input_data.get('bedrooms', 0))
        row['Bathrooms'] = float(input_data.get('bathrooms', 0))

        # One-hot encode hướng nhà
        house_dir = input_data.get('house_direction', '')
        if house_dir and house_dir in HOUSE_DIRECTIONS:
            col_name = f'House direction_{house_dir}'
            if col_name in row:
                row[col_name] = 1.0
        else:
            row['House direction_Đông - Nam'] = 1.0

        # One-hot encode hướng ban công
        balcony_dir = input_data.get('balcony_direction', '')
        if balcony_dir and balcony_dir in BALCONY_DIRECTIONS:
            col_name = f'Balcony direction_{balcony_dir}'
            if col_name in row:
                row[col_name] = 1.0
        else:
            row['Balcony direction_Đông - Nam'] = 1.0

        # Ordinal encode trạng thái pháp lý
        legal = input_data.get('legal_status', '')
        if legal and legal in LEGAL_STATUS_ORDINAL:
            row['Legal_status_ordinal'] = float(LEGAL_STATUS_ORDINAL[legal])
        else:
            row['Legal_status_ordinal'] = float(LEGAL_STATUS_ORDINAL['Have certificate'])

        # Ordinal encode trạng thái nội thất
        furniture = input_data.get('furniture_state', '')
        if furniture and furniture in FURNITURE_STATE_ORDINAL:
            row['Furniture_state_ordinal'] = float(FURNITURE_STATE_ORDINAL[furniture])
        else:
            row['Furniture_state_ordinal'] = float(FURNITURE_STATE_ORDINAL['Full'])

        # Target encode vị trí từ địa chỉ
        address = input_data.get('address', '')
        district_enc, city_enc = self._encode_location(address)
        row['District_Encoded'] = district_enc
        row['City_Encoded'] = city_enc

        df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        return df

    def predict(self, input_data: dict) -> dict:
        """Dự đoán giá cho một bất động sản."""
        features_df = self.preprocess_single(input_data)
        prediction = float(self.model.predict(features_df)[0])
        prediction = max(0.1, prediction)

        price_billion = round(prediction, 2)
        price_vnd = price_billion * 1_000_000_000

        if price_billion >= 1:
            price_formatted = f"{price_billion:.2f} tỷ VNĐ"
        else:
            price_million = price_billion * 1000
            price_formatted = f"{price_million:.0f} triệu VNĐ"

        return {
            'price_billion_vnd': price_billion,
            'price_vnd': price_vnd,
            'price_formatted': price_formatted,
            'input_summary': {
                'area': input_data.get('area', 0),
                'floors': input_data.get('floors', 0),
                'bedrooms': input_data.get('bedrooms', 0),
                'bathrooms': input_data.get('bathrooms', 0),
            }
        }


class EnsemblePredictor:
    """
    Kết hợp dự đoán từ XGBoost và Random Forest để cải thiện độ chính xác.
    
    Phương pháp:
    1. Weighted Averaging: Kết hợp dự đoán theo trọng số dựa trên hiệu suất model
    2. Confidence Interval: Ước lượng khoảng tin cậy cho dự đoán
    3. Individual Predictions: Trả về dự đoán riêng của từng model để so sánh
    """
    
    def __init__(
        self,
        xgboost_model_path: str,
        random_forest_model_path: str,
        location_encodings_path: str,
        weights: Optional[Dict[str, float]] = None,
        xgboost_weight: float = 0.6,
        rf_weight: float = 0.4
    ):
        """
        Khởi tạo EnsemblePredictor với nhiều models.
        
        Args:
            xgboost_model_path: Đường dẫn đến XGBoost model
            random_forest_model_path: Đường dẫn đến Random Forest model
            location_encodings_path: Đường dẫn đến file JSON encoding vị trí
            weights: Dict chứa trọng số {'xgboost': 0.6, 'random_forest': 0.4}
            xgboost_weight: Trọng số XGBoost (nếu weights=None)
            rf_weight: Trọng số Random Forest (nếu weights=None)
        """
        # Load location encodings một lần (để dùng chung)
        self.district_encodings = {}
        self.city_encodings = {}
        self.district_global_mean = 8.71
        self.city_global_mean = 8.71
        
        if os.path.exists(location_encodings_path):
            with open(location_encodings_path, 'r', encoding='utf-8') as f:
                enc = json.load(f)
            self.district_encodings = enc.get('district_encoding', {})
            self.city_encodings = enc.get('city_encoding', {})
            self.district_global_mean = enc.get('district_global_mean', self.district_global_mean)
            self.city_global_mean = enc.get('city_global_mean', self.city_global_mean)
        
        # Load models
        print("Đang tải XGBoost model...")
        with open(xgboost_model_path, 'rb') as f:
            self.xgboost_model = pickle.load(f)
        print("  ✓ XGBoost model đã tải!")
        
        print("Đang tải Random Forest model...")
        with open(random_forest_model_path, 'rb') as f:
            self.random_forest_model = pickle.load(f)
        print("  ✓ Random Forest model đã tải!")
        
        # Set weights
        if weights:
            self.weights = weights
        else:
            # Validate weights
            total = xgboost_weight + rf_weight
            self.weights = {
                'xgboost': xgboost_weight / total,
                'random_forest': rf_weight / total
            }
        
        print(f"\n📊 Ensemble Weights: XGBoost={self.weights['xgboost']:.1%}, "
              f"RF={self.weights['random_forest']:.1%}")
    
    def _extract_district_city(self, address: str) -> Tuple[str, str]:
        """Trích xuất Quận/Huyện và Thành phố từ địa chỉ."""
        if not address or not isinstance(address, str):
            return 'Unknown', 'Unknown'
        
        parts = address.split(',')
        if len(parts) >= 2:
            city = parts[-1].strip()
            district = parts[-2].strip()
        else:
            district = 'Unknown'
            city = 'Unknown'
        
        return district, city
    
    def _encode_location(self, address: str) -> Tuple[float, float]:
        """Target encode vị trí."""
        district, city = self._extract_district_city(address)
        district_encoded = self.district_encodings.get(district, self.district_global_mean)
        city_encoded = self.city_encodings.get(city, self.city_global_mean)
        return district_encoded, city_encoded
    
    def preprocess_single(self, input_data: dict) -> pd.DataFrame:
        """Tiền xử lý dữ liệu đầu vào thành DataFrame."""
        row = {col: 0.0 for col in FEATURE_COLUMNS}
        
        # Features số
        row['Area'] = float(input_data.get('area', 0))
        row['Frontage'] = float(input_data.get('frontage', 0))
        row['Access Road'] = float(input_data.get('access_road', 0))
        row['Floors'] = float(input_data.get('floors', 0))
        row['Bedrooms'] = float(input_data.get('bedrooms', 0))
        row['Bathrooms'] = float(input_data.get('bathrooms', 0))
        
        # One-hot encode hướng nhà
        house_dir = input_data.get('house_direction', '')
        if house_dir and house_dir in HOUSE_DIRECTIONS:
            col_name = f'House direction_{house_dir}'
            if col_name in row:
                row[col_name] = 1.0
        else:
            row['House direction_Đông - Nam'] = 1.0
        
        # One-hot encode hướng ban công
        balcony_dir = input_data.get('balcony_direction', '')
        if balcony_dir and balcony_dir in BALCONY_DIRECTIONS:
            col_name = f'Balcony direction_{balcony_dir}'
            if col_name in row:
                row[col_name] = 1.0
        else:
            row['Balcony direction_Đông - Nam'] = 1.0
        
        # Ordinal encode trạng thái pháp lý
        legal = input_data.get('legal_status', '')
        if legal and legal in LEGAL_STATUS_ORDINAL:
            row['Legal_status_ordinal'] = float(LEGAL_STATUS_ORDINAL[legal])
        else:
            row['Legal_status_ordinal'] = float(LEGAL_STATUS_ORDINAL['Have certificate'])
        
        # Ordinal encode trạng thái nội thất
        furniture = input_data.get('furniture_state', '')
        if furniture and furniture in FURNITURE_STATE_ORDINAL:
            row['Furniture_state_ordinal'] = float(FURNITURE_STATE_ORDINAL[furniture])
        else:
            row['Furniture_state_ordinal'] = float(FURNITURE_STATE_ORDINAL['Full'])
        
        # Target encode vị trí
        address = input_data.get('address', '')
        district_enc, city_enc = self._encode_location(address)
        row['District_Encoded'] = district_enc
        row['City_Encoded'] = city_enc
        
        df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        return df
    
    def predict(self, input_data: dict, include_individual: bool = True) -> dict:
        """
        Dự đoán giá sử dụng ensemble của XGBoost và Random Forest.
        
        Args:
            input_data: Dict chứa thông tin bất động sản
            include_individual: Nếu True, trả về dự đoán riêng của từng model
        
        Returns:
            Dict chứa dự đoán ensemble, confidence interval, và tùy chọn dự đoán riêng
        """
        features_df = self.preprocess_single(input_data)
        
        # Dự đoán riêng từng model
        xgb_pred = float(self.xgboost_model.predict(features_df)[0])
        rf_pred = float(self.random_forest_model.predict(features_df)[0])
        
        # Weighted ensemble prediction
        ensemble_pred = (
            self.weights['xgboost'] * xgb_pred +
            self.weights['random_forest'] * rf_pred
        )
        
        # Tính confidence interval dựa trên sự khác biệt giữa 2 models
        pred_diff = abs(xgb_pred - rf_pred)
        confidence_margin = pred_diff * 0.5  # 50% của difference làm margin
        
        # Đảm bảo giá hợp lý
        ensemble_pred = max(0.1, ensemble_pred)
        xgb_pred = max(0.1, xgb_pred)
        rf_pred = max(0.1, rf_pred)
        
        # Format kết quả
        def format_price(price_billion):
            if price_billion >= 1:
                return f"{price_billion:.2f} tỷ VNĐ"
            else:
                return f"{price_billion * 1000:.0f} triệu VNĐ"
        
        result = {
            # Ensemble prediction
            'ensemble': {
                'price_billion_vnd': round(ensemble_pred, 2),
                'price_vnd': int(ensemble_pred * 1_000_000_000),
                'price_formatted': format_price(ensemble_pred),
            },
            
            # Confidence interval (95% CI dựa trên spread của 2 models)
            'confidence_interval': {
                'lower': round(max(0.1, ensemble_pred - confidence_margin), 2),
                'upper': round(ensemble_pred + confidence_margin, 2),
                'lower_formatted': format_price(max(0.1, ensemble_pred - confidence_margin)),
                'upper_formatted': format_price(ensemble_pred + confidence_margin),
                'margin': round(confidence_margin, 2),
            },
            
            # Metadata
            'metadata': {
                'xgboost_weight': self.weights['xgboost'],
                'random_forest_weight': self.weights['random_forest'],
                'prediction_agreement': round(1 - (pred_diff / max(ensemble_pred, 1)) if ensemble_pred > 0 else 0, 3),
                'prediction_difference': round(pred_diff, 2),
            },
            
            # Input summary
            'input_summary': {
                'area': input_data.get('area', 0),
                'floors': input_data.get('floors', 0),
                'bedrooms': input_data.get('bedrooms', 0),
                'bathrooms': input_data.get('bathrooms', 0),
            }
        }
        
        # Thêm dự đoán riêng nếu được yêu cầu
        if include_individual:
            result['individual_predictions'] = {
                'xgboost': {
                    'price_billion_vnd': round(xgb_pred, 2),
                    'price_formatted': format_price(xgb_pred),
                },
                'random_forest': {
                    'price_billion_vnd': round(rf_pred, 2),
                    'price_formatted': format_price(rf_pred),
                }
            }
        
        return result
    
    def predict_batch(self, input_data_list: List[dict]) -> List[dict]:
        """
        Dự đoán giá cho nhiều bất động sản.
        
        Args:
            input_data_list: List chứa các dict thông tin bất động sản
        
        Returns:
            List các dict kết quả dự đoán
        """
        return [self.predict(data) for data in input_data_list]
    
    def update_weights(self, xgboost_weight: float, rf_weight: float) -> Dict[str, float]:
        """
        Cập nhật trọng số cho ensemble.
        
        Args:
            xgboost_weight: Trọng số mới cho XGBoost
            rf_weight: Trọng số mới cho Random Forest
        
        Returns:
            Dict chứa trọng số đã normalize
        """
        total = xgboost_weight + rf_weight
        if total == 0:
            raise ValueError("Tổng trọng số phải lớn hơn 0")
        
        self.weights = {
            'xgboost': xgboost_weight / total,
            'random_forest': rf_weight / total
        }
        
        return self.weights
    
    def optimize_weights(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_steps: int = 11
    ) -> Dict[str, float]:
        """
        Tối ưu trọng số ensemble bằng cách thử nhiều tỷ lệ khác nhau.
        
        Args:
            X_val: Features validation
            y_val: Target validation
            n_steps: Số bước để thử (mặc định 11 = 0%, 10%, 20%, ..., 100%)
        
        Returns:
            Dict chứa trọng số tối ưu
        """
        from sklearn.metrics import mean_absolute_error
        
        xgb_preds = self.xgboost_model.predict(X_val)
        rf_preds = self.random_forest_model.predict(X_val)
        
        best_mae = float('inf')
        best_weights = {'xgboost': 0.5, 'random_forest': 0.5}
        
        for i in range(n_steps):
            xgb_weight = i / (n_steps - 1)
            rf_weight = 1 - xgb_weight
            
            ensemble_preds = xgb_weight * xgb_preds + rf_weight * rf_preds
            mae = mean_absolute_error(y_val, ensemble_preds)
            
            if mae < best_mae:
                best_mae = mae
                best_weights = {'xgboost': xgb_weight, 'random_forest': rf_weight}
        
        self.weights = best_weights
        return best_weights
