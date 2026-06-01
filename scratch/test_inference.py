import os
import sys
import json
import io

# Fix output encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predict import HousePricePredictor, EnsemblePredictor

def main():
    xgb_path = os.path.join(BASE_DIR, 'models', 'xgboost_model.pkl')
    rf_path = os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl')
    encodings_path = os.path.join(BASE_DIR, 'models', 'location_encodings.json')
    weights_path = os.path.join(BASE_DIR, 'models', 'ensemble_weights.json')
    
    # Load ensemble weights
    xgb_weight, rf_weight = 0.6, 0.4
    if os.path.exists(weights_path):
        with open(weights_path, 'r') as f:
            w = json.load(f)
        xgb_weight = w.get('xgboost_weight', 0.6)
        rf_weight = w.get('random_forest_weight', 0.4)
    
    print("=" * 70)
    print("        AI HOUSE PRICE REGRESSION - SAMPLE PREDICTION TEST")
    print("=" * 70)
    
    # Initialize EnsemblePredictor
    predictor = EnsemblePredictor(
        xgboost_model_path=xgb_path,
        random_forest_model_path=rf_path,
        location_encodings_path=encodings_path,
        xgboost_weight=xgb_weight,
        rf_weight=rf_weight
    )
    
    # Test cases
    test_cases = [
        {
            "name": "Case 1: Biệt thự/Nhà phố cao cấp tại Quận 1 (CBD)",
            "data": {
                "area": 120.0,
                "floors": 4.0,
                "frontage": 6.0,
                "access_road": 8.0,
                "bedrooms": 4.0,
                "bathrooms": 4.0,
                "house_direction": "Đông - Nam",
                "balcony_direction": "Đông - Nam",
                "legal_status": "Have certificate",
                "furniture_state": "Full",
                "address": "15 Nguyễn Huệ, Phường Bến Nghé, Quận 1, TP.HCM",
                "property_type": "Nhà"
            }
        },
        {
            "name": "Case 2: Nhà phố trung cấp tại Quận Tân Bình",
            "data": {
                "area": 80.0,
                "floors": 3.0,
                "frontage": 4.5,
                "access_road": 5.0,
                "bedrooms": 3.0,
                "bathrooms": 3.0,
                "house_direction": "Đông",
                "balcony_direction": "Đông",
                "legal_status": "Have certificate",
                "furniture_state": "Basic",
                "address": "250 Cộng Hòa, Phường 13, Quận Tân Bình, TP.HCM",
                "property_type": "Nhà"
            }
        },
        {
            "name": "Case 3: Căn hộ chung cư bình dân tại Huyện Bình Chánh",
            "data": {
                "area": 60.0,
                "floors": 1.0,
                "frontage": 0.0,
                "access_road": 10.0,
                "bedrooms": 2.0,
                "bathrooms": 1.0,
                "house_direction": "Nam",
                "balcony_direction": "Nam",
                "legal_status": "Sale contract",
                "furniture_state": "Empty",
                "address": "Đường Nguyễn Văn Linh, Huyện Bình Chánh, TP.HCM",
                "property_type": "Chung cư"
            }
        }
    ]
    
    for case in test_cases:
        print("\n" + "-" * 70)
        print(f"📌 {case['name']}")
        print("-" * 70)
        for k, v in case['data'].items():
            if k != 'address':
                print(f"  • {k}: {v}")
        print(f"  • Địa chỉ: {case['data']['address']}")
        
        try:
            res = predictor.predict(case['data'])
            print("\n🔮 KẾT QUẢ DỰ ĐOÁN:")
            print(f"  • XGBoost Prediction:      {res['individual_predictions']['xgboost']['price_formatted']}")
            print(f"  • Random Forest Prediction: {res['individual_predictions']['random_forest']['price_formatted']}")
            print(f"  • Ensemble Price (State):   {res['ensemble']['price_formatted']}")
            print(f"  • Khoảng tin cậy 95%:      {res['confidence_interval']['lower']} - {res['confidence_interval']['upper']} tỷ VNĐ")
            print(f"  • Vùng vị trí phân tích:   {res['location_analysis']['district']} | {res['location_analysis']['ward']}")
        except Exception as e:
            print(f"  ❌ Lỗi dự đoán: {e}")

if __name__ == '__main__':
    main()
