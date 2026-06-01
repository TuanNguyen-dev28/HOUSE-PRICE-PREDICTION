import os
import sys
import json
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predict import EnsemblePredictor

def main():
    xgb_path = os.path.join(BASE_DIR, 'models', 'xgboost_model.pkl')
    rf_path = os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl')
    encodings_path = os.path.join(BASE_DIR, 'models', 'location_encodings.json')
    weights_path = os.path.join(BASE_DIR, 'models', 'ensemble_weights.json')
    
    xgb_weight, rf_weight = 0.6, 0.4
    if os.path.exists(weights_path):
        with open(weights_path, 'r') as f:
            w = json.load(f)
        xgb_weight = w.get('xgboost_weight', 0.6)
        rf_weight = w.get('random_forest_weight', 0.4)
        
    predictor = EnsemblePredictor(
        xgboost_model_path=xgb_path,
        random_forest_model_path=rf_path,
        location_encodings_path=encodings_path,
        xgboost_weight=xgb_weight,
        rf_weight=rf_weight
    )
    
    # Baseline house in Tan Binh
    base_house = {
        "area": 80.0,
        "floors": 3.0,
        "frontage": 4.5,
        "access_road": 5.0,
        "bedrooms": 3.0,
        "bathrooms": 3.0,
        "house_direction": "Đông",
        "balcony_direction": "Đông",
        "furniture_state": "Basic",
        "address": "250 Cộng Hòa, Phường 13, Quận Tân Bình, TP.HCM",
        "property_type": "Nhà"
    }
    
    legal_statuses = [
        ("Have certificate", "Sổ đỏ / Sổ hồng"),
        ("Sale contract", "Hợp đồng mua bán"),
        ("In progress", "Đang chờ cấp sổ"),
        ("Pending", "Chưa có sổ / Giấy viết tay")
    ]
    
    print("=" * 70)
    print("   AI HOUSE PRICE REGRESSION - LEGAL STATUS IMPACT TEST")
    print("=" * 70)
    print(f"House info: {base_house['area']}m², {base_house['floors']} floors, Tân Bình district")
    
    for status_code, status_name in legal_statuses:
        house = base_house.copy()
        house["legal_status"] = status_code
        
        try:
            res = predictor.predict(house)
            print(f"\n➔ Pháp lý: {status_name} ({status_code})")
            print(f"  • XGBoost Price:   {res['individual_predictions']['xgboost']['price_formatted']}")
            print(f"  • RF Price:        {res['individual_predictions']['random_forest']['price_formatted']}")
            print(f"  • Ensemble Price:  {res['ensemble']['price_formatted']}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

if __name__ == '__main__':
    main()
