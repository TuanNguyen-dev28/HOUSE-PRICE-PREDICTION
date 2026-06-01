"""
Dự đoán giá nhà — Ứng dụng Web Flask
Phục vụ giao diện web đẹp và cung cấp REST API cho dự đoán giá.
Sử dụng Ensemble (XGBoost + Random Forest) để cải thiện độ chính xác.
"""
import os
import sys
import io

# Sửa lỗi UTF-8 trên Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

from src.predict import HousePricePredictor, EnsemblePredictor
from src.preprocess import normalize_district_name  # Import district normalization
import unicodedata

# City name normalization
CITY_ALIASES = {
    'HCM': 'TP.HCM',
    'TpHCM': 'TP.HCM',
    'TP.HCM': 'TP.HCM',
    'TPHCM': 'TP.HCM',
    'Ho Chi Minh': 'TP.HCM',
    'TP. Ho Chi Minh': 'TP.HCM',
}

def remove_diacritics(text):
    """Remove Vietnamese diacritics for comparison."""
    if not text:
        return text
    nfkd = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

def normalize_city_name(city):
    """Normalize city name to TP.HCM."""
    if not city or city == 'Unknown':
        return 'TP.HCM'
    city = city.strip()
    
    # Remove diacritics for matching
    city_flat = remove_diacritics(city).lower()
    
    # Direct match
    if city in CITY_ALIASES:
        return CITY_ALIASES[city]
    
    # Case-insensitive match (with diacritics removed)
    for alias, normalized in CITY_ALIASES.items():
        alias_flat = remove_diacritics(alias).lower()
        if alias_flat == city_flat:
            return normalized
    
    # If contains HCM-related keywords, normalize
    if 'hcm' in city_flat or 'ho chi minh' in city_flat:
        return 'TP.HCM'
    
    return city

# ─── Cấu hình ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn models cho ensemble
STACKING_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'stacking_ensemble_model.pkl')
XGBOOST_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgboost_model.pkl')
RANDOM_FOREST_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl')
LOCATION_ENCODINGS_PATH = os.path.join(BASE_DIR, 'models', 'location_encodings.json')

# Fallback model (single model)
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgboost_model.pkl')

RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'house_data_hcm.csv')  # Chỉ TP.HCM
FEATURE_IMP_PATH = os.path.join(BASE_DIR, 'models', 'feature_importances.csv')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Giới hạn validation
VALIDATION_LIMITS = {
    'area': {'min': 1, 'max': 1000, 'default': 50},
    'frontage': {'min': 0, 'max': 100, 'default': 0},
    'access_road': {'min': 0, 'max': 100, 'default': 0},
    'floors': {'min': 1, 'max': 50, 'default': 1},
    'bedrooms': {'min': 0, 'max': 20, 'default': 1},
    'bathrooms': {'min': 0, 'max': 20, 'default': 1},
}

# ─── Thiết lập App ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=STATIC_DIR)

# Khởi tạo predictor
print("=" * 60)
print("  DỰ ĐOÁN GIÁ NHÀ AI — Đang khởi động...")
print("=" * 60)

predictor = None
using_ensemble = False

# Thử khởi tạo Weighted Ensemble (60% XGBoost + 40% Random Forest)
if os.path.exists(XGBOOST_MODEL_PATH) and os.path.exists(RANDOM_FOREST_MODEL_PATH):
    try:
        # Đọc trọng số từ file nếu có
        ensemble_weights_path = os.path.join(BASE_DIR, 'models', 'ensemble_weights.json')
        xgb_weight, rf_weight = 0.6, 0.4  # Mặc định
        if os.path.exists(ensemble_weights_path):
            with open(ensemble_weights_path, 'r') as f:
                weights_data = json.load(f)
            xgb_weight = weights_data.get('xgboost_weight', 0.6)
            rf_weight = weights_data.get('random_forest_weight', 0.4)
            print(f"  Đã tải trọng số từ ensemble_weights.json: XGB={xgb_weight:.0%}, RF={rf_weight:.0%}")

        print(f"\n🔄 Đang khởi tạo Weighted Ensemble (XGB={xgb_weight:.0%} + RF={rf_weight:.0%})...")
        predictor = EnsemblePredictor(
            xgboost_model_path=XGBOOST_MODEL_PATH,
            random_forest_model_path=RANDOM_FOREST_MODEL_PATH,
            location_encodings_path=LOCATION_ENCODINGS_PATH,
            xgboost_weight=xgb_weight,
            rf_weight=rf_weight,
        )
        using_ensemble = True
        print("  ✓ Weighted Ensemble Predictor khởi tạo thành công!")
    except Exception as e:
        print(f"  ⚠️ Lỗi khởi tạo Ensemble: {e}")
        predictor = None

# Fallback sang single model (XGBoost)
if predictor is None:
    if os.path.exists(FALLBACK_MODEL_PATH):
        print(f"\n🔄 Sử dụng single model: {FALLBACK_MODEL_PATH}")
        predictor = HousePricePredictor(FALLBACK_MODEL_PATH, LOCATION_ENCODINGS_PATH)
    else:
        print("⚠️ Không tìm thấy model! Chạy 'python -m src.train' trước.")

# Tải sẵn thống kê dataset
print("\nĐang tải thống kê dataset...")
raw_df = pd.read_csv(RAW_DATA_PATH)

def _parse_address(addr):
    """Trích xuất 'Quận/Huyện, Thành phố' từ chuỗi địa chỉ và normalize."""
    if not isinstance(addr, str):
        return 'Unknown, TP.HCM'
    parts = addr.split(',')
    if len(parts) >= 2:
        city = normalize_city_name(parts[-1].strip())
        district = normalize_district_name(parts[-2].strip())
    else:
        district = 'Unknown'
        city = 'TP.HCM'
    return f"{district}, {city}"

raw_df['_district_city'] = raw_df['Address'].apply(_parse_address)

# Tính thống kê theo vị trí
_loc_groups = raw_df.dropna(subset=['_district_city', 'Price']).groupby('_district_city')
location_stats = []
for loc, grp in _loc_groups:
    if len(grp) < 10:
        continue
    location_stats.append({
        'location': loc,
        'count': int(len(grp)),
        'avg_price': round(float(grp['Price'].mean()), 2),
        'median_price': round(float(grp['Price'].median()), 2),
        'min_price': round(float(grp['Price'].min()), 2),
        'max_price': round(float(grp['Price'].max()), 2),
        'avg_area': round(float(grp['Area'].mean()), 1),
    })
location_stats.sort(key=lambda x: x['count'], reverse=True)
print(f"Đã tính thống kê cho {len(location_stats)} vị trí.")

# Ánh xạ vị trí -> danh sách địa chỉ
loc_to_addresses = {}
for loc, grp in _loc_groups:
    addrs = grp['Address'].dropna().unique().tolist()
    if addrs:
        loc_to_addresses[loc] = addrs


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Phục vụ giao diện web chính."""
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/<path:filename>')
def static_files(filename):
    """Phục vụ các file tĩnh (CSS, JS, images)."""
    return send_from_directory(STATIC_DIR, filename)


def validate_input(data):
    """
    Validate và sanitize dữ liệu đầu vào.
    Trả về (is_valid, errors, sanitized_data).
    """
    errors = []
    sanitized = {}

    # Các trường bắt buộc
    required = ['area', 'floors', 'bedrooms', 'bathrooms']
    for field in required:
        val = data.get(field)
        if val is None or val == '':
            errors.append(f'Thiếu trường bắt buộc: {field}')
            continue

        try:
            val = float(val)
        except (ValueError, TypeError):
            errors.append(f'Giá trị không hợp lệ cho {field}: phải là số')
            continue

        if field in VALIDATION_LIMITS:
            limits = VALIDATION_LIMITS[field]
            if val < limits['min'] or val > limits['max']:
                errors.append(f'{field} phải nằm trong khoảng {limits["min"]} và {limits["max"]}')

        sanitized[field] = val

    # Các trường tùy chọn
    optional = ['frontage', 'access_road']
    for field in optional:
        val = data.get(field)
        if val is None or val == '':
            val = VALIDATION_LIMITS[field]['default']
        else:
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = VALIDATION_LIMITS[field]['default']

        if field in VALIDATION_LIMITS:
            limits = VALIDATION_LIMITS[field]
            val = max(limits['min'], min(limits['max'], val))

        sanitized[field] = val

    # Các trường phân loại
    categorical = ['house_direction', 'balcony_direction', 'legal_status', 'furniture_state']
    for field in categorical:
        sanitized[field] = data.get(field, 'Unknown')

    sanitized['property_type'] = data.get('property_type', 'Nhà')

    # Địa chỉ (tùy chọn)
    sanitized['address'] = data.get('address', '')

    return len(errors) == 0, errors, sanitized


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Dự đoán giá nhà từ JSON đầu vào.
    Sử dụng Ensemble nếu có sẵn.

    Expected JSON body:
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
        "address": ""
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Không có dữ liệu JSON'}), 400

        # Validate đầu vào
        is_valid, errors, validated_data = validate_input(data)
        if not is_valid:
            return jsonify({'error': 'Validation thất bại', 'details': errors}), 400

        # Dự đoán với ensemble
        result = predictor.predict(validated_data)
        
        # Thêm mode vào response
        result['mode'] = 'ensemble' if using_ensemble else 'single_model'
        
        return jsonify({'success': True, **result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/single', methods=['POST'])
def predict_single():
    """
    Dự đoán với single model (XGBoost hoặc Random Forest).
    Query param: model=xgboost|random_forest
    
    Hữu ích khi muốn so sánh trực tiếp kết quả của từng model.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Không có dữ liệu JSON'}), 400

        is_valid, errors, validated_data = validate_input(data)
        if not is_valid:
            return jsonify({'error': 'Validation thất bại', 'details': errors}), 400
        
        model_type = request.args.get('model', 'xgboost').lower()
        
        # Import models trực tiếp cho single prediction
        import pickle
        
        if model_type == 'xgboost' and os.path.exists(XGBOOST_MODEL_PATH):
            with open(XGBOOST_MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        elif model_type == 'random_forest' and os.path.exists(RANDOM_FOREST_MODEL_PATH):
            with open(RANDOM_FOREST_MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        else:
            return jsonify({'error': f'Model {model_type} không tồn tại'}), 404
        
        # Predict với single model
        features_raw_df = predictor.preprocess_single(validated_data)
        pred = float(model.predict(features_raw_df)[0])
        pred = max(0.1, pred)
        
        return jsonify({
            'success': True,
            'model': model_type,
            'price_billion_vnd': round(pred, 2),
            'price_vnd': int(pred * 1_000_000_000),
            'price_formatted': f"{pred:.2f} tỷ VNĐ" if pred >= 1 else f"{pred * 1000:.0f} triệu VNĐ"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare', methods=['POST'])
def compare_models():
    """
    So sánh dự đoán của XGBoost, Random Forest và Ensemble.
    Trả về cả 3 kết quả để dễ dàng so sánh.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Không có dữ liệu JSON'}), 400

        is_valid, errors, validated_data = validate_input(data)
        if not is_valid:
            return jsonify({'error': 'Validation thất bại', 'details': errors}), 400
        
        import pickle
        
        # Load both models
        xgb_pred = rf_pred = ensemble_pred = None
        xgb_model_path = XGBOOST_MODEL_PATH
        rf_model_path = RANDOM_FOREST_MODEL_PATH
        
        # XGBoost prediction
        if os.path.exists(xgb_model_path):
            with open(xgb_model_path, 'rb') as f:
                xgb_model = pickle.load(f)
            features_raw_df = predictor.preprocess_single(validated_data)
            xgb_pred = float(xgb_model.predict(features_raw_df)[0])
            xgb_pred = max(0.1, xgb_pred)
        
        # Random Forest prediction
        if os.path.exists(rf_model_path):
            with open(rf_model_path, 'rb') as f:
                rf_model = pickle.load(f)
            features_raw_df = predictor.preprocess_single(validated_data)
            rf_pred = float(rf_model.predict(features_raw_df)[0])
            rf_pred = max(0.1, rf_pred)
        
        # Ensemble prediction (weighted average)
        if xgb_pred and rf_pred:
            ensemble_pred = 0.6 * xgb_pred + 0.4 * rf_pred
        
        def format_price(p):
            if p is None:
                return None
            return f"{p:.2f} tỷ VNĐ" if p >= 1 else f"{p * 1000:.0f} triệu VNĐ"
        
        result = {
            'xgboost': {
                'price_billion_vnd': round(xgb_pred, 2) if xgb_pred else None,
                'price_formatted': format_price(xgb_pred),
            },
            'random_forest': {
                'price_billion_vnd': round(rf_pred, 2) if rf_pred else None,
                'price_formatted': format_price(rf_pred),
            },
            'ensemble': {
                'price_billion_vnd': round(ensemble_pred, 2) if ensemble_pred else None,
                'price_formatted': format_price(ensemble_pred),
            },
            'comparison': {}
        }
        
        # Thêm comparison metrics nếu có đủ dữ liệu
        if xgb_pred and rf_pred:
            diff = abs(xgb_pred - rf_pred)
            avg = (xgb_pred + rf_pred) / 2
            result['comparison'] = {
                'absolute_difference': round(diff, 2),
                'percentage_difference': round((diff / avg) * 100, 2) if avg > 0 else 0,
                'models_agree': diff < 0.5,  # Models "đồng ý" nếu khác nhau < 0.5 tỷ
            }
        
        return jsonify({'success': True, **result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """Trả về thống kê dataset cho dashboard."""
    try:
        stats_data = {
            'total_properties': int(len(raw_df)),
            'price': {
                'mean': round(float(raw_df['Price'].mean()), 2),
                'median': round(float(raw_df['Price'].median()), 2),
                'min': round(float(raw_df['Price'].min()), 2),
                'max': round(float(raw_df['Price'].max()), 2),
                'std': round(float(raw_df['Price'].std()), 2),
            },
            'area': {
                'mean': round(float(raw_df['Area'].mean()), 1),
                'median': round(float(raw_df['Area'].median()), 1),
                'min': round(float(raw_df['Area'].min()), 1),
                'max': round(float(raw_df['Area'].max()), 1),
            },
            'price_distribution': _get_price_distribution(),
            'area_distribution': _get_area_distribution(),
            'model_mode': 'ensemble' if using_ensemble else 'single_model',
        }

        if os.path.exists(FEATURE_IMP_PATH):
            fi_raw_df = pd.read_csv(FEATURE_IMP_PATH)
            stats_data['feature_importances'] = {
                'features': fi_raw_df['feature'].tolist()[:15],
                'importances': [round(x, 4) for x in fi_raw_df['importance'].tolist()[:15]],
            }

        return jsonify(stats_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/locations', methods=['GET'])
def locations():
    """
    Trả về thống kê vị trí, tùy chọn lọc theo ?q= search query.
    """
    try:
        query = request.args.get('q', '').strip().lower()
        results = location_stats

        if query:
            results = [
                loc for loc in location_stats
                if query in loc['location'].lower()
            ]

        return jsonify({
            'locations': results[:50],
            'total': len(results),
            'query': query or None,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/location-detail/<path:location>', methods=['GET'])
def location_detail(location):
    """
    Trả về thống kê chi tiết cho một vị trí.
    """
    try:
        match = None
        for loc in location_stats:
            if loc['location'] == location:
                match = loc
                break

        if not match:
            return jsonify({'error': f'Không tìm thấy vị trí: {location}'}), 404

        loc_raw_df = raw_df[raw_df['_district_city'] == location]
        bins = [0, 2, 4, 6, 8, 10, 15, 20, 30, 50, float('inf')]
        counts, edges = np.histogram(loc_raw_df['Price'].dropna(), bins=bins)
        labels = []
        for i in range(len(bins)-1):
            if i == len(bins) - 2:
                labels.append(f">{bins[i]}")
            else:
                labels.append(f"{bins[i]}-{bins[i+1]}")
        price_dist = {
            'labels': labels,
            'counts': counts.tolist(),
        }

        sample_addresses = loc_to_addresses.get(location, [])
        representative_address = sample_addresses[0] if sample_addresses else ''

        return jsonify({
            **match,
            'price_distribution': price_dist,
            'representative_address': representative_address,
            'sample_addresses_count': len(sample_addresses),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _get_price_distribution():
    """Lấy dữ liệu histogram giá."""
    bins = [0, 2, 4, 6, 8, 10, 15, 20, 30, 50, float('inf')]
    counts, edges = np.histogram(raw_df['Price'].dropna(), bins=bins)
    labels = []
    for i in range(len(bins)-1):
        if i == len(bins) - 2:
            labels.append(f">{bins[i]}")
        else:
            labels.append(f"{bins[i]}-{bins[i+1]}")
    return {'labels': labels, 'counts': counts.tolist()}


def _get_area_distribution():
    """Lấy dữ liệu histogram diện tích."""
    bins = [0, 30, 50, 70, 100, 150, 200, 300, float('inf')]
    counts, edges = np.histogram(raw_df['Area'].dropna(), bins=bins)
    labels = []
    for i in range(len(bins)-1):
        if i == len(bins) - 2:
            labels.append(f">{bins[i]}")
        else:
            labels.append(f"{bins[i]}-{bins[i+1]}")
    return {'labels': labels, 'counts': counts.tolist()}


@app.route('/api/shap/<filename>')
def get_shap_plot(filename):
    """Cung cấp các biểu đồ SHAP đã được tạo trong quá trình training."""
    shap_dir = os.path.join(BASE_DIR, 'models', 'shap_plots')
    if not os.path.exists(os.path.join(shap_dir, filename)):
        return jsonify({'error': 'Image not found'}), 404
    return send_from_directory(shap_dir, filename)


@app.route('/api/location-data', methods=['GET'])
def get_location_data():
    """
    Trả về danh sách tất cả các quận, phường, đường cho autocomplete.
    """
    try:
        # Define all HCMC districts (with normalized names for deduplication)
        ALL_HCMC_DISTRICTS = {
            # Quận nội thành (theo số)
            'Quận 1': 'Quận 1', 'Quận 3': 'Quận 3', 'Quận 4': 'Quận 4',
            'Quận 5': 'Quận 5', 'Quận 6': 'Quận 6', 'Quận 7': 'Quận 7',
            'Quận 8': 'Quận 8', 'Quận 10': 'Quận 10', 'Quận 11': 'Quận 11',
            'Quận 12': 'Quận 12',
            # Quận nội thành (theo tên)
            'Tân Bình': 'Quận Tân Bình', 'Quận Tân Bình': 'Quận Tân Bình',
            'Tân Phú': 'Quận Tân Phú', 'Quận Tân Phú': 'Quận Tân Phú',
            'Bình Tân': 'Quận Bình Tân', 'Quận Bình Tân': 'Quận Bình Tân',
            'Bình Thạnh': 'Quận Bình Thạnh', 'Quận Bình Thạnh': 'Quận Bình Thạnh',
            'Phú Nhuận': 'Quận Phú Nhuận', 'Quận Phú Nhuận': 'Quận Phú Nhuận',
            'Gò Vấp': 'Quận Gò Vấp', 'Quận Gò Vấp': 'Quận Gò Vấp',
            # Quận 2, 9 cũ → Thành phố Thủ Đức (sáp nhập 2021)
            'Quận 2': 'Thành phố Thủ Đức', 'Quận 2 (Thủ Đức)': 'Thành phố Thủ Đức',
            'Quận 9': 'Thành phố Thủ Đức', 'Quận 9 (Thủ Đức)': 'Thành phố Thủ Đức',
            'Thủ Đức': 'Thành phố Thủ Đức',
            # Huyện ngoại thành
            'Bình Chánh': 'Huyện Bình Chánh', 'Huyện Bình Chánh': 'Huyện Bình Chánh',
            'Cần Giờ': 'Huyện Cần Giờ', 'Huyện Cần Giờ': 'Huyện Cần Giờ',
            'Nhà Bè': 'Huyện Nhà Bè', 'Huyện Nhà Bè': 'Huyện Nhà Bè',
            'Hóc Môn': 'Huyện Hóc Môn', 'Huyện Hóc Môn': 'Huyện Hóc Môn',
            'Củ Chi': 'Huyện Củ Chi', 'Huyện Củ Chi': 'Huyện Củ Chi',
        }
        
        # All valid districts to ensure completeness
        ALL_VALID_DISTRICTS = [
            'Quận 1', 'Quận 3', 'Quận 4', 'Quận 5', 'Quận 6', 'Quận 7', 'Quận 8',
            'Quận 10', 'Quận 11', 'Quận 12',
            'Quận Tân Bình', 'Quận Tân Phú', 'Quận Bình Tân',
            'Quận Bình Thạnh', 'Quận Phú Nhuận', 'Quận Gò Vấp',
            'Thành phố Thủ Đức',
            'Huyện Bình Chánh', 'Huyện Cần Giờ', 'Huyện Nhà Bè',
            'Huyện Hóc Môn', 'Huyện Củ Chi',
        ]
        
        # Get districts from data and normalize
        districts_from_data = raw_df['District'].dropna().unique().tolist()
        
        seen = set()
        districts = []
        for d in districts_from_data:
            normalized = ALL_HCMC_DISTRICTS.get(d, d)
            if normalized not in seen:
                seen.add(normalized)
                districts.append(normalized)
        
        # Ensure ALL valid districts are included even if not in data
        for d in ALL_VALID_DISTRICTS:
            if d not in seen:
                seen.add(d)
                districts.append(d)
        
        # Sort: Quận 1, 2, 3... first, then others alphabetically
        def district_sort_key(x):
            if x.startswith('Quận '):
                parts = x.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return (0, int(parts[1]), x)
            elif x == 'Thành phố Thủ Đức':
                return (1, 0, x)  # After numbered districts
            return (2, 0, x)  # Huyện last
        
        districts.sort(key=district_sort_key)
        
        # Extract wards from Address
        wards = set()
        for addr in raw_df['Address'].dropna():
            parts = addr.split(',')
            if len(parts) >= 3:
                ward = parts[-3].strip()
                if ward and len(ward) > 2:
                    wards.add(ward)
        wards = sorted(list(wards))
        
        # Extract streets from Address
        streets = set()
        for addr in raw_df['Address'].dropna():
            parts = addr.split(',')
            if len(parts) >= 1:
                first_part = parts[0].strip()
                # Remove house number prefix
                import re
                street = re.sub(r'^\d+[\/\d\w]*\s*', '', first_part)
                street = re.sub(r'^\d+[A-Za-z]?\s*', '', street)
                street = street.strip()
                if street and len(street) > 2:
                    streets.add(street)
        streets = sorted(list(streets))
        
        return jsonify({
            'districts': districts,
            'wards': wards,
            'streets': streets,
            'counts': {
                'districts': len(districts),
                'wards': len(wards),
                'streets': len(streets),
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search-streets', methods=['GET'])
def search_streets():
    """
    Tìm kiếm đường theo query string.
    """
    try:
        query = request.args.get('q', '').strip().lower()
        if len(query) < 2:
            return jsonify({'streets': []})
        
        import re
        streets = set()
        for addr in raw_df['Address'].dropna():
            parts = addr.split(',')
            if len(parts) >= 1:
                first_part = parts[0].strip()
                street = re.sub(r'^\d+[\/\d\w]*\s*', '', first_part)
                street = re.sub(r'^\d+[A-Za-z]?\s*', '', street)
                street = street.strip()
                if street and len(street) > 2:
                    # Check if matches query
                    if query in street.lower() or query in _remove_accents(street.lower()):
                        streets.add(street)
        
        # Sort by relevance (exact match first, then starts with)
        streets_list = sorted(list(streets))
        streets_list.sort(key=lambda x: (
            0 if x.lower() == query else 
            1 if x.lower().startswith(query) else 
            2
        ))
        
        return jsonify({
            'streets': streets_list[:30],
            'query': query,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _remove_accents(text):
    """Remove Vietnamese accents from text."""
    import unicodedata
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = text.replace('đ', 'd').replace('Đ', 'D')
    return text


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Server đang chạy tại http://localhost:5000")
    print("=" * 60 + "\n")
    # Tắt use_reloader để tránh sinh ra tiến trình python.exe chạy ngầm (zombie process)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
