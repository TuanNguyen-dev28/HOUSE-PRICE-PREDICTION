"""
Dự đoán giá nhà — Module dự đoán (Anti-Leakage Edition)
Hỗ trợ Smoothed Target Encoding và Geo Intelligence features.

Key Features:
1. Smoothed Target Encoding (giảm leakage)
2. Geo Intelligence (Urban Index, Premium Detection)
3. Rule-based Premium Street/District detection
4. Ensemble prediction (XGBoost + Random Forest)
"""
import numpy as np
import pandas as pd
import pickle
import os
import json
from typing import Optional, Dict, List, Tuple

from src.location_features import (
    DISTRICT_CENTROIDS,
    CBD_COORDINATES,
    METRO_STATIONS,
    HOSPITALS,
    SCHOOLS,
    MARKETS,
    haversine_distance,
    find_nearest_distance,
    is_within_radius,
)
from src.geo_intelligence import (
    PremiumLocationDetector,
    GeoIntelligence,
    get_location_score,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE COLUMNS - Must match training features
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_COLUMNS = [
    # Basic numeric features
    'Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms',
    
    # Land Price features
    'Land_Price_Per_M2', 'Land_Value',
    
    # Property type
    'Is_Apartment',
    
    # Lat/Lon
    'Latitude', 'Longitude',
    
    # Distance features
    'distance_to_cbd',
    'distance_to_nearest_metro', 'near_metro',
    'distance_to_nearest_hospital', 'near_hospital',
    'distance_to_nearest_school', 'near_school',
    'distance_to_nearest_market', 'near_market',
    
    # Cluster features
    'Cluster_Size', 'Cluster_Lat', 'Cluster_Lon',
    
    # Geo Intelligence features
    'Urban_Development_Index',
    'Amenity_Count_1km', 'Amenity_Count_3km',
    'Total_Amenity_Score',
    'Transit_Score', 'Walkability_Score', 'Overall_Accessibility',
    'Distance_North', 'Distance_East',
    'Quadrant_NE', 'Quadrant_NW', 'Quadrant_SE', 'Quadrant_SW',
    'Distance_From_CBD_Normalized',
    'Near_CBD', 'Very_Near_CBD', 'Near_Metro_Core', 'High_Amenity',
    
    # Premium Location features
    'Premium_Location_Score',
    'Is_Premium_Street', 'Is_Premium_District', 'Is_Premium_Ward',
    'Premium_Indicators_Count', 'Is_Premium_Location',
    'Premium_Bonus_Multiplier',
    
    # Direction one-hot encoded (House)
    'House direction_Bắc', 'House direction_Nam', 'House direction_Tây',
    'House direction_Tây - Bắc', 'House direction_Tây - Nam',
    'House direction_Đông', 'House direction_Đông - Bắc', 'House direction_Đông - Nam',
    
    # Direction one-hot encoded (Balcony)
    'Balcony direction_Bắc', 'Balcony direction_Nam',
    'Balcony direction_Tây', 'Balcony direction_Tây - Bắc',
    'Balcony direction_Tây - Nam', 'Balcony direction_Đông',
    'Balcony direction_Đông - Bắc', 'Balcony direction_Đông - Nam',
    
    # Ordinal encoded
    'Legal_status_ordinal', 'Furniture_state_ordinal',
    
    # Target encoded location
    'District_Encoded', 'Ward_Encoded', 'Street_Encoded',
]


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_ENSEMBLE_WEIGHTS = {'xgboost': 0.6, 'random_forest': 0.4}

HOUSE_DIRECTIONS = ['Bắc', 'Nam', 'Tây', 'Tây - Bắc', 'Tây - Nam', 'Đông', 'Đông - Bắc', 'Đông - Nam']
BALCONY_DIRECTIONS = ['Bắc', 'Nam', 'Tây', 'Tây - Bắc', 'Tây - Nam', 'Đông', 'Đông - Bắc', 'Đông - Nam']

LEGAL_STATUS_ORDINAL = {
    'Have certificate': 3, 'Sale contract': 2, 'In progress': 1, 'Pending': 0,
}
LEGAL_STATUSES = list(LEGAL_STATUS_ORDINAL.keys())

FURNITURE_STATE_ORDINAL = {'Full': 2, 'Basic': 1, 'Empty': 0}
FURNITURE_STATES = list(FURNITURE_STATE_ORDINAL.keys())

LAND_PRICES = {}
LAND_PRICES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'land_prices_hcm.json')
if os.path.exists(LAND_PRICES_PATH):
    with open(LAND_PRICES_PATH, 'r', encoding='utf-8') as f:
        LAND_PRICES = json.load(f).get('districts', {})


# ═══════════════════════════════════════════════════════════════════════════════
# PREMIUM DETECTOR - ENHANCED FOR LUXURY PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════════

PREMIUM_DETECTOR = PremiumLocationDetector(
    street_bonus=0.50,      # 50% bonus for premium streets (was 20%)
    district_bonus=0.25,     # 25% bonus for premium districts (was 10%)
    metro_bonus=0.10,       # 10% bonus for near metro (was 5%)
)


# ═══════════════════════════════════════════════════════════════════════════════
# ULTRA-PREMIUM ZONES - CBD Core areas with extreme land prices
# These areas have land prices > 300 triệu/m²
# ═══════════════════════════════════════════════════════════════════════════════

ULTRA_PREMIUM_STREETS = {
    # District 1 - CBD Core
    'Nguyễn Huệ': {'base_price': 700, 'district': 'Quận 1'},
    'Đồng Khởi': {'base_price': 600, 'district': 'Quận 1'},
    'Lê Lợi': {'base_price': 550, 'district': 'Quận 1'},
    'Hai Bà Trưng': {'base_price': 500, 'district': 'Quận 1'},
    'Pasteur': {'base_price': 480, 'district': 'Quận 1'},
    'Lê Thánh Tôn': {'base_price': 520, 'district': 'Quận 1'},
    'Tràng Tiền': {'base_price': 480, 'district': 'Quận 1'},
    'Đakao': {'base_price': 450, 'district': 'Quận 1'},
    # District 3 - Premium
    'Võ Văn Tần': {'base_price': 280, 'district': 'Quận 3'},
    'Nguyễn Đình Chiểu': {'base_price': 260, 'district': 'Quận 3'},
    # Binh Thanh
    'Điện Biên Phủ': {'base_price': 180, 'district': 'Bình Thạnh'},
}

ULTRA_PREMIUM_DISTRICTS = {
    'Quận 1': {'multiplier': 3.0, 'base_land_price': 500},
    'Quận 3': {'multiplier': 1.8, 'base_land_price': 250},
    'Ba Đình': {'multiplier': 2.5, 'base_land_price': 350},  # HN
}

# CBD Core radius (km) - properties within this are ultra-premium
CBD_CORE_RADIUS_KM = 1.0


def get_street_land_price(street: str, district: str) -> float:
    """
    Get land price per m² for a street in a district.
    Uses street-level lookup first, then district-level.
    
    Returns:
        Land price in triệu/m²
    """
    if not street or street == 'Unknown':
        return _get_district_base_price(district)
    
    # Normalize street name for matching - remove accents
    street_normalized = _remove_accents(street.lower().strip())
    
    # Try exact match first
    for name, info in ULTRA_PREMIUM_STREETS.items():
        name_norm = _remove_accents(name.lower())
        if street_normalized == name_norm:
            return info['base_price']
    
    # Try partial match
    for name, info in ULTRA_PREMIUM_STREETS.items():
        name_norm = _remove_accents(name.lower())
        if street_normalized in name_norm or name_norm in street_normalized:
            return info['base_price']
    
    # Try district-level streets from land_prices.json
    # First find matching district (handle accent differences)
    matched_district = _find_matching_district(district)
    if matched_district and matched_district in LAND_PRICES:
        streets_data = LAND_PRICES[matched_district].get('streets', {})
        for street_name, price in streets_data.items():
            street_name_norm = _remove_accents(street_name.lower())
            if street_normalized in street_name_norm or street_name_norm in street_normalized:
                return price
    
    # Fallback to district average
    return _get_district_base_price(district)


def _remove_accents(text: str) -> str:
    """Remove Vietnamese accents from text."""
    import unicodedata
    if not isinstance(text, str):
        return ""
    # Normalize to NFD form (decomposed)
    nfkd = unicodedata.normalize('NFD', text)
    # Remove combining characters (accents)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def _find_matching_district(district: str) -> str:
    """Find matching district in LAND_PRICES dict (handles accent differences)."""
    if not district:
        return None
    
    district_norm = _remove_accents(district.lower().strip())
    
    # Exact match first
    if district in LAND_PRICES:
        return district
    
    # Try normalized match
    for d_name in LAND_PRICES.keys():
        d_norm = _remove_accents(d_name.lower())
        if district_norm == d_norm:
            return d_name
        # Partial match
        if district_norm in d_norm or d_norm in district_norm:
            return d_name
    
    return None


def _get_district_base_price(district: str) -> float:
    """Get base land price for a district."""
    matched = _find_matching_district(district)
    if matched:
        return LAND_PRICES[matched].get('price_avg', 100)
    
    # Default fallback
    return 100  # triệu/m²


# ═══════════════════════════════════════════════════════════════════════════════
# HOUSE PRICE PREDICTOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class HousePricePredictor:
    """Single model predictor với anti-leakage features."""
    
    def __init__(self, model_path, location_encodings_path=None):
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print("Model loaded successfully.")
        
        self.district_encodings = {}
        self.ward_encodings = {}
        self.street_encodings = {}
        self.global_mean = 8.71
        
        if location_encodings_path and os.path.exists(location_encodings_path):
            print(f"Loading location encodings from {location_encodings_path}...")
            with open(location_encodings_path, 'r', encoding='utf-8') as f:
                enc = json.load(f)
            self.district_encodings = enc.get('district_encoding', {})
            self.ward_encodings = enc.get('ward_encoding', {})
            self.street_encodings = enc.get('street_encoding', {})
            self.global_mean = enc.get('global_mean', 8.71)
            print(f"  Loaded {len(self.district_encodings)} districts, "
                  f"{len(self.ward_encodings)} wards, {len(self.street_encodings)} streets")
    
    def _get_lat_lon(self, district: str) -> Tuple[float, float]:
        """Lấy tọa độ từ district name."""
        if district in DISTRICT_CENTROIDS:
            return DISTRICT_CENTROIDS[district]
        for name, coords in DISTRICT_CENTROIDS.items():
            if name.lower() == district.lower():
                return coords
            if district.lower() in name.lower():
                return coords
        return DISTRICT_CENTROIDS.get("Unknown", (10.8231, 106.6292))
    
    def _calculate_distance_features(self, lat: float, lon: float) -> Dict[str, float]:
        """Tính distance features từ tọa độ."""
        if pd.isna(lat) or pd.isna(lon):
            return {f: np.nan for f in [
                'distance_to_cbd', 'distance_to_nearest_metro', 'near_metro',
                'distance_to_nearest_hospital', 'near_hospital',
                'distance_to_nearest_school', 'near_school',
                'distance_to_nearest_market', 'near_market',
            ]}
        
        return {
            'distance_to_cbd': haversine_distance(lat, lon, CBD_COORDINATES['lat'], CBD_COORDINATES['lon']),
            'distance_to_nearest_metro': find_nearest_distance(lat, lon, METRO_STATIONS),
            'near_metro': is_within_radius(lat, lon, METRO_STATIONS, 1.0),
            'distance_to_nearest_hospital': find_nearest_distance(lat, lon, HOSPITALS),
            'near_hospital': is_within_radius(lat, lon, HOSPITALS, 3.0),
            'distance_to_nearest_school': find_nearest_distance(lat, lon, SCHOOLS),
            'near_school': is_within_radius(lat, lon, SCHOOLS, 2.0),
            'distance_to_nearest_market': find_nearest_distance(lat, lon, MARKETS),
            'near_market': is_within_radius(lat, lon, MARKETS, 1.5),
        }
    
    def _extract_location_parts(self, address: str) -> Dict[str, str]:
        """Trích xuất các cấp địa lý từ address."""
        if not address or not isinstance(address, str):
            return {'street': 'Unknown', 'ward': 'Unknown', 'district': 'Unknown', 'city': 'Unknown'}
        
        import re
        parts = [p.strip() for p in address.split(',')]
        
        res = {
            'city': parts[-1].strip() if len(parts) >= 1 else 'Unknown',
            'district': parts[-2].strip() if len(parts) >= 2 else 'Unknown',
            'ward': parts[-3].strip() if len(parts) >= 3 else 'Unknown',
        }
        
        # Extract street - take first part (before first comma) and clean
        if len(parts) >= 1:
            street_raw = parts[0].strip()
            # Remove house number prefix like "100/", "123A", etc.
            street_clean = re.sub(r'^[\d]+[\/\d\w]*\s*', '', street_raw)
            street_clean = re.sub(r'^\d+[A-Za-z]?\s*', '', street_clean)
            res['street'] = street_clean if street_clean else 'Unknown'
        else:
            res['street'] = 'Unknown'
        
        return res
    
    def _encode_location(self, address: str) -> Dict[str, float]:
        """Smoothed target encoding."""
        locs = self._extract_location_parts(address)
        return {
            'District_Encoded': self.district_encodings.get(locs['district'], self.global_mean),
            'Ward_Encoded': self.ward_encodings.get(locs['ward'], self.global_mean),
            'Street_Encoded': self.street_encodings.get(locs['street'], self.global_mean),
        }
    
    def _add_geo_intelligence(self, lat: float, lon: float, dist_features: Dict, cbd: Tuple[float, float] = None) -> Dict[str, float]:
        """Tính các geo intelligence features."""
        if cbd is None:
            cbd = CBD_COORDINATES
        
        features = {}
        
        # Urban Development Index
        dist_cbd = dist_features.get('distance_to_cbd', 0)
        features['Urban_Development_Index'] = max(0, 1 - dist_cbd / 20) if dist_cbd else 0.5
        
        # Amenity counts
        features['Amenity_Count_1km'] = sum([
            dist_features.get('near_metro', 0),
            dist_features.get('near_market', 0),
        ])
        features['Amenity_Count_3km'] = sum([
            dist_features.get('near_metro', 0),
            dist_features.get('near_hospital', 0),
            dist_features.get('near_school', 0),
            dist_features.get('near_market', 0),
        ])
        features['Total_Amenity_Score'] = features['Amenity_Count_3km']
        
        # Transit & Walkability Scores
        features['Transit_Score'] = 1.0 if dist_features.get('near_metro', 0) else 0.3
        features['Walkability_Score'] = 0.5 if dist_features.get('near_market', 0) else 0.2
        features['Overall_Accessibility'] = (features['Transit_Score'] + features['Walkability_Score']) / 2
        
        # Distance Normalized
        features['Distance_From_CBD_Normalized'] = min(1, dist_cbd / 20)
        
        # Quadrant features (relative to CBD)
        delta_lat = lat - cbd['lat']
        delta_lon = lon - cbd['lon']
        features['Distance_North'] = max(0, delta_lat)
        features['Distance_East'] = max(0, delta_lon)
        features['Quadrant_NE'] = 1.0 if (delta_lat >= 0 and delta_lon >= 0) else 0.0
        features['Quadrant_NW'] = 1.0 if (delta_lat >= 0 and delta_lon < 0) else 0.0
        features['Quadrant_SE'] = 1.0 if (delta_lat < 0 and delta_lon >= 0) else 0.0
        features['Quadrant_SW'] = 1.0 if (delta_lat < 0 and delta_lon < 0) else 0.0
        
        # Boolean location features
        features['Near_CBD'] = 1.0 if dist_cbd < 5 else 0.0
        features['Very_Near_CBD'] = 1.0 if dist_cbd < 2 else 0.0
        features['Near_Metro_Core'] = 1.0 if dist_features.get('near_metro', 0) else 0.0
        features['High_Amenity'] = 1.0 if features['Amenity_Count_3km'] >= 3 else 0.0
        
        return features
    
    def _add_premium_features(self, street: str, district: str, ward: str = None, near_metro: float = 0) -> Dict[str, float]:
        """Rule-based premium detection (no target encoding)."""
        info = PREMIUM_DETECTOR.get_premium_info(street, district, ward)
        premium_level = info['premium_level']
        
        features = {
            'Is_Premium_Street': int(info['is_premium_street']),
            'Is_Premium_District': int(info['is_premium_district']),
            'Is_Premium_Ward': int(info['is_premium_ward']),
            'Premium_Indicators_Count': premium_level,
            'Premium_Location_Score': premium_level / 3,
            'Is_Premium_Location': 1.0 if premium_level >= 2 else 0.0,
            'Premium_Bonus_Multiplier': info['bonus_multiplier'] * (1 + near_metro * 0.05),
        }
        
        return features
    
    def preprocess_single(self, input_data: dict) -> pd.DataFrame:
        """Tiền xử lý dữ liệu đầu vào."""
        row = {col: 0.0 for col in FEATURE_COLUMNS}
        
        # Basic features
        row['Area'] = float(input_data.get('area', 0))
        row['Frontage'] = float(input_data.get('frontage', 0))
        row['Access Road'] = float(input_data.get('access_road', 0))
        row['Floors'] = float(input_data.get('floors', 0))
        row['Bedrooms'] = float(input_data.get('bedrooms', 0))
        row['Bathrooms'] = float(input_data.get('bathrooms', 0))
        
        # Property type
        property_type = input_data.get('property_type', 'Nhà')
        row['Is_Apartment'] = 1.0 if property_type == 'Chung cư' else 0.0
        
        # Extract location first (needed for land price lookup)
        address = input_data.get('address', '')
        locs = self._extract_location_parts(address)
        
        # Land Price (ENHANCED - use street-level lookup for luxury)
        area = float(input_data.get('area', 100))
        
        # Try user-provided land price first
        user_land_price = float(input_data.get('land_price_per_m2', 0))
        
        if user_land_price > 0:
            # Use user's land price if provided
            land_price = user_land_price
        else:
            # Look up from street-level data (THIS IS KEY FOR LUXURY)
            land_price = get_street_land_price(locs.get('street', ''), locs.get('district', ''))
        
        row['Land_Price_Per_M2'] = land_price
        row['Land_Value'] = (area * land_price) / 1000
        
        # Directions
        house_dir = input_data.get('house_direction', '')
        if house_dir and house_dir in HOUSE_DIRECTIONS:
            col_name = f'House direction_{house_dir}'
            if col_name in row:
                row[col_name] = 1.0
        
        balcony_dir = input_data.get('balcony_direction', '')
        if balcony_dir and balcony_dir in BALCONY_DIRECTIONS:
            col_name = f'Balcony direction_{balcony_dir}'
            if col_name in row:
                row[col_name] = 1.0
        
        # Ordinal features
        legal = input_data.get('legal_status', '')
        row['Legal_status_ordinal'] = float(LEGAL_STATUS_ORDINAL.get(legal, 3))
        
        furniture = input_data.get('furniture_state', '')
        row['Furniture_state_ordinal'] = float(FURNITURE_STATE_ORDINAL.get(furniture, 2))
        
        # Location encoding (uses locs already extracted above)
        encoded_locs = self._encode_location(address)
        for col, val in encoded_locs.items():
            if col in row:
                row[col] = val
        
        # Lat/Lon
        lat, lon = self._get_lat_lon(locs.get('district', ''))
        row['Latitude'] = lat
        row['Longitude'] = lon
        
        # Distance features
        dist_features = self._calculate_distance_features(lat, lon)
        for feat_name, feat_value in dist_features.items():
            if feat_name in row:
                row[feat_name] = feat_value
        
        # Geo Intelligence
        geo_features = self._add_geo_intelligence(lat, lon, dist_features)
        for feat_name, feat_value in geo_features.items():
            if feat_name in row:
                row[feat_name] = feat_value
        
        # Premium features
        premium_features = self._add_premium_features(
            locs['street'], locs['district'], locs['ward'],
            dist_features.get('near_metro', 0)
        )
        for feat_name, feat_value in premium_features.items():
            if feat_name in row:
                row[feat_name] = feat_value
        
        # Cluster features (fallback defaults)
        row['Cluster_Size'] = 10.0
        row['Cluster_Lat'] = lat
        row['Cluster_Lon'] = lon
        
        df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        return df
    
    def predict(self, input_data: dict) -> dict:
        """Dự đoán giá."""
        features_df = self.preprocess_single(input_data)
        prediction = float(self.model.predict(features_df)[0])
        prediction = max(0.1, prediction)
        
        price_billion = round(prediction, 2)
        price_vnd = price_billion * 1_000_000_000
        
        if price_billion >= 1:
            price_formatted = f"{price_billion:.2f} tỷ VNĐ"
        else:
            price_formatted = f"{price_billion * 1000:.0f} triệu VNĐ"
        
        return {
            'price_billion_vnd': price_billion,
            'price_vnd': price_vnd,
            'price_formatted': price_formatted,
            'location_analysis': self._extract_location_parts(input_data.get('address', '')),
            'input_summary': {
                'area': input_data.get('area', 0),
                'floors': input_data.get('floors', 0),
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE PREDICTOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class EnsemblePredictor:
    """Ensemble predictor (XGBoost + Random Forest) với anti-leakage features."""
    
    def __init__(
        self,
        xgboost_model_path: str,
        random_forest_model_path: str,
        location_encodings_path: str,
        weights: Optional[Dict[str, float]] = None,
        xgboost_weight: float = 0.6,
        rf_weight: float = 0.4
    ):
        # Load encodings
        self.district_encodings = {}
        self.ward_encodings = {}
        self.street_encodings = {}
        self.global_mean = 8.71
        
        if os.path.exists(location_encodings_path):
            with open(location_encodings_path, 'r', encoding='utf-8') as f:
                enc = json.load(f)
            self.district_encodings = enc.get('district_encoding', {})
            self.ward_encodings = enc.get('ward_encoding', {})
            self.street_encodings = enc.get('street_encoding', {})
            self.global_mean = enc.get('global_mean', 8.71)
            print(f"  Loaded encodings: {len(self.district_encodings)} districts, "
                  f"{len(self.ward_encodings)} wards, {len(self.street_encodings)} streets")
        
        # Load models
        print("Loading XGBoost model...")
        with open(xgboost_model_path, 'rb') as f:
            self.xgboost_model = pickle.load(f)
        
        print("Loading Random Forest model...")
        with open(random_forest_model_path, 'rb') as f:
            self.random_forest_model = pickle.load(f)
        
        # Set weights
        if weights:
            self.weights = weights
        else:
            total = xgboost_weight + rf_weight
            self.weights = {
                'xgboost': xgboost_weight / total,
                'random_forest': rf_weight / total
            }
        
        print(f"\nEnsemble Weights: XGBoost={self.weights['xgboost']:.1%}, "
              f"RF={self.weights['random_forest']:.1%}")
    
    def _get_lat_lon(self, district: str) -> Tuple[float, float]:
        """Lấy tọa độ từ district."""
        if district in DISTRICT_CENTROIDS:
            return DISTRICT_CENTROIDS[district]
        for name, coords in DISTRICT_CENTROIDS.items():
            if name.lower() == district.lower():
                return coords
        return DISTRICT_CENTROIDS.get("Unknown", (10.8231, 106.6292))
    
    def _calculate_distance_features(self, lat: float, lon: float) -> Dict[str, float]:
        """Tính distance features."""
        if pd.isna(lat) or pd.isna(lon):
            return {f: np.nan for f in [
                'distance_to_cbd', 'distance_to_nearest_metro', 'near_metro',
                'distance_to_nearest_hospital', 'near_hospital',
                'distance_to_nearest_school', 'near_school',
                'distance_to_nearest_market', 'near_market',
            ]}
        
        return {
            'distance_to_cbd': haversine_distance(lat, lon, CBD_COORDINATES['lat'], CBD_COORDINATES['lon']),
            'distance_to_nearest_metro': find_nearest_distance(lat, lon, METRO_STATIONS),
            'near_metro': is_within_radius(lat, lon, METRO_STATIONS, 1.0),
            'distance_to_nearest_hospital': find_nearest_distance(lat, lon, HOSPITALS),
            'near_hospital': is_within_radius(lat, lon, HOSPITALS, 3.0),
            'distance_to_nearest_school': find_nearest_distance(lat, lon, SCHOOLS),
            'near_school': is_within_radius(lat, lon, SCHOOLS, 2.0),
            'distance_to_nearest_market': find_nearest_distance(lat, lon, MARKETS),
            'near_market': is_within_radius(lat, lon, MARKETS, 1.5),
        }
    
    def _extract_location_parts(self, address: str) -> Dict[str, str]:
        """Trích xuất location parts."""
        if not address or not isinstance(address, str):
            return {'street': 'Unknown', 'ward': 'Unknown', 'district': 'Unknown', 'city': 'Unknown'}
        
        parts = address.split(',')
        res = {
            'city': parts[-1].strip() if len(parts) >= 1 else 'Unknown',
            'district': parts[-2].strip() if len(parts) >= 2 else 'Unknown',
            'ward': parts[-3].strip() if len(parts) >= 3 else 'Unknown',
        }
        
        if len(parts) >= 4:
            street_raw = parts[-4].strip()
            import re
            street_clean = re.sub(r'^\d+[\/\d\w]*\s*', '', street_raw)
            res['street'] = street_clean if street_clean else 'Unknown'
        else:
            res['street'] = 'Unknown'
        
        return res
    
    def _encode_location(self, address: str) -> Dict[str, float]:
        """Smoothed target encoding."""
        locs = self._extract_location_parts(address)
        return {
            'District_Encoded': self.district_encodings.get(locs['district'], self.global_mean),
            'Ward_Encoded': self.ward_encodings.get(locs['ward'], self.global_mean),
            'Street_Encoded': self.street_encodings.get(locs['street'], self.global_mean),
        }
    
    def _add_geo_intelligence(self, lat: float, lon: float, dist_features: Dict, cbd: Tuple[float, float] = None) -> Dict[str, float]:
        """Tính geo intelligence features."""
        if cbd is None:
            cbd = CBD_COORDINATES
        
        features = {}
        
        dist_cbd = dist_features.get('distance_to_cbd', 0)
        features['Urban_Development_Index'] = max(0, 1 - dist_cbd / 20) if dist_cbd else 0.5
        
        features['Amenity_Count_1km'] = sum([
            dist_features.get('near_metro', 0),
            dist_features.get('near_market', 0),
        ])
        features['Amenity_Count_3km'] = sum([
            dist_features.get('near_metro', 0),
            dist_features.get('near_hospital', 0),
            dist_features.get('near_school', 0),
            dist_features.get('near_market', 0),
        ])
        features['Total_Amenity_Score'] = features['Amenity_Count_3km']
        features['Transit_Score'] = 1.0 if dist_features.get('near_metro', 0) else 0.3
        features['Walkability_Score'] = 0.5 if dist_features.get('near_market', 0) else 0.2
        features['Overall_Accessibility'] = (features['Transit_Score'] + features['Walkability_Score']) / 2
        features['Distance_From_CBD_Normalized'] = min(1, dist_cbd / 20)
        
        delta_lat = lat - cbd['lat']
        delta_lon = lon - cbd['lon']
        features['Distance_North'] = max(0, delta_lat)
        features['Distance_East'] = max(0, delta_lon)
        features['Quadrant_NE'] = 1.0 if (delta_lat >= 0 and delta_lon >= 0) else 0.0
        features['Quadrant_NW'] = 1.0 if (delta_lat >= 0 and delta_lon < 0) else 0.0
        features['Quadrant_SE'] = 1.0 if (delta_lat < 0 and delta_lon >= 0) else 0.0
        features['Quadrant_SW'] = 1.0 if (delta_lat < 0 and delta_lon < 0) else 0.0
        features['Near_CBD'] = 1.0 if dist_cbd < 5 else 0.0
        features['Very_Near_CBD'] = 1.0 if dist_cbd < 2 else 0.0
        features['Near_Metro_Core'] = 1.0 if dist_features.get('near_metro', 0) else 0.0
        features['High_Amenity'] = 1.0 if features['Amenity_Count_3km'] >= 3 else 0.0
        
        return features
    
    def _add_premium_features(self, street: str, district: str, ward: str = None, near_metro: float = 0) -> Dict[str, float]:
        """Rule-based premium detection."""
        info = PREMIUM_DETECTOR.get_premium_info(street, district, ward)
        premium_level = info['premium_level']
        
        return {
            'Is_Premium_Street': int(info['is_premium_street']),
            'Is_Premium_District': int(info['is_premium_district']),
            'Is_Premium_Ward': int(info['is_premium_ward']),
            'Premium_Indicators_Count': premium_level,
            'Premium_Location_Score': premium_level / 3,
            'Is_Premium_Location': 1.0 if premium_level >= 2 else 0.0,
            'Premium_Bonus_Multiplier': info['bonus_multiplier'] * (1 + near_metro * 0.05),
        }
    
    def preprocess_single(self, input_data: dict) -> pd.DataFrame:
        """Tiền xử lý dữ liệu."""
        row = {col: 0.0 for col in FEATURE_COLUMNS}
        
        # Basic features
        row['Area'] = float(input_data.get('area', 0))
        row['Frontage'] = float(input_data.get('frontage', 0))
        row['Access Road'] = float(input_data.get('access_road', 0))
        row['Floors'] = float(input_data.get('floors', 0))
        row['Bedrooms'] = float(input_data.get('bedrooms', 0))
        row['Bathrooms'] = float(input_data.get('bathrooms', 0))
        
        # Property type
        property_type = input_data.get('property_type', 'Nhà')
        row['Is_Apartment'] = 1.0 if property_type == 'Chung cư' else 0.0
        
        # Extract location first (needed for land price lookup)
        address = input_data.get('address', '')
        locs = self._extract_location_parts(address)
        
        # Land Price (ENHANCED - use street-level lookup for luxury)
        area = float(input_data.get('area', 100))
        user_land_price = float(input_data.get('land_price_per_m2', 0))
        
        if user_land_price > 0:
            land_price = user_land_price
        else:
            land_price = get_street_land_price(locs.get('street', ''), locs.get('district', ''))
        
        row['Land_Price_Per_M2'] = land_price
        row['Land_Value'] = (area * land_price) / 1000
        
        # Directions
        house_dir = input_data.get('house_direction', '')
        if house_dir and house_dir in HOUSE_DIRECTIONS:
            col_name = f'House direction_{house_dir}'
            if col_name in row:
                row[col_name] = 1.0
        
        balcony_dir = input_data.get('balcony_direction', '')
        if balcony_dir and balcony_dir in BALCONY_DIRECTIONS:
            col_name = f'Balcony direction_{balcony_dir}'
            if col_name in row:
                row[col_name] = 1.0
        
        # Ordinal
        legal = input_data.get('legal_status', '')
        row['Legal_status_ordinal'] = float(LEGAL_STATUS_ORDINAL.get(legal, 3))
        
        furniture = input_data.get('furniture_state', '')
        row['Furniture_state_ordinal'] = float(FURNITURE_STATE_ORDINAL.get(furniture, 2))
        
        # Location encoding (uses locs already extracted above)
        encoded_locs = self._encode_location(address)
        for col, val in encoded_locs.items():
            if col in row:
                row[col] = val
        
        # Lat/Lon
        lat, lon = self._get_lat_lon(locs.get('district', ''))
        row['Latitude'] = lat
        row['Longitude'] = lon
        
        # Distance features
        dist_features = self._calculate_distance_features(lat, lon)
        for feat_name, feat_value in dist_features.items():
            if feat_name in row:
                row[feat_name] = feat_value
        
        # Geo Intelligence
        geo_features = self._add_geo_intelligence(lat, lon, dist_features)
        for feat_name, feat_value in geo_features.items():
            if feat_name in row:
                row[feat_name] = feat_value
        
        # Premium features
        premium_features = self._add_premium_features(
            locs['street'], locs['district'], locs['ward'],
            dist_features.get('near_metro', 0)
        )
        for feat_name, feat_value in premium_features.items():
            if feat_name in row:
                row[feat_name] = feat_value
        
        # Cluster features (fallback defaults)
        row['Cluster_Size'] = 10.0
        row['Cluster_Lat'] = lat
        row['Cluster_Lon'] = lon
        
        df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        return df
    
    def predict(self, input_data: dict, include_individual: bool = True) -> dict:
        """Ensemble prediction."""
        features_df = self.preprocess_single(input_data)
        
        xgb_pred = float(self.xgboost_model.predict(features_df)[0])
        rf_pred = float(self.random_forest_model.predict(features_df)[0])
        
        ensemble_pred = (
            self.weights['xgboost'] * xgb_pred +
            self.weights['random_forest'] * rf_pred
        )
        
        pred_diff = abs(xgb_pred - rf_pred)
        confidence_margin = pred_diff * 0.5
        
        ensemble_pred = max(0.1, ensemble_pred)
        xgb_pred = max(0.1, xgb_pred)
        rf_pred = max(0.1, rf_pred)
        
        def format_price(price_billion):
            if price_billion >= 1:
                return f"{price_billion:.2f} tỷ VNĐ"
            return f"{price_billion * 1000:.0f} triệu VNĐ"
        
        return {
            'ensemble': {
                'price_billion_vnd': round(ensemble_pred, 2),
                'price_vnd': int(ensemble_pred * 1_000_000_000),
                'price_formatted': format_price(ensemble_pred),
            },
            'location_analysis': self._extract_location_parts(input_data.get('address', '')),
            'confidence_interval': {
                'lower': round(max(0.1, ensemble_pred - confidence_margin), 2),
                'upper': round(ensemble_pred + confidence_margin, 2),
                'margin': round(confidence_margin, 2),
            },
            'metadata': {
                'xgboost_weight': self.weights['xgboost'],
                'random_forest_weight': self.weights['random_forest'],
                'prediction_difference': round(pred_diff, 2),
            },
            'individual_predictions': {
                'xgboost': {'price_billion_vnd': round(xgb_pred, 2), 'price_formatted': format_price(xgb_pred)},
                'random_forest': {'price_billion_vnd': round(rf_pred, 2), 'price_formatted': format_price(rf_pred)},
            } if include_individual else None
        }
    
    def predict_batch(self, input_list: List[dict], include_individual: bool = True) -> List[dict]:
        """Predict for a batch of inputs."""
        return [self.predict(inp, include_individual=include_individual) for inp in input_list]
    
    def update_weights(self, xgboost_weight: float, rf_weight: float) -> Dict[str, float]:
        """Update ensemble weights."""
        total = xgboost_weight + rf_weight
        if total == 0:
            raise ValueError("Total weight cannot be zero")
        
        self.weights = {
            'xgboost': xgboost_weight / total,
            'random_forest': rf_weight / total
        }
        return self.weights


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def predict_price(input_data: dict) -> dict:
    """
    Quick prediction function.
    
    Args:
        input_data: Dict với các fields:
            - area: Diện tích (m²)
            - floors: Số tầng
            - frontage: Mặt tiền (m)
            - bedrooms: Số phòng ngủ
            - bathrooms: Số phòng tắm
            - address: Địa chỉ đầy đủ
            - land_price_per_m2: Giá đất/m² (triệu)
            - legal_status: Tình trạng pháp lý
            - property_type: Loại bất động sản
    
    Returns:
        Dict với price prediction
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")
    encodings_path = os.path.join(BASE_DIR, "models", "location_encodings.json")
    
    predictor = HousePricePredictor(model_path, encodings_path)
    return predictor.predict(input_data)


if __name__ == "__main__":
    # Test prediction
    print("=" * 60)
    print("  TESTING PREDICTOR")
    print("=" * 60)
    
    test_input = {
        'area': 100,
        'floors': 4,
        'frontage': 5,
        'bedrooms': 4,
        'bathrooms': 3,
        'address': '123 Lê Lợi, Phường Bến Nghé, Quận 1, TP.HCM',
        'land_price_per_m2': 500,
        'legal_status': 'Have certificate',
        'property_type': 'Nhà',
    }
    
    print("\nTest Input:")
    for k, v in test_input.items():
        print(f"  {k}: {v}")
    
    try:
        result = predict_price(test_input)
        print(f"\nPredicted Price: {result['price_formatted']}")
    except Exception as e:
        print(f"\nPrediction failed: {e}")
        print("Make sure to run preprocessing and training first!")
