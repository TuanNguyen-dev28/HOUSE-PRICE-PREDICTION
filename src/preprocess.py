"""
Data Preprocessing Module - Enhanced với Anti-Leakage Target Encoding
Sử dụng Smoothed/K-Fold Target Encoding để giảm data leakage và overfitting.

Key Improvements:
1. Smoothed Target Encoding với high smoothing cho Street
2. K-Fold OOF Encoding để ngăn leakage trong training
3. Geographic Hierarchy với giảm trọng số Street
4. Geo Intelligence features (KMeans, Urban Index, etc.)
5. Premium Location Detection không dùng target encoding

Architecture:
Raw Address → Geocoding → Lat/Lon → Distance Features → 
Geo Intelligence → Smoothed Encoding → Land Value Features → Ensemble Model
"""
import numpy as np
import pandas as pd
import os
import sys
import io
import json
from sklearn.model_selection import KFold

# Fix UTF-8 on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import new modules
from src.smooth_encoding import (
    SmoothedTargetEncoder,
    KFoldTargetEncoder,
    GeoWeightedEncoder,
    create_leakage_free_encoder,
    get_premium_street_bonus,
)
from src.geo_intelligence import (
    GeoIntelligence,
    PremiumLocationDetector,
    add_geo_features,
    get_location_score,
)
from src.location_features import (
    add_lat_lon_from_district,
    generate_location_features,
    DISTRICT_CENTROIDS,
    save_location_data,
)


def load_data(path):
    """Tải dữ liệu từ file CSV"""
    return pd.read_csv(path)


# ═══════════════════════════════════════════════════════════════════════════════
# DISTRICT NORMALIZATION - Chuẩn hóa tên quận/huyện để tránh trùng lặp
# ═══════════════════════════════════════════════════════════════════════════════

DISTRICT_ALIASES = {
    # Quận nội thành - các biến thể
    'Tân Bình': 'Quận Tân Bình',
    'quận tân bình': 'Quận Tân Bình',
    'Quận Tân Bình': 'Quận Tân Bình',
    'Tân Phú': 'Quận Tân Phú',
    'quận tân phú': 'Quận Tân Phú',
    'Quận Tân Phú': 'Quận Tân Phú',
    'Bình Tân': 'Quận Bình Tân',
    'quận bình tân': 'Quận Bình Tân',
    'Quận Bình Tân': 'Quận Bình Tân',
    'Bình Thạnh': 'Quận Bình Thạnh',
    'quận bình thạnh': 'Quận Bình Thạnh',
    'Quận Bình Thạnh': 'Quận Bình Thạnh',
    'Phú Nhuận': 'Quận Phú Nhuận',
    'quận phú nhuận': 'Quận Phú Nhuận',
    'Quận Phú Nhuận': 'Quận Phú Nhuận',
    'Gò Vấp': 'Quận Gò Vấp',
    'quận gò vấp': 'Quận Gò Vấp',
    'Quận Gò Vấp': 'Quận Gò Vấp',
    
    # Huyện ngoại thành - các biến thể  
    'Bình Chánh': 'Huyện Bình Chánh',
    'huyện bình chánh': 'Huyện Bình Chánh',
    'Huyện Bình Chánh': 'Huyện Bình Chánh',
    'Cần Giờ': 'Huyện Cần Giờ',
    'huyện cần giờ': 'Huyện Cần Giờ',
    'Huyện Cần Giờ': 'Huyện Cần Giờ',
    'Nhà Bè': 'Huyện Nhà Bè',
    'huyện nhà bè': 'Huyện Nhà Bè',
    'Huyện Nhà Bè': 'Huyện Nhà Bè',
    'Hóc Môn': 'Huyện Hóc Môn',
    'huyện hóc môn': 'Huyện Hóc Môn',
    'Huyện Hóc Môn': 'Huyện Hóc Môn',
    'Củ Chi': 'Huyện Củ Chi',
    'huyện củ chi': 'Huyện Củ Chi',
    'Huyện Củ Chi': 'Huyện Củ Chi',
    
    # Quận 2, 9 cũ → Thành phố Thủ Đức (sáp nhập 2021)
    'Quận 2': 'Thành phố Thủ Đức',
    'quận 2': 'Thành phố Thủ Đức',
    'Quận 2 (Thủ Đức)': 'Thành phố Thủ Đức',
    'Quận 9': 'Thành phố Thủ Đức',
    'quận 9': 'Thành phố Thủ Đức',
    'Quận 9 (Thủ Đức)': 'Thành phố Thủ Đức',
    'Thủ Đức': 'Thành phố Thủ Đức',
    'thành phố thủ đức': 'Thành phố Thủ Đức',
    
    # TP.HCM variations
    'Hồ Chí Minh': 'TP.HCM',
    'TP HCM': 'TP.HCM',
    'TP. Hồ Chí Minh': 'TP.HCM',
    'HCM': 'TP.HCM',
    'Tp.HCM': 'TP.HCM',
    'tp.hcm': 'TP.HCM',
}

ALL_VALID_DISTRICTS = [
    # Quận nội thành (theo số)
    'Quận 1', 'Quận 3', 'Quận 4', 'Quận 5', 'Quận 6', 'Quận 7', 'Quận 8',
    'Quận 10', 'Quận 11', 'Quận 12',
    # Quận nội thành (theo tên)
    'Quận Tân Bình', 'Quận Tân Phú', 'Quận Bình Tân',
    'Quận Bình Thạnh', 'Quận Phú Nhuận', 'Quận Gò Vấp',
    # Thành phố Thủ Đức (sáp nhập từ Q2, Q9, Thủ Đức)
    'Thành phố Thủ Đức',
    # Huyện ngoại thành
    'Huyện Bình Chánh', 'Huyện Cần Giờ', 'Huyện Nhà Bè',
    'Huyện Hóc Môn', 'Huyện Củ Chi',
]


def normalize_district_name(district: str) -> str:
    """
    Normalize tên quận/huyện về dạng chuẩn.
    Xử lý các biến thể như 'Tân Bình' → 'Quận Tân Bình'.
    """
    if not district or district == 'Unknown':
        return 'Unknown'
    
    district = district.strip()
    
    # 1. Check aliases
    if district in DISTRICT_ALIASES:
        return DISTRICT_ALIASES[district]
    
    # 2. Check lowercase match
    lower_district = district.lower()
    if lower_district in DISTRICT_ALIASES:
        return DISTRICT_ALIASES[lower_district]
    
    # 3. Check if already valid
    if district in ALL_VALID_DISTRICTS:
        return district
    
    # 4. Return as-is if not found (might be valid district not in list)
    return district


def normalize_districts_in_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize tất cả tên quận/huyện trong DataFrame.
    Áp dụng sau khi extract District từ Address.
    """
    df = df.copy()
    
    if 'District' in df.columns:
        df['District'] = df['District'].apply(normalize_district_name)
        
        # Count unique after normalization
        unique_before = df['District'].nunique()
        print(f"    District normalization: {unique_before} unique districts")
    
    return df


def encode_property_type(df):
    """
    Mã hóa Property_Type (Nhà/Chung cư)
    Chung cư: 1, Nhà: 0
    """
    df = df.copy()
    if 'Property_Type' in df.columns:
        df['Is_Apartment'] = (df['Property_Type'] == 'Chung cư').astype(int)
        df = df.drop(columns=['Property_Type'])
    return df


def add_land_price_feature(df):
    """
    Thêm Land_Price_Per_M2 và Land_Value
    Đây là feature QUAN TRỌNG NHẤT quyết định giá nhà
    """
    df = df.copy()
    
    if 'Land_Price_Per_M2' in df.columns:
        df['Land_Price_Per_M2'] = pd.to_numeric(df['Land_Price_Per_M2'], errors='coerce').fillna(50)
        df['Land_Value'] = (df['Area'] * df['Land_Price_Per_M2']) / 1000
        print(f"    Land_Price_Per_M2: min={df['Land_Price_Per_M2'].min():.0f}M, max={df['Land_Price_Per_M2'].max():.0f}M, mean={df['Land_Price_Per_M2'].mean():.0f}M")
        print(f"    Land_Value: min={df['Land_Value'].min():.2f}T, max={df['Land_Value'].max():.2f}T, mean={df['Land_Value'].mean():.2f}T")
    else:
        df['Land_Price_Per_M2'] = 50
        df['Land_Value'] = (df['Area'] * df['Land_Price_Per_M2']) / 1000
    
    return df


def extract_location_features(df):
    """
    Trích xuất Đường, Phường/Xã, Quận/Huyện và Thành phố từ cột Địa chỉ.
    """
    if 'Address' not in df.columns:
        return df
    
    df = df.copy()
    df['Address'] = df['Address'].fillna('Unknown')
    
    parts = df['Address'].str.split(',')
    
    df['City'] = parts.str[-1].str.strip().fillna('Unknown')
    df['District'] = parts.str[-2].str.strip().fillna('Unknown')
    df['Ward'] = parts.str[-3].str.strip().fillna('Unknown')
    
    def clean_street(p):
        if len(p) < 4:
            return 'Unknown'
        street_raw = p[-4].strip()
        import re
        street_clean = re.sub(r'^\d+[\/\d\w]*\s*', '', street_raw)
        return street_clean if street_clean else 'Unknown'
    
    df['Street'] = parts.apply(clean_street)
    
    return df


def encode_directions(df):
    """One-hot encode cho hướng nhà và ban công."""
    df = df.copy()
    
    direction_cols = ['House direction', 'Balcony direction']
    for col in direction_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=False)
    
    return df


def encode_ordinal_features(df):
    """Ordinal encode cho Pháp lý và Nội thất."""
    df = df.copy()
    
    # Legal status
    legal_order = {
        'Have certificate': 3,
        'Sale contract': 2,
        'In progress': 1,
        'Pending': 0,
    }
    if 'Legal status' in df.columns:
        df['Legal status'] = df['Legal status'].fillna('Pending')
        df['Legal_status_ordinal'] = df['Legal status'].map(legal_order).fillna(0).astype(int)
        df = df.drop(columns=['Legal status'])
    
    # Furniture state
    furniture_order = {
        'Full': 2,
        'Basic': 1,
        'Empty': 0,
    }
    if 'Furniture state' in df.columns:
        df['Furniture state'] = df['Furniture state'].fillna('Empty')
        df['Furniture_state_ordinal'] = df['Furniture state'].map(furniture_order).fillna(0).astype(int)
        df = df.drop(columns=['Furniture state'])
    
    return df


def apply_smoothed_target_encoding(
    df: pd.DataFrame,
    target_col: str = 'Price',
    use_kfold: bool = True,
) -> tuple:
    """
    Apply smoothed target encoding với anti-leakage protections.
    
    Args:
        df: DataFrame cần encode
        target_col: Tên cột target
        use_kfold: Nếu True, dùng K-Fold OOF encoding
        
    Returns:
        Tuple of (encoded_df, encoder, encodings_dict)
    """
    print("  Applying Smoothed Target Encoding (Anti-Leakage)...")
    
    # Smoothing factors - Street cao nhất vì ít data nhất
    smoothing_config = {
        'District': 10,   # Có nhiều data
        'Ward': 30,       # Medium
        'Street': 50,     # Ít data nhất, smoothing cao
        'Cluster': 20,
    }
    
    if use_kfold:
        # K-Fold OOF Encoding (recommended)
        encoder = KFoldTargetEncoder(
            smoothing_district=smoothing_config['District'],
            smoothing_ward=smoothing_config['Ward'],
            smoothing_street=smoothing_config['Street'],
            smoothing_cluster=smoothing_config['Cluster'],
            min_samples=5,
            n_folds=5,
            reduce_street_importance=0.3,  # 30% blend với global mean
        )
        df_encoded = encoder.fit_transform_kfold(df, target_col)
        print("    Using K-Fold OOF Encoding (prevents leakage)")
    else:
        # Simple Smoothed Encoding
        encoder = SmoothedTargetEncoder(
            smoothing_district=smoothing_config['District'],
            smoothing_ward=smoothing_config['Ward'],
            smoothing_street=smoothing_config['Street'],
            smoothing_cluster=smoothing_config['Cluster'],
            min_samples=5,
            reduce_street_importance=True,
        )
        df_encoded = encoder.fit_transform(df, target_col)
        print("    Using Smoothed Encoding")
    
    # Get final encoder for saving
    final_encoder = encoder.final_encoder_ if hasattr(encoder, 'final_encoder_') else encoder
    
    # Create encodings dict for saving
    encodings = {
        'global_mean': final_encoder.get_global_mean(),
        'district_encoding': final_encoder.get_encoding('District'),
        'ward_encoding': final_encoder.get_encoding('Ward'),
        'street_encoding': final_encoder.get_encoding('Street'),
        'smoothing_config': smoothing_config,
        'anti_leakage': {
            'use_kfold': use_kfold,
            'reduce_street_importance': 0.3,
            'min_samples': 5,
        }
    }
    
    # Print summary
    summary = final_encoder.summary()
    print(f"    Global mean: {summary['global_mean']:.2f} tỷ")
    for level, stats in summary['levels'].items():
        print(f"    {level}: {stats['valid_categories']}/{stats['total_categories']} valid categories")
    
    return df_encoded, final_encoder, encodings


def apply_geo_intelligence(df: pd.DataFrame, n_clusters: int = 20) -> pd.DataFrame:
    """
    Apply Geo Intelligence features.
    
    Args:
        df: DataFrame cần transform
        n_clusters: Số lượng clusters
        
    Returns:
        DataFrame với geo features mới
    """
    print("  Applying Geo Intelligence Features...")
    
    # Add all geo features
    df = add_geo_features(df, n_clusters=n_clusters)
    
    # Count new columns
    geo_cols = [
        'Location_Cluster', 'Cluster_Size', 'Cluster_Lat', 'Cluster_Lon',
        'Urban_Development_Index', 'Amenity_Count_1km', 'Amenity_Count_3km',
        'Total_Amenity_Score', 'Transit_Score', 'Walkability_Score',
        'Overall_Accessibility', 'Distance_North', 'Distance_East',
        'Quadrant_NE', 'Quadrant_NW', 'Quadrant_SE', 'Quadrant_SW',
        'Distance_From_CBD_Normalized', 'Near_CBD', 'Very_Near_CBD',
        'Near_Metro_Core', 'High_Amenity', 'Premium_Location_Score',
        'Is_Premium_Street', 'Is_Premium_District', 'Is_Premium_Ward',
        'Premium_Indicators_Count', 'Is_Premium_Location', 'Premium_Bonus_Multiplier',
    ]
    
    added_cols = [c for c in geo_cols if c in df.columns]
    print(f"    Added {len(added_cols)} geo intelligence features")
    
    return df


def preprocess_data(
    df: pd.DataFrame,
    apply_target_encoding: bool = True,
    encodings: dict = None,
    use_kfold: bool = True,
    n_clusters: int = 20,
) -> tuple:
    """
    Tiền xử lý dữ liệu với Anti-Leakage Pipeline.
    
    Pipeline:
    1. Extract location features
    2. Add Lat/Lon
    3. Add distance features
    4. Add geo intelligence
    5. Add land price features
    6. Encode directions & ordinal
    7. Apply smoothed target encoding
    8. Clean up columns
    
    Args:
        df: DataFrame raw data
        apply_target_encoding: Nếu True, áp dụng target encoding
        encodings: Dict encodings (cho test data)
        use_kfold: Nếu True, dùng K-Fold encoding
        n_clusters: Số clusters cho geo intelligence
        
    Returns:
        Tuple: (processed_df, encodings_dict)
    """
    df = df.copy()
    
    print("\n" + "=" * 60)
    print("  ANTI-LEAKAGE PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # 1. Drop rows with missing Price
    df = df.dropna(subset=['Price'])
    print(f"\n[1] Loaded {len(df)} samples")
    
    # 2. Extract location from Address
    if 'Street' not in df.columns or 'Ward' not in df.columns:
        print("\n[2] Extracting location features...")
        df = extract_location_features(df)
    
    # 2.5 Normalize district names (IMPORTANT - prevents duplicates!)
    print("\n[2.5] Normalizing district names...")
    df = normalize_districts_in_df(df)
    
    # 3. Encode Property Type
    print("\n[3] Encoding property type...")
    df = encode_property_type(df)
    
    # 4. Add Lat/Lon from District centroid
    print("\n[4] Adding Lat/Lon coordinates...")
    df = add_lat_lon_from_district(df, district_col='District')
    
    # 5. Generate distance features
    print("\n[5] Generating distance features...")
    df = generate_location_features(df)
    
    # 6. Apply Geo Intelligence (KMeans, Urban Index, Premium Detection)
    print("\n[6] Applying geo intelligence...")
    df = apply_geo_intelligence(df, n_clusters=n_clusters)
    
    # 7. Add Land Price features (QUAN TRỌNG!)
    print("\n[7] Adding land price features...")
    df = add_land_price_feature(df)
    
    # 8. Encode directions
    print("\n[8] Encoding directions...")
    df = encode_directions(df)
    
    # 9. Encode ordinal features
    print("\n[9] Encoding ordinal features...")
    df = encode_ordinal_features(df)
    
    # 10. Apply Smoothed Target Encoding
    encodings_result = None
    if apply_target_encoding:
        print("\n[10] Applying smoothed target encoding...")
        df, encoder, encodings_result = apply_smoothed_target_encoding(
            df, target_col='Price', use_kfold=use_kfold
        )
    else:
        print("\n[10] Skipping target encoding (using provided encodings)")
    
    # 11. Drop text columns
    print("\n[11] Cleaning up columns...")
    cols_to_drop = ['Address', 'District', 'City', 'Ward', 'Street']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # 12. Fill missing numeric values
    print("\n[12] Filling missing values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    print("\n" + "=" * 60)
    print(f"  PREPROCESSING COMPLETE! Shape: {df.shape}")
    print("=" * 60)
    
    return df, encodings_result


def save_encodings(encodings: dict, path: str) -> None:
    """Lưu encodings ra file JSON."""
    if encodings is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(encodings, f, ensure_ascii=False, indent=2)
    print(f"  → Encodings saved to {path}")


def load_encodings(path: str) -> dict:
    """Tải encodings từ file JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """Main preprocessing pipeline."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    raw_path = os.path.join(base_dir, "data", "raw", "house_data_hcm.csv")
    processed_path = os.path.join(base_dir, "data", "processed", "house_processed.csv")
    encodings_path = os.path.join(base_dir, "models", "location_encodings.json")
    
    print("\n" + "=" * 60)
    print("  PREPROCESSING PIPELINE - Anti-Leakage Edition")
    print("=" * 60)
    
    print(f"\nLoading data from {raw_path}...")
    df = load_data(raw_path)
    print(f"  Raw data shape: {df.shape}")
    
    print("\nPreprocessing data...")
    df_processed, encodings = preprocess_data(
        df,
        apply_target_encoding=True,
        use_kfold=True,  # K-Fold OOF encoding
        n_clusters=20,
    )
    
    # Save processed data
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_processed.to_csv(processed_path, index=False)
    print(f"\n→ Processed data saved to {processed_path}")
    
    # Save encodings
    save_encodings(encodings, encodings_path)
    
    # Save location data
    location_data_path = os.path.join(base_dir, "models", "location_data.json")
    save_location_data(location_data_path)
    print(f"→ Location data saved to {location_data_path}")
    
    print("\n" + "=" * 60)
    print("  ✅ PREPROCESSING COMPLETE!")
    print("=" * 60)
    
    # Print feature summary
    print("\n📊 FEATURE SUMMARY:")
    print(f"  Total features: {len(df_processed.columns)}")
    print(f"  Original: Area, Floors, Bedrooms, etc.")
    print(f"  New: Geo Intelligence, Smoothed Encoding, Urban Index")
    
    # Show new geo features
    geo_features = [
        'Urban_Development_Index', 'Overall_Accessibility',
        'Premium_Location_Score', 'Is_Premium_Location',
        'Cluster_Encoded', 'Distance_From_CBD_Normalized',
    ]
    available = [f for f in geo_features if f in df_processed.columns]
    print(f"\n  New Geo Features ({len(available)}):")
    for f in available[:6]:
        print(f"    - {f}")


if __name__ == "__main__":
    main()
