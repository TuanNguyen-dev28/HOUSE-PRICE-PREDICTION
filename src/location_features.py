"""
Location Features Module - Tính toán khoảng cách từ bất động sản đến các điểm quan trọng.
Feature Engineering giúp model "nghĩ" như con người khi định giá nhà.
"""
import numpy as np
import pandas as pd
import os
import json
from math import radians, cos, sin, asin, sqrt

# ═══════════════════════════════════════════════════════════════════════════════
# DISTRICT ALIASES - Chuẩn hóa tên quận/huyện TRƯỚC KHI tra cứu tọa độ
# ═══════════════════════════════════════════════════════════════════════════════

DISTRICT_NAME_ALIASES = {
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
}


def normalize_district_for_lookup(district):
    """Normalize tên quận về dạng chuẩn để tra cứu tọa độ."""
    if not district or district == 'Unknown':
        return 'Unknown'
    district = district.strip()
    return DISTRICT_NAME_ALIASES.get(district, district)


# ═══════════════════════════════════════════════════════════════════════════════
# COORDINATE DATA - Tọa độ các điểm quan trọng trong TP.HCM
# ═══════════════════════════════════════════════════════════════════════════════

# Trung tâm các Quận/Huyện TP.HCM - CHỈ DÙNG TÊN CHUẨN
DISTRICT_CENTROIDS = {
    # === Quận nội thành (theo số) ===
    "Quận 1": (10.7758, 106.7004),
    "Quận 3": (10.7794, 106.6868),
    "Quận 4": (10.7644, 106.7087),
    "Quận 5": (10.7558, 106.6825),
    "Quận 6": (10.7475, 106.6742),
    "Quận 7": (10.7419, 106.7274),
    "Quận 8": (10.7365, 106.6825),
    "Quận 10": (10.7678, 106.6742),
    "Quận 11": (10.7644, 106.6643),
    "Quận 12": (10.8525, 106.6540),
    
    # === Quận nội thành (theo tên) ===
    "Quận Tân Bình": (10.8009, 106.6535),
    "Quận Tân Phú": (10.7864, 106.6317),
    "Quận Bình Tân": (10.7864, 106.5968),
    "Quận Bình Thạnh": (10.8034, 106.7087),
    "Quận Phú Nhuận": (10.7957, 106.6893),
    "Quận Gò Vấp": (10.8396, 106.6661),
    
    # === Thành phố Thủ Đức (sáp nhập Q2, Q9, Thủ Đức) ===
    "Thành phố Thủ Đức": (10.8610, 106.7536),
    
    # === Huyện ngoại thành ===
    "Huyện Bình Chánh": (10.8610, 106.5000),
    "Huyện Cần Giờ": (10.5000, 106.9000),
    "Huyện Nhà Bè": (10.6789, 106.7015),
    "Huyện Hóc Môn": (10.8946, 106.5945),
    "Huyện Củ Chi": (10.9897, 106.4890),
    
    # === Default ===
    "Unknown": (10.8231, 106.6292),  # Default: TP.HCM center
}


# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTANT LOCATIONS - Các điểm quan trọng để tính distance features
# ═══════════════════════════════════════════════════════════════════════════════

# CBD (Central Business District) - Trung tâm tài chính Q1
CBD_COORDINATES = {
    "name": "Quận 1 Center",
    "lat": 10.7758,
    "lon": 106.7004
}

# Metro Stations (Tàu điện ngầm Metro) - Updated 2024
METRO_STATIONS = [
    {"name": "Bến Thành", "lat": 10.7729, "lon": 106.6984},
    {"name": "Sài Gòn", "lat": 10.7799, "lon": 106.7005},
    {"name": "Nhà hát Thành phố", "lat": 10.7776, "lon": 106.7044},
    {"name": "Ba Son", "lat": 10.7725, "lon": 106.7105},
    {"name": "Văn Thánh", "lat": 10.7633, "lon": 106.7210},
    {"name": "Tân Cảng", "lat": 10.7605, "lon": 106.7320},
    {"name": "Thủ Thiêm", "lat": 10.7678, "lon": 106.7450},
    {"name": "An Phú", "lat": 10.7833, "lon": 106.7580},
    {"name": "Rạch Chiếc", "lat": 10.7950, "lon": 106.7700},
    {"name": "Phú Mỹ Hưng", "lat": 10.8011, "lon": 106.7790},
    {"name": "Thảo Điền", "lat": 10.8092, "lon": 106.7880},
    {"name": "An Khánh", "lat": 10.8180, "lon": 106.7950},
    {"name": "Khu Công nghệ cao", "lat": 10.8267, "lon": 106.8010},
    {"name": "Suối Tiên", "lat": 10.8356, "lon": 106.8060},
    # Line 2 (An Suong - Tham Luong)
    {"name": "Bến Xe Miền Tây", "lat": 10.7389, "lon": 106.6220},
    {"name": "Chợ An Đông", "lat": 10.7475, "lon": 106.6550},
    {"name": "Đinh Tiên Hoàng", "lat": 10.7550, "lon": 106.6750},
]

# Major Hospitals TP.HCM
HOSPITALS = [
    {"name": "Chợ Rẫy", "lat": 10.7633, "lon": 106.6844},
    {"name": "115 People's", "lat": 10.7815, "lon": 106.6965},
    {"name": "FV Hospital", "lat": 10.7289, "lon": 106.7080},
    {"name": "Vinmec Central Park", "lat": 10.8299, "lon": 106.7175},
    {"name": "Tam Duc Heart Hospital", "lat": 10.7260, "lon": 106.7050},
    {"name": "Children's Hospital 1", "lat": 10.7780, "lon": 106.6780},
    {"name": "Oncology Hospital", "lat": 10.7700, "lon": 106.6920},
]

# Schools/Universities TP.HCM
SCHOOLS = [
    {"name": "ĐH Bách Khoa", "lat": 10.7727, "lon": 106.6654},
    {"name": "ĐH Kinh tế TP.HCM", "lat": 10.7729, "lon": 106.6865},
    {"name": "ĐH Quốc tế", "lat": 10.8720, "lon": 106.7850},
    {"name": "ĐH KHTN", "lat": 10.7632, "lon": 106.6820},
    {"name": "ĐH Ngoại thương", "lat": 10.7697, "lon": 106.6942},
    {"name": "ĐH Luật", "lat": 10.7805, "lon": 106.6980},
    {"name": "Trường THPT chuyên Lê Hồng Phong", "lat": 10.7820, "lon": 106.6830},
    {"name": "Trường THPT Năng khiếu", "lat": 10.8700, "lon": 106.7820},
]

# Markets/Shopping Centers TP.HCM
MARKETS = [
    {"name": "Ben Thanh Market", "lat": 10.7724, "lon": 106.6982},
    {"name": "Saigon Square", "lat": 10.7738, "lon": 106.6930},
    {"name": "Diamond Plaza", "lat": 10.7790, "lon": 106.6955},
    {"name": "Vincom Center", "lat": 10.7805, "lon": 106.6990},
    {"name": "Takashimaya", "lat": 10.7815, "lon": 106.7025},
    {"name": "AEON Mall Bình Tân", "lat": 10.7650, "lon": 106.5950},
    {"name": "Lotte Mart", "lat": 10.7970, "lon": 106.6250},
    {"name": "SC Vivo City", "lat": 10.8060, "lon": 106.7310},
    {"name": "Paragon Center", "lat": 10.8110, "lon": 106.6880},
]


# ═══════════════════════════════════════════════════════════════════════════════
# DISTANCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Tính khoảng cách Haversine (km) giữa 2 điểm.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers

    return c * r


def find_nearest_distance(lat, lon, locations_list):
    """
    Tìm khoảng cách đến điểm gần nhất trong danh sách.
    """
    if pd.isna(lat) or pd.isna(lon):
        return np.nan

    min_distance = float('inf')
    for loc in locations_list:
        dist = haversine_distance(lat, lon, loc['lat'], loc['lon'])
        if dist < min_distance:
            min_distance = dist

    return min_distance


def is_within_radius(lat, lon, locations_list, radius_km):
    """
    Kiểm tra xem có điểm nào trong danh sách trong bán kính radius_km không.
    Returns: 1 if yes, 0 if no
    """
    if pd.isna(lat) or pd.isna(lon):
        return np.nan

    for loc in locations_list:
        dist = haversine_distance(lat, lon, loc['lat'], loc['lon'])
        if dist <= radius_km:
            return 1
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FEATURE ENGINEERING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_location_features(df):
    """
    Tạo các distance features từ Lat/Lon.

    Features tạo ra:
    - distance_to_cbd: Khoảng cách đến trung tâm Q1 (km)
    - distance_to_nearest_metro: Khoảng cách đến metro gần nhất (km)
    - near_metro: Có metro trong bán kính 1km (1/0)
    - distance_to_nearest_hospital: Khoảng cách đến bệnh viện gần nhất (km)
    - near_hospital: Có bệnh viện trong bán kính 3km (1/0)
    - distance_to_nearest_school: Khoảng cách đến trường gần nhất (km)
    - near_school: Có trường trong bán kính 2km (1/0)
    - distance_to_nearest_market: Khoảng cách đến chợ/trung tâm TM gần nhất (km)
    - near_market: Có chợ/TM center trong bán kính 1.5km (1/0)

    Args:
        df: DataFrame với cột 'Latitude', 'Longitude'

    Returns:
        DataFrame với các features mới
    """
    df = df.copy()

    # Kiểm tra xem đã có Lat/Lon chưa
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        print("  ⚠️ Warning: Không tìm thấy Latitude/Longitude. Bỏ qua distance features.")
        return df

    print("  Đang tạo distance features...")

    # 1. Distance to CBD
    df['distance_to_cbd'] = df.apply(
        lambda row: haversine_distance(
            row['Latitude'], row['Longitude'],
            CBD_COORDINATES['lat'], CBD_COORDINATES['lon']
        ) if pd.notna(row['Latitude']) else np.nan,
        axis=1
    )

    # 2. Distance to nearest Metro
    df['distance_to_nearest_metro'] = df.apply(
        lambda row: find_nearest_distance(row['Latitude'], row['Longitude'], METRO_STATIONS)
        if pd.notna(row['Latitude']) else np.nan,
        axis=1
    )

    # 3. Near Metro (within 1km)
    df['near_metro'] = df.apply(
        lambda row: is_within_radius(row['Latitude'], row['Longitude'], METRO_STATIONS, 1.0)
        if pd.notna(row['Latitude']) else np.nan,
        axis=1
    )

    # 4. Distance to nearest Hospital
    df['distance_to_nearest_hospital'] = df.apply(
        lambda row: find_nearest_distance(row['Latitude'], row['Longitude'], HOSPITALS)
        if pd.notna(row['Latitude']) else np.nan,
        axis=1
    )

    # 5. Near Hospital (within 3km)
    df['near_hospital'] = df.apply(
        lambda row: is_within_radius(row['Latitude'], row['Longitude'], HOSPITALS, 3.0)
        if pd.notna(row['Latitude']) else np.nan,
        axis=1
    )

    # 6. Distance to nearest School
    df['distance_to_nearest_school'] = df.apply(
        lambda row: find_nearest_distance(row['Latitude'], row['Longitude'], SCHOOLS)
        if pd.notna(row['Latitude']) else np.nan,
        axis=1
    )

    # 7. Near School (within 2km)
    df['near_school'] = df.apply(
        lambda row: is_within_radius(row['Latitude'], row['Longitude'], SCHOOLS, 2.0)
        if pd.notna(row['Latitude']) else np.nan,
        axis=1
    )

    # 8. Distance to nearest Market
    df['distance_to_nearest_market'] = df.apply(
        lambda row: find_nearest_distance(row['Latitude'], row['Longitude'], MARKETS)
        if pd.notna(row['Latitude']) else np.nan,
        axis=1
    )

    # 9. Near Market (within 1.5km)
    df['near_market'] = df.apply(
        lambda row: is_within_radius(row['Latitude'], row['Longitude'], MARKETS, 1.5)
        if pd.notna(row['Latitude']) else np.nan,
        axis=1
    )

    # Log summary
    print(f"    ✓ distance_to_cbd: mean={df['distance_to_cbd'].mean():.2f} km")
    print(f"    ✓ distance_to_nearest_metro: mean={df['distance_to_nearest_metro'].mean():.2f} km")
    print(f"    ✓ near_metro: {df['near_metro'].sum():.0f} properties ({df['near_metro'].mean()*100:.1f}%)")
    print(f"    ✓ near_hospital: {df['near_hospital'].sum():.0f} properties ({df['near_hospital'].mean()*100:.1f}%)")
    print(f"    ✓ near_school: {df['near_school'].sum():.0f} properties ({df['near_school'].mean()*100:.1f}%)")
    print(f"    ✓ near_market: {df['near_market'].sum():.0f} properties ({df['near_market'].mean()*100:.1f}%)")

    return df


def add_lat_lon_from_district(df, district_col='District'):
    """
    Thêm Lat/Lon từ District centroid.

    Args:
        df: DataFrame
        district_col: Tên cột chứa District

    Returns:
        DataFrame với cột Latitude, Longitude
    """
    df = df.copy()

    # Initialize columns
    df['Latitude'] = np.nan
    df['Longitude'] = np.nan

    # Create normalized district column for matching
    df['_district_normalized'] = df[district_col].apply(normalize_district_for_lookup)

    # Match normalized district names to coordinates
    for district, (lat, lon) in DISTRICT_CENTROIDS.items():
        mask = df['_district_normalized'] == district
        df.loc[mask & df['Latitude'].isna(), 'Latitude'] = lat
        df.loc[mask & df['Longitude'].isna(), 'Longitude'] = lon

    # Drop helper column
    df = df.drop(columns=['_district_normalized'])

    # Count how many were matched
    matched = df['Latitude'].notna().sum()
    total = len(df)
    unmatched = df[df['Latitude'].isna()][district_col].value_counts().head(10)
    
    print(f"  Đã gán Lat/Lon: {matched}/{total} ({matched/total*100:.1f}%)")
    if len(unmatched) > 0:
        print(f"  Top 10 districts chưa matched:")
        for d, c in unmatched.items():
            print(f"    - {d}: {c} rows")

    return df


def save_location_data(path, districts=DISTRICT_CENTROIDS, metro=METRO_STATIONS,
                        hospitals=HOSPITALS, schools=SCHOOLS, markets=MARKETS):
    """Lưu location data ra JSON để reuse."""
    data = {
        # Chỉ lưu các district names CHÍNH THỨC (đã normalize)
        "district_centroids": {k: {"lat": v[0], "lon": v[1]} for k, v in districts.items()},
        "metro_stations": METRO_STATIONS,
        "hospitals": HOSPITALS,
        "schools": SCHOOLS,
        "markets": MARKETS,
        "cbd": CBD_COORDINATES,
        # Metadata
        "_note": "Chỉ dùng tên quận/huyện chuẩn. Các biến thể phải được normalize trước khi tra cứu."
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  → Location data saved to {path}")


def load_location_data(path):
    """Load location data từ JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test với sample data
    print("=" * 60)
    print("  LOCATION FEATURES MODULE - Test")
    print("=" * 60)

    # Test haversine
    dist = haversine_distance(10.7758, 106.7004, 10.8060, 106.7310)
    print(f"\n  Test khoảng cách Bến Thành → SC Vivo City: {dist:.2f} km")

    # Test với sample DataFrame
    test_df = pd.DataFrame({
        'District': ['Quận 7', 'Quận 1', 'Gò Vấp', 'Hoàn Kiếm'],
        'Latitude': [10.7419, 10.7758, 10.8396, 21.0285],
        'Longitude': [106.7274, 106.7004, 106.6661, 105.8521]
    })

    print("\n  Test DataFrame:")
    print(test_df)

    result = generate_location_features(test_df)
    print("\n  Features tạo ra:")
    print(result[['District', 'distance_to_cbd', 'near_metro', 'near_hospital']])
