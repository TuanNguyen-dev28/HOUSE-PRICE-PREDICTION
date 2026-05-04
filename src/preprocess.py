"""
Data Preprocessing Module - Fixed Version
Properly handles target encoding without data leakage using out-of-fold encoding.
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


def load_data(path):
    """Tải dữ liệu từ file CSV"""
    return pd.read_csv(path)


def extract_district_city(df):
    """
    Trích xuất Quận/Huyện và Thành phố từ cột Địa chỉ.
    Định dạng địa chỉ: "... , Quận/Huyện, Thành phố"
    """
    if 'Address' not in df.columns:
        return df
    
    df = df.copy()
    df['Address'] = df['Address'].fillna('Unknown')
    
    parts = df['Address'].str.split(',')
    # Thành phố là phần cuối cùng
    df['City'] = parts.str[-1].str.strip()
    
    # Quận/Huyện là phần thứ hai từ cuối
    df['District'] = parts.str[-2].str.strip()
    
    # Xử lý trường hợp split thất bại (ít hơn 2 phần)
    mask = df['District'].isna() | (df['District'] == df['Address'])
    df.loc[mask, 'District'] = 'Unknown'
    df.loc[mask, 'City'] = 'Unknown'
    
    return df


def target_encode_with_smoothing(train_df, target_col='Price', min_samples=30, smoothing_weight=10):
    """
    Target encoding với smoothing để giảm overfitting.
    Sử dụng công thức:
    encoded = (count * mean + smoothing_weight * global_mean) / (count + smoothing_weight)
    
    Args:
        train_df: DataFrame training
        target_col: Tên cột target
        min_samples: Ngưỡng tối thiểu để tính mean thực
        smoothing_weight: Trọng số smoothing
        
    Returns:
        dict: Encoding mappings
    """
    global_mean = train_df[target_col].mean()
    
    # District encoding với smoothing
    district_stats = train_df.groupby('District')[target_col].agg(['mean', 'count'])
    district_encodings = {}
    for district in district_stats.index:
        count = district_stats.loc[district, 'count']
        mean = district_stats.loc[district, 'mean']
        # Smoothing formula
        encoded = (count * mean + smoothing_weight * global_mean) / (count + smoothing_weight)
        district_encodings[district] = encoded
    
    # City encoding với smoothing
    city_stats = train_df.groupby('City')[target_col].agg(['mean', 'count'])
    city_encodings = {}
    for city in city_stats.index:
        count = city_stats.loc[city, 'count']
        mean = city_stats.loc[city, 'mean']
        encoded = (count * mean + smoothing_weight * global_mean) / (count + smoothing_weight)
        city_encodings[city] = encoded
    
    return {
        'district_encoding': district_encodings,
        'city_encoding': city_encodings,
        'district_global_mean': global_mean,
        'city_global_mean': global_mean
    }


def target_encode_with_oof(train_df, n_folds=5, target_col='Price'):
    """
    Target Encoding với Out-of-Fold để tránh data leakage.
    Mỗi fold được encode sử dụng mean từ các fold KHÁC.
    
    Args:
        train_df: DataFrame training (sẽ được thêm cột encoded)
        n_folds: Số folds cho cross-validation
        target_col: Tên cột target
        
    Returns:
        DataFrame với cột District_Encoded và City_Encoded đã được encode
        dict chứa encodings để lưu và sử dụng cho test data
    """
    df = train_df.copy()
    global_mean = df[target_col].mean()
    
    # Initialize encoded columns
    df['District_Encoded'] = global_mean
    df['City_Encoded'] = global_mean
    
    # Store out-of-fold predictions
    district_oof = np.full(len(df), global_mean)
    city_oof = np.full(len(df), global_mean)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(df):
        # Calculate means from training fold
        train_fold = df.iloc[train_idx]
        
        # District encoding from train fold
        district_means = train_fold.groupby('District')[target_col].mean()
        district_oof[val_idx] = df.iloc[val_idx]['District'].map(district_means).fillna(global_mean)
        
        # City encoding from train fold
        city_means = train_fold.groupby('City')[target_col].mean()
        city_oof[val_idx] = df.iloc[val_idx]['City'].map(city_means).fillna(global_mean)
    
    df['District_Encoded'] = district_oof
    df['City_Encoded'] = city_oof
    
    # Build encoding dict for test data (from FULL training data for best test predictions)
    # Note: For final model training, we use full data for encoding
    encodings = target_encode_with_smoothing(df, target_col)
    
    return df, encodings


def apply_target_encoding(df, encodings):
    """
    Áp dụng target encoding từ dictionary encodings.
    Dùng cho test data hoặc inference.
    
    Args:
        df: DataFrame cần encode
        encodings: Dict chứa encoding mappings
        
    Returns:
        DataFrame với cột District_Encoded và City_Encoded
    """
    df = df.copy()
    
    district_enc = encodings['district_encoding']
    city_enc = encodings['city_encoding']
    district_global = encodings['district_global_mean']
    city_global = encodings['city_global_mean']
    
    df['District_Encoded'] = df['District'].map(district_enc).fillna(district_global)
    df['City_Encoded'] = df['City'].map(city_enc).fillna(city_global)
    
    return df


def preprocess_data(df, apply_target_encoding_flag=True, encodings=None):
    """
    Tiền xử lý dữ liệu với mã hóa cho các cột phân loại.
    
    Args:
        df: DataFrame raw data
        apply_target_encoding_flag: Nếu True, áp dụng OOF target encoding
        encodings: Dict encodings (nếu có, dùng cho test data)
        
    Returns:
        DataFrame đã được preprocess
        dict encodings (nếu apply_target_encoding_flag=True, để lưu)
    """
    # Tạo bản sao để tránh sửa đổi dữ liệu gốc
    df = df.copy()
    
    # Xóa các hàng có Price bị thiếu
    df = df.dropna(subset=['Price'])
    
    # Trích xuất Quận/Huyện và Thành phố từ Địa chỉ
    df = extract_district_city(df)
    
    # One-hot encode cho hướng nhà và ban công (không có thứ tự)
    direction_cols = ['House direction', 'Balcony direction']
    for col in direction_cols:
        df[col] = df[col].fillna('Unknown')
        df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=False)
    
    # Ordinal encode cho Pháp lý (có thứ tự rõ ràng theo thực tế thị trường)
    # Có sổ → giá cao nhất, Đang chờ → giá thấp nhất
    legal_order = {
        'Have certificate': 3,  # Có sổ đỏ/sổ hồng → giá cao nhất
        'Sale contract': 2,     # Hợp đồng mua bán
        'In progress': 1,       # Đang làm sổ
        'Pending': 0,           # Chưa có giấy tờ → giá thấp nhất
    }
    df['Legal status'] = df['Legal status'].fillna('Pending')
    df['Legal_status_ordinal'] = df['Legal status'].map(legal_order).fillna(0).astype(int)
    df = df.drop(columns=['Legal status'])
    
    # Ordinal encode cho Nội thất (có thứ tự rõ ràng)
    # Nội thất đầy đủ → giá cao hơn
    furniture_order = {
        'Full': 2,    # Nội thất đầy đủ → giá cao nhất
        'Basic': 1,   # Nội thất cơ bản
        'Empty': 0,   # Không nội thất → giá thấp nhất
    }
    df['Furniture state'] = df['Furniture state'].fillna('Empty')
    df['Furniture_state_ordinal'] = df['Furniture state'].map(furniture_order).fillna(0).astype(int)
    df = df.drop(columns=['Furniture state'])
    
    # Áp dụng target encoding cho các features vị trí
    if apply_target_encoding_flag:
        if encodings is None:
            # Training mode: compute OOF encodings
            df, encodings = target_encode_with_oof(df)
        else:
            # Inference mode: use provided encodings
            df = apply_target_encoding(df, encodings)
    
    # Xóa các cột text gốc Address, District, City (giữ lại phiên bản encoded)
    cols_to_drop = ['Address', 'District', 'City']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Điền các giá trị thiếu còn lại bằng median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df, encodings


def save_encodings(encodings, path):
    """
    Lưu encodings ra file JSON.
    Keys được đặt theo format mà predict.py sử dụng.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(encodings, f, ensure_ascii=False, indent=2)


def load_encodings(path):
    """Tải encodings từ file JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    # Tải dữ liệu thô
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "data", "raw", "house_data.csv")
    processed_path = os.path.join(base_dir, "data", "processed", "house_processed.csv")
    encodings_path = os.path.join(base_dir, "models", "location_encodings.json")
    
    print("=" * 60)
    print("  PREPROCESSING PIPELINE (Fixed - No Data Leakage)")
    print("=" * 60)
    
    print("\nĐang tải dữ liệu thô...")
    df = load_data(raw_path)
    print(f"  Kích thước dữ liệu gốc: {df.shape}")
    print(f"  Các cột: {df.columns.tolist()}")
    
    print("\nĐang tiền xử lý dữ liệu...")
    df_processed, encodings = preprocess_data(df, apply_target_encoding_flag=True)
    print(f"  Kích thước dữ liệu đã xử lý: {df_processed.shape}")
    print(f"  Các cột đã mã hóa: {df_processed.columns.tolist()}")
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    os.makedirs(os.path.dirname(encodings_path), exist_ok=True)
    
    # Lưu dữ liệu đã xử lý
    df_processed.to_csv(processed_path, index=False)
    print(f"\n  Dữ liệu đã xử lý được lưu tại: {processed_path}")
    
    # Lưu encodings với keys đúng format
    save_encodings(encodings, encodings_path)
    print(f"  Encodings được lưu tại: {encodings_path}")
    
    # In thông tin encodings
    print(f"\n  Số lượng district encodings: {len(encodings['district_encoding'])}")
    print(f"  Số lượng city encodings: {len(encodings['city_encoding'])}")
    print(f"  Global mean: {encodings['district_global_mean']:.4f}")
    
    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
