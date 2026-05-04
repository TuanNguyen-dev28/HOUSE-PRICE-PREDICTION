"""
Shared utilities for address parsing.
Centralized location extraction logic to avoid duplication across files.
"""
import pandas as pd


def extract_district_city(address):
    """
    Trích xuất Quận/Huyện và Thành phố từ địa chỉ đầy đủ.
    Định dạng địa chỉ: "... , Quận/Huyện, Thành phố"
    
    Args:
        address: Chuỗi địa chỉ đầy đủ
        
    Returns:
        tuple: (district, city)
    """
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


def add_district_city_columns(df, address_column='Address'):
    """
    Thêm cột District và City vào DataFrame từ cột địa chỉ.
    
    Args:
        df: DataFrame chứa cột địa chỉ
        address_column: Tên cột chứa địa chỉ
        
    Returns:
        DataFrame với 2 cột mới: District, City
    """
    df = df.copy()
    
    if address_column not in df.columns:
        df['District'] = 'Unknown'
        df['City'] = 'Unknown'
        return df
    
    df[address_column] = df[address_column].fillna('Unknown')
    
    # Apply extraction to each address
    extracted = df[address_column].apply(extract_district_city)
    df['District'] = extracted.apply(lambda x: x[0])
    df['City'] = extracted.apply(lambda x: x[1])
    
    return df


def normalize_location_name(name):
    """
    Chuẩn hóa tên địa điểm (loại bỏ tiền tố/suffix không nhất quán).
    
    Args:
        name: Tên quận/huyện hoặc thành phố
        
    Returns:
        str: Tên đã chuẩn hóa
    """
    if not name or not isinstance(name, str):
        return 'Unknown'
    
    name = name.strip()
    
    # Xóa các suffix không cần thiết
    suffixes_to_remove = ['.', ' TP.', ' TP ', 'TX.', 'TX ']
    for suffix in suffixes_to_remove:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    
    # Chuẩn hóa các tên phổ biến
    normalizations = {
        'HCM': 'Hồ Chí Minh',
        'TPHCM': 'Hồ Chí Minh',
        'TpHCM': 'Hồ Chí Minh',
        'HCM': 'Hồ Chí Minh',
        'HN': 'Hà Nội',
        'Ha Noi': 'Hà Nội',
        'Ho Chi Minh': 'Hồ Chí Minh',
        'Ho Chi Minh city': 'Hồ Chí Minh',
        'Hồ Chí Minh city': 'Hồ Chí Minh',
        'Tp. Ho Chi Minh': 'Hồ Chí Minh',
    }
    
    for old, new in normalizations.items():
        if name.lower() == old.lower():
            return new
    
    return name
