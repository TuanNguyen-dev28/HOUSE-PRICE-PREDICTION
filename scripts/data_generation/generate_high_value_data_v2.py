"""
Generate high-value house data with CORRECT address format matching existing data.
Format: "Đường [street], Phường [ward], [district], [city]"
"""

import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# Real street names for central districts (matching existing data format)
HCM_CENTRAL_STREETS = {
    "Quận 1": ["Nguyễn Huệ", "Đồng Khởi", "Lê Lợi", "Hai Bà Trưng", "Pasteur", "Nguyễn Trãi", "Đề Thám", "Trần Hưng Đạo", "Võ Văn Kiệt", "Lê Thánh Tôn", "Nguyễn Thị Minh Khai", "Phạm Ngũ Lão", "Bùi Thị Xuân", "Võ Thị Sáu", "Nguyễn Bỉnh Khiêm"],
    "Quận 3": ["Nguyễn Đình Chiểu", "Điện Biên Phủ", "Lê Văn Sỹ", "Trần Quốc Toản", "Võ Văn Kiệt", "Pasteur", "Nguyễn Thượng Hiền", "Phan Kế Bính", "Đặng Trần Côn", "Bà Huyện Thanh Quan", "Huỳnh Khương An"],
    "Quận 4": ["Đại lộ Võ Văn Kiệt", "Hoàng Diệu", "Tôn Đản", "Bến Vân Đồn", "An Dương Vương", "Khánh Hội", "Vĩnh Khánh", "Đoàn Văn Bơ"],
    "Quận 5": ["Nguyễn Trãi", "Trần Hưng Đạo", "An Dương Vương", "Hải Thượng Lãn Ông", "Châu Văn Liêm", "Thuận Kiều", "Lê Hồng Phong"],
    "Quận 10": ["3/2", "Nguyễn Kim", "Lý Thái Tổ", "Cách Mạng Tháng Tám", "Tô Hiến Thành", "Thành Thái", "Trần Bình Trọng", "Hòa Hảo"],
    "Phú Nhuận": ["Phan Xích Long", "Nguyễn Trọng Tuyển", "Trần Hữu Trang", "Phan Đăng Lưu", "Hoàng Minh Giám"],
    "Bình Thạnh": ["Điện Biên Phủ", "Phan Văn Trị", "Võ Tánh", "Nguyễn Văn Đậu", "Nơ Trang Long", "Bạch Đằng", "Xô Viết Nghệ Tĩnh", "Lê Quang Định"],
    "Tân Bình": ["Cộng Hòa", "Trường Chinh", "Hoàng Văn Thụ", "Phan Đình Phùng", "Trương Công Định", "Tân Sơn Nhì"],
    "Tân Phú": ["Lê Trọn", "Tân Hương", "Đông Hưng Thuận", "Hiệp Tân", "Tây Thạnh"],
}

HANOI_CENTRAL_STREETS = {
    "Ba Đình": ["Điện Biên Phủ", "Hoàng Hoa Thám", "Ngọc Hà", "Phan Đình Phùng", "Đội Cấn", "Kim Mã", "Giảng Võ", "Láng Hạ", "Đào Tấn"],
    "Hoàn Kiếm": ["Hàng Bài", "Hàng Đào", "Lê Thái Tổ", "Tràng Tiền", "Hàng Bông", "Bà Triệu", "Lý Thường Kiệt", "Trần Hưng Đạo"],
    "Hai Bà Trưng": ["Minh Khai", "Trần Khát Chân", "Bạch Đằng", "Quang Trung", "Phố Huế", "Lò Đúc", "Đại Cồ Việt", "Kim Ngưu"],
    "Đống Đa": ["Thái Hà", "Tôn Đức Thắng", "Huỳnh Thúc Kháng", "Xã Đàn", "Phạm Ngọc Thạch", "Chùa Bộc", "Trung Liệt", "Khâm Thiên"],
    "Cầu Giấy": ["Cầu Giấy", "Xuân Thủy", "Dịch Vọng", "Trung Kính", "Yên Hòa", "Thiên Hiền", "Nguyễn Phong Sắc"],
    "Thanh Xuân": ["Nguyễn Trãi", "Thanh Xuân", "Trần Duy Hưng", "Khuất Duy Tiến", "Lê Văn Lương", "Phương Liệt"],
    "Tây Hồ": ["Lạc Long Quân", "An Dương Vương", "Âu Cơ", "Hoàng Sa", "Thụy Khuê", "Phú Thượng", "Yên Phụ", "Nhật Tân"],
}

# Ward names
HCM_WARDS = {
    "Quận 1": ["Bến Nghé", "Cầu Kho", "Cầu Ông Lãnh", "Đa Kao", "Nguyễn Cư Trinh", "Nguyễn Thái Bình", "Phạm Hồng Thái", "Tân Định"],
    "Quận 3": ["Phường 1", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Võ Thị Sáu", "Phường 9", "Phường 10", "Phường 11", "Phường 12"],
    "Quận 4": ["Phường 1", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 6", "Phường 8", "Phường 9", "Phường 10", "Phường 13", "Phường 14", "Phường 15", "Phường 16", "Phường 17", "Phường 18"],
    "Quận 5": ["Phường 1", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 6", "Phường 7", "Phường 8", "Phường 9", "Phường 10", "Phường 11", "Phường 12", "Phường 13", "Phường 14"],
    "Quận 10": ["Phường 1", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 6", "Phường 7", "Phường 8", "Phường 9", "Phường 10", "Phường 11", "Phường 12", "Phường 13", "Phường 14", "Phường 15"],
    "Phú Nhuận": ["Phường 1", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 7", "Phường 8", "Phường 9", "Phường 10", "Phường 11", "Phường 12", "Phường 13", "Phường 15", "Phường 17"],
    "Bình Thạnh": ["Phường 1", "Phường 2", "Phường 3", "Phường 5", "Phường 6", "Phường 7", "Phường 11", "Phường 12", "Phường 13", "Phường 14", "Phường 15", "Phường 17", "Phường 19", "Phường 21", "Phường 22", "Phường 24", "Phường 25", "Phường 26", "Phường 27", "Phường 28"],
    "Tân Bình": ["Phường 1", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 6", "Phường 7", "Phường 8", "Phường 9", "Phường 10", "Phường 11", "Phường 12", "Phường 13", "Phường 14", "Phường 15"],
    "Tân Phú": ["Phường 1", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 6", "Phường 7", "Phường 8", "Phường 9", "Phường 10", "Phường 11"],
}

HANOI_WARDS = {
    "Ba Đình": ["Phú Mỹ", "Điện Biên", "Đội Cấn", "Giảng Võ", "Kim Mã", "Liễu Giai", "Ngọc Hà", "Ngọc Khánh", "Phúc Xá", "Quán Thánh", "Thành Công", "Trúc Bạch"],
    "Hoàn Kiếm": ["Chương Dương", "Cửa Đông", "Cửa Nam", "Đồng Xuân", "Hàng Bạc", "Hàng Bài", "Hàng Bông", "Hàng Đào", "Hàng Gai", "Lý Thái Tổ", "Phan Chu Trinh", "Phúc Tân", "Trần Hưng Đạo", "Tràng Tiền"],
    "Hai Bà Trưng": ["Bạch Đằng", "Bách Khoa", "Đống Mác", "Đồng Nhân", "Đồng Tâm", "Lê Đại Hành", "Minh Khai", "Ngô Thì Nhậm", "Nguyễn Du", "Phạm Đình Hổ", "Phố Huế", "Quỳnh Lôi", "Thanh Lương", "Thanh Nhàn", "Trần Khát Chân", "Trương Định", "Văn Miếu"],
    "Đống Đa": ["Cát Linh", "Hàng Bột", "Khâm Thiên", "Láng Hạ", "Láng Thượng", "Nam Đồng", "Ngã Tư Sở", "Ô Chợ Dừa", "Phường Chương Dương", "Thịnh Quang", "Thổ Quan", "Trung Phụng", "Văn Chương", "Văn Miếu"],
    "Cầu Giấy": ["Dịch Vọng", "Dịch Vọng Hậu", "Nghĩa Đô", "Nghĩa Tân", "Quan Hoa", "Trung Hòa", "Yên Hòa"],
    "Thanh Xuân": ["Hạ Đình", "Khương Đình", "Khương Hạ", "Khương Trung", "Kim Giang", "Nhân Chính", "Phương Liệt", "Thanh Xuân Bắc", "Thanh Xuân Nam", "Thanh Xuân Trung"],
    "Tây Hồ": ["Bưởi", "Nhật Tân", "Phú Thượng", "Quảng An", "Thụy Khuê", "Tứ Liên", "Xuân La", "Yên Phụ"],
}

DIRECTIONS = ["Đông", "Tây", "Nam", "Bắc", "Đông - Bắc", "Tây - Bắc", "Đông - Nam", "Tây - Nam"]
LEGAL_STATUS = ["Have certificate", "Sale contract", "In progress", "Pending"]
FURNITURE_STATE = ["Full", "Basic", "Empty", None]

def generate_hcm_house():
    """Generate a realistic high-value house in HCM City"""
    districts = list(HCM_CENTRAL_STREETS.keys())
    district = random.choice(districts)
    street = random.choice(HCM_CENTRAL_STREETS[district])
    ward = random.choice(HCM_WARDS[district])
    
    # CORRECT FORMAT: "Đường [street], Phường [ward], [district], Hồ Chí Minh"
    street_num = random.randint(1, 200)
    address = f"{street_num}, Đường {street}, {ward}, {district}, Hồ Chí Minh"
    
    # Price varies by district
    district_price_ranges = {
        "Quận 1": (35, 150),
        "Quận 3": (30, 120),
        "Quận 4": (25, 80),
        "Quận 5": (22, 60),
        "Quận 10": (22, 70),
        "Phú Nhuận": (25, 85),
        "Bình Thạnh": (22, 75),
        "Tân Bình": (20, 60),
        "Tân Phú": (18, 50),
    }
    
    price_range = district_price_ranges.get(district, (20, 60))
    price = round(random.uniform(price_range[0], price_range[1]), 2)
    
    area = round(random.uniform(30, 120), 1)
    frontage = round(random.uniform(3, 8), 1) if random.random() < 0.6 else None
    access_road = round(random.uniform(2, 10), 1) if random.random() < 0.5 else None
    house_direction = random.choice(DIRECTIONS) if random.random() < 0.7 else None
    balcony_direction = random.choice(DIRECTIONS) if random.random() < 0.6 and house_direction else None
    floors = random.randint(3, 6) if random.random() < 0.8 else random.randint(1, 3)
    bedrooms = random.randint(2, 6) if random.random() < 0.7 else None
    bathrooms = random.randint(2, 5) if random.random() < 0.6 else None
    legal_status = random.choice(LEGAL_STATUS) if random.random() < 0.8 else None
    furniture = random.choice(FURNITURE_STATE) if random.random() < 0.6 else None
    
    return {
        "Address": address,
        "Area": area,
        "Frontage": frontage,
        "Access Road": access_road,
        "House direction": house_direction,
        "Balcony direction": balcony_direction,
        "Floors": floors,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Legal status": legal_status,
        "Furniture state": furniture,
        "Price": price
    }

def generate_hanoi_house():
    """Generate a realistic high-value house in Hanoi"""
    districts = list(HANOI_CENTRAL_STREETS.keys())
    district = random.choice(districts)
    street = random.choice(HANOI_CENTRAL_STREETS[district])
    ward = random.choice(HANOI_WARDS[district])
    
    # CORRECT FORMAT: "Đường [street], Phường [ward], [district], Hà Nội"
    street_num = random.randint(1, 200)
    address = f"{street_num}, Đường {street}, {ward}, {district}, Hà Nội"
    
    # Price varies by district
    district_price_ranges = {
        "Ba Đình": (30, 150),
        "Hoàn Kiếm": (35, 180),
        "Hai Bà Trưng": (25, 80),
        "Đống Đa": (22, 70),
        "Cầu Giấy": (25, 90),
        "Thanh Xuân": (22, 65),
        "Tây Hồ": (35, 200)
    }
    
    price_range = district_price_ranges.get(district, (20, 60))
    price = round(random.uniform(price_range[0], price_range[1]), 2)
    
    area = round(random.uniform(35, 100), 1)
    frontage = round(random.uniform(3, 6), 1) if random.random() < 0.6 else None
    access_road = round(random.uniform(2, 8), 1) if random.random() < 0.5 else None
    house_direction = random.choice(DIRECTIONS) if random.random() < 0.7 else None
    balcony_direction = random.choice(DIRECTIONS) if random.random() < 0.6 and house_direction else None
    floors = random.randint(4, 7) if random.random() < 0.75 else random.randint(1, 4)
    bedrooms = random.randint(3, 7) if random.random() < 0.7 else None
    bathrooms = random.randint(2, 5) if random.random() < 0.6 else None
    legal_status = random.choice(LEGAL_STATUS) if random.random() < 0.8 else None
    furniture = random.choice(FURNITURE_STATE) if random.random() < 0.6 else None
    
    return {
        "Address": address,
        "Area": area,
        "Frontage": frontage,
        "Access Road": access_road,
        "House direction": house_direction,
        "Balcony direction": balcony_direction,
        "Floors": floors,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Legal status": legal_status,
        "Furniture state": furniture,
        "Price": price
    }

def main():
    print("=" * 60)
    print("Generating high-value house data with correct address format")
    print("=" * 60)
    
    # Generate data
    num_hcm = 1000
    num_hanoi = 500
    num_total = num_hcm + num_hanoi
    
    data = []
    
    print(f"Generating {num_hcm} records for HCM City...")
    for i in range(num_hcm):
        data.append(generate_hcm_house())
        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{num_hcm}")
    
    print(f"Generating {num_hanoi} records for Hanoi...")
    for i in range(num_hanoi):
        data.append(generate_hanoi_house())
        if (i + 1) % 250 == 0:
            print(f"  Generated {i + 1}/{num_hanoi}")
    
    new_df = pd.DataFrame(data)
    
    # Filter only > 20 ty
    new_df = new_df[new_df['Price'] > 20]
    
    # Load existing data
    print("\nLoading existing data...")
    existing_df = pd.read_csv("data/raw/house_data.csv")
    print(f"Existing records: {len(existing_df)}")
    
    # Backup original
    existing_df.to_csv("data/raw/house_data_backup.csv", index=False)
    print("Backup saved to: data/raw/house_data_backup.csv")
    
    # Merge
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_csv("data/raw/house_data.csv", index=False)
    
    print(f"\nTotal records after merge: {len(combined_df)}")
    print(f"New records added: {len(new_df)}")
    print(f"Price range: {combined_df['Price'].min():.2f} - {combined_df['Price'].max():.2f}")
    
    # Show sample
    print("\nSample new addresses:")
    for addr in new_df['Address'].head(5):
        print(f"  {addr}")

if __name__ == "__main__":
    main()
