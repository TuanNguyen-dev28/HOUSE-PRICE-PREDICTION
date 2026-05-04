import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

HCM_CENTRAL = {
    "Quận 1": ["Nguyễn Huệ", "Đồng Khởi", "Lê Lợi"],
    "Quận 3": ["Nguyễn Đình Chiểu", "Điện Biên Phủ", "Lê Văn Sỹ"],
    "Quận 4": ["Đại lộ Võ Văn Kiệt", "Hoàng Diệu", "Tôn Đản"],
    "Quận 5": ["Nguyễn Trãi", "Trần Hưng Đạo", "An Dương Vương"],
    "Quận 10": ["3/2", "Nguyễn Kim", "Lý Thái Tổ"],
    "Phú Nhuận": ["Phan Xích Long", "Nguyễn Trọng Tuyển"],
    "Bình Thạnh": ["Điện Biên Phủ", "Phan Văn Trị", "Võ Tánh"],
    "Tân Bình": ["Cộng Hòa", "Trường Chinh", "Hoàng Văn Thụ"],
    "Tân Phú": ["Lê Trọn", "Tân Hương"],
}

HANOI_CENTRAL = {
    "Ba Đình": ["Điện Biên Phủ", "Hoàng Hoa Thám", "Ngọc Hà"],
    "Hoàn Kiếm": ["Hàng Bài", "Hàng Đào", "Lê Thái Tổ"],
    "Hai Bà Trưng": ["Minh Khai", "Trần Khát Chân", "Bạch Đằng"],
    "Đống Đa": ["Thái Hà", "Tôn Đức Thắng", "Huỳnh Thúc Kháng"],
    "Cầu Giấy": ["Cầu Giấy", "Xuân Thủy", "Dịch Vọng"],
    "Thanh Xuân": ["Nguyễn Trãi", "Thanh Xuân", "Trần Duy Hưng"],
    "Tây Hồ": ["Lạc Long Quân", "An Dương Vương", "Âu Cơ"],
}

DIRECTIONS = ["Đông", "Tây", "Nam", "Bắc", "Đông - Bắc", "Tây - Bắc", "Đông - Nam", "Tây - Nam"]
LEGAL = ["Have certificate", "Sale contract", "In progress", "Pending"]
FURNITURE = ["Full", "Basic", "Empty", None]

def gen_hcm():
    d = random.choice(list(HCM_CENTRAL.keys()))
    s = random.choice(HCM_CENTRAL[d])
    n = random.randint(1, 200)
    addr = f"{n}, Đường {s}, Phường 1, {d}, Hồ Chí Minh"
    price = round(random.uniform(20, 150), 2)
    return {
        "Address": addr, "Area": round(random.uniform(30, 120), 1),
        "Frontage": round(random.uniform(3, 8), 1) if random.random() < 0.6 else None,
        "Access Road": round(random.uniform(2, 10), 1) if random.random() < 0.5 else None,
        "House direction": random.choice(DIRECTIONS) if random.random() < 0.7 else None,
        "Balcony direction": random.choice(DIRECTIONS) if random.random() < 0.6 else None,
        "Floors": random.randint(3, 6), "Bedrooms": random.randint(2, 6) if random.random() < 0.7 else None,
        "Bathrooms": random.randint(2, 5) if random.random() < 0.6 else None,
        "Legal status": random.choice(LEGAL) if random.random() < 0.8 else None,
        "Furniture state": random.choice(FURNITURE) if random.random() < 0.6 else None,
        "Price": price
    }

def gen_hanoi():
    d = random.choice(list(HANOI_CENTRAL.keys()))
    s = random.choice(HANOI_CENTRAL[d])
    n = random.randint(1, 200)
    addr = f"{n}, Đường {s}, Phường 1, {d}, Hà Nội"
    price = round(random.uniform(20, 200), 2)
    return {
        "Address": addr, "Area": round(random.uniform(35, 100), 1),
        "Frontage": round(random.uniform(3, 6), 1) if random.random() < 0.6 else None,
        "Access Road": round(random.uniform(2, 8), 1) if random.random() < 0.5 else None,
        "House direction": random.choice(DIRECTIONS) if random.random() < 0.7 else None,
        "Balcony direction": random.choice(DIRECTIONS) if random.random() < 0.6 else None,
        "Floors": random.randint(4, 7), "Bedrooms": random.randint(3, 7) if random.random() < 0.7 else None,
        "Bathrooms": random.randint(2, 5) if random.random() < 0.6 else None,
        "Legal status": random.choice(LEGAL) if random.random() < 0.8 else None,
        "Furniture state": random.choice(FURNITURE) if random.random() < 0.6 else None,
        "Price": price
    }

# Load existing and backup
existing = pd.read_csv("data/raw/house_data.csv")
existing.to_csv("data/raw/house_data_backup.csv", index=False)

# Generate new data
new_data = [gen_hcm() for _ in range(1000)] + [gen_hanoi() for _ in range(500)]
new_df = pd.DataFrame(new_data)
new_df = new_df[new_df['Price'] > 20]

# Merge
combined = pd.concat([existing, new_df], ignore_index=True)
combined.to_csv("data/raw/house_data.csv", index=False)

print(f"Done! Total records: {len(combined)}")
print(f"New records added: {len(new_df)}")
print(f"Price range: {combined['Price'].min():.2f} - {combined['Price'].max():.2f}")
print("\nSample new addresses:")
for a in new_df['Address'].head(3): print(f"  {a}")
