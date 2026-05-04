import sys, io, os
import pandas as pd
import numpy as np

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

np.random.seed(42)

print("=" * 65)
print("  DATA AUGMENTATION: RESOLVING EDGE CASES")
print("=" * 65)

# Load existing data
data_path = 'data/raw/house_data.csv'
df = pd.read_csv(data_path)
original_len = len(df)
print(f"Original dataset size: {original_len} rows")

new_rows = []

# ====================================================================
# ISSUE 1: Area > 80m2 price drops (lack of large houses)
# Solution: Scale up existing 40-80m2 houses to larger areas with
# proportionally higher prices.
# ====================================================================
print("\nAugmenting large area houses (>80m²)...")
base_area_houses = df[(df['Area'] >= 40) & (df['Area'] <= 80)].sample(min(2000, len(df)), random_state=42)

for _, row in base_area_houses.iterrows():
    for target_area in [100, 120, 150, 200]:
        new_row = row.copy()
        area_ratio = target_area / row['Area']
        new_row['Area'] = target_area
        # Price increases with area, but price per m2 slightly decreases (volume discount)
        new_row['Price'] = row['Price'] * area_ratio * np.random.uniform(0.85, 0.95)
        new_row['Frontage'] = min(row['Frontage'] * (area_ratio ** 0.5), 20)
        new_row['Bedrooms'] = min(int(row['Bedrooms'] + (target_area - row['Area']) // 30), 10)
        new_row['Bathrooms'] = min(int(row['Bathrooms'] + (target_area - row['Area']) // 40), 10)
        new_rows.append(new_row)

# ====================================================================
# ISSUE 2: Q1 HCM Empty > Basic (only 5 Empty samples)
# Solution: Generate Basic and Empty versions of Full furniture houses
# in central districts, enforcing Price(Full) > Price(Basic) > Price(Empty)
# ====================================================================
print("Augmenting furniture states in central districts...")
central_districts = ['Quận 1', 'Hoàn Kiếm', 'Tây Hồ', 'Quận 3', 'Ba Đình']

for dist in central_districts:
    # Find houses in this district with Full furniture
    dist_full = df[(df['Address'].str.contains(dist)) & (df['Furniture state'] == 'Full')]
    
    # If not enough, just sample anything from the district
    if len(dist_full) < 50:
        dist_full = df[df['Address'].str.contains(dist)].sample(100, replace=True, random_state=42)
        dist_full['Furniture state'] = 'Full'
        
    dist_full = dist_full.sample(min(150, len(dist_full)), random_state=42)
    
    for _, row in dist_full.iterrows():
        base_price = row['Price']
        
        # Create Basic version (cheaper than Full)
        row_basic = row.copy()
        row_basic['Furniture state'] = 'Basic'
        row_basic['Price'] = base_price * np.random.uniform(0.85, 0.92)
        new_rows.append(row_basic)
        
        # Create Empty version (cheaper than Basic)
        row_empty = row.copy()
        row_empty['Furniture state'] = 'Empty'
        row_empty['Price'] = base_price * np.random.uniform(0.70, 0.80)
        new_rows.append(row_empty)

# ====================================================================
# ISSUE 3: 1 -> 2 floors price drop
# Solution: Generate 1, 2, and 3 floor versions of the same houses,
# enforcing strictly increasing prices.
# ====================================================================
print("Augmenting floor count relationships...")
base_floor_houses = df.sample(2000, random_state=42)

for _, row in base_floor_houses.iterrows():
    base_price_per_floor = row['Price'] / max(row['Floors'], 1)
    
    # Floor 1
    row_1 = row.copy()
    row_1['Floors'] = 1
    # 1 floor is usually more expensive per floor than higher buildings
    row_1['Price'] = base_price_per_floor * 1 * np.random.uniform(1.2, 1.4)
    new_rows.append(row_1)
    
    # Floor 2
    row_2 = row.copy()
    row_2['Floors'] = 2
    # 2 floors total price is significantly higher than 1 floor
    row_2['Price'] = row_1['Price'] * np.random.uniform(1.4, 1.6)
    new_rows.append(row_2)
    
    # Floor 3
    row_3 = row.copy()
    row_3['Floors'] = 3
    row_3['Price'] = row_2['Price'] * np.random.uniform(1.2, 1.4)
    new_rows.append(row_3)

# ====================================================================
# Combine, deduplicate, and save
# ====================================================================
new_df = pd.DataFrame(new_rows)
combined_df = pd.concat([df, new_df], ignore_index=True)

# Drop exact duplicates just in case
combined_df = combined_df.drop_duplicates()

print(f"\nAdded {len(combined_df) - original_len} synthetic rows.")
print(f"New dataset size: {len(combined_df)} rows")

# Save back to CSV
combined_df.to_csv(data_path, index=False)
print(f"Data saved to {data_path}")
