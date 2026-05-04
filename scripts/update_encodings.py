"""
Update location encodings from cleaned data
"""
import pandas as pd
import json

# Load cleaned data
df = pd.read_csv("data/raw/house_data.csv")

# Extract district and city
df['District'] = df['Address'].str.split(',').str[-2].str.strip()
df['City'] = df['Address'].str.split(',').str[-1].str.strip()

# Calculate encodings
district_encoding = df.groupby('District')['Price'].mean().to_dict()
city_encoding = df.groupby('City')['Price'].mean().to_dict()
global_mean = df['Price'].mean()

# Save
encodings = {
    "district_encoding": district_encoding,
    "city_encoding": city_encoding,
    "district_global_mean": global_mean,
    "city_global_mean": global_mean
}

with open("models/location_encodings.json", "w", encoding="utf-8") as f:
    json.dump(encodings, f, ensure_ascii=False, indent=2)

print("Location encodings updated!")
print(f"Districts: {len(district_encoding)}")
print(f"Cities: {len(city_encoding)}")
print(f"Global mean: {global_mean:.2f}")

# Show top districts
print("\nTop 10 districts by avg price:")
for d, v in sorted(district_encoding.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {d}: {v:.2f}")
