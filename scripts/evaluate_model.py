"""
Comprehensive test: Legal Status & Furniture State ordering
across MULTIPLE locations and property sizes.
"""
import sys, io, os
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import EnsemblePredictor

predictor = EnsemblePredictor(
    xgboost_model_path='models/xgboost_model.pkl',
    random_forest_model_path='models/random_forest_model.pkl',
    location_encodings_path='models/location_encodings.json',
)

# ====================================================================
# TEST 1: Legal Status ordering across different locations
# ====================================================================
print("=" * 75)
print("  TEST 1: PHÁP LÝ — Cùng nhà, khác pháp lý (nhiều vị trí)")
print("=" * 75)

locations = [
    ("Quận 1, Hồ Chí Minh", "Trung tâm HCM"),
    ("Quận Cầu Giấy, Hà Nội", "Trung tâm HN"),
    ("Gò Vấp, Hồ Chí Minh", "Ngoại ô HCM"),
    ("Hà Đông, Hà Nội", "Ngoại ô HN"),
    ("Hải Châu, Đà Nẵng", "Đà Nẵng"),
]

legal_statuses = ['Have certificate', 'Sale contract', 'In progress', 'Pending']
legal_pass = 0
legal_total = 0

for addr_suffix, label in locations:
    base = {
        'area': 80, 'frontage': 5, 'access_road': 6,
        'floors': 3, 'bedrooms': 3, 'bathrooms': 2,
        'house_direction': 'Đông - Nam',
        'balcony_direction': 'Nam',
        'furniture_state': 'Full',
        'address': f'123 Đường ABC, {addr_suffix}',
    }
    
    prices = []
    print(f"\n  📍 {label} ({addr_suffix}):")
    for legal in legal_statuses:
        test_data = {**base, 'legal_status': legal}
        result = predictor.predict(test_data)
        price = result['ensemble']['price_billion_vnd']
        prices.append(price)
        print(f"     {legal:<20} → {price:.2f} tỷ")
    
    legal_total += 1
    if prices[0] >= prices[1] >= prices[2] >= prices[3]:
        print(f"     ✅ Đúng thứ tự! Có sổ({prices[0]:.2f}) ≥ HĐ({prices[1]:.2f}) ≥ Đang làm({prices[2]:.2f}) ≥ Chờ({prices[3]:.2f})")
        legal_pass += 1
    else:
        print(f"     ❌ SAI thứ tự!")

# ====================================================================
# TEST 2: Furniture State ordering across different locations
# ====================================================================
print("\n\n" + "=" * 75)
print("  TEST 2: NỘI THẤT — Cùng nhà, khác nội thất (nhiều vị trí)")
print("=" * 75)

furniture_states = ['Full', 'Basic', 'Empty']
furn_pass = 0
furn_total = 0

for addr_suffix, label in locations:
    base = {
        'area': 80, 'frontage': 5, 'access_road': 6,
        'floors': 3, 'bedrooms': 3, 'bathrooms': 2,
        'house_direction': 'Đông - Nam',
        'balcony_direction': 'Nam',
        'legal_status': 'Have certificate',
        'address': f'123 Đường ABC, {addr_suffix}',
    }
    
    prices = []
    print(f"\n  📍 {label} ({addr_suffix}):")
    for furn in furniture_states:
        test_data = {**base, 'furniture_state': furn}
        result = predictor.predict(test_data)
        price = result['ensemble']['price_billion_vnd']
        prices.append(price)
        print(f"     {furn:<10} → {price:.2f} tỷ")
    
    furn_total += 1
    if prices[0] >= prices[1] >= prices[2]:
        print(f"     ✅ Đúng! Full({prices[0]:.2f}) ≥ Basic({prices[1]:.2f}) ≥ Empty({prices[2]:.2f})")
        furn_pass += 1
    else:
        print(f"     ❌ SAI thứ tự!")

# ====================================================================
# TEST 3: Legal + Furniture combined — giá chênh lệch hợp lý?
# ====================================================================
print("\n\n" + "=" * 75)
print("  TEST 3: CHÊNH LỆCH GIÁ — Có sổ + Full vs Chờ xử lý + Empty")
print("=" * 75)

for addr_suffix, label in locations:
    base = {
        'area': 80, 'frontage': 5, 'access_road': 6,
        'floors': 3, 'bedrooms': 3, 'bathrooms': 2,
        'house_direction': 'Đông - Nam',
        'balcony_direction': 'Nam',
        'address': f'123 Đường ABC, {addr_suffix}',
    }
    
    # Best case: certificate + full furniture
    best = {**base, 'legal_status': 'Have certificate', 'furniture_state': 'Full'}
    # Worst case: pending + empty
    worst = {**base, 'legal_status': 'Pending', 'furniture_state': 'Empty'}
    
    best_price = predictor.predict(best)['ensemble']['price_billion_vnd']
    worst_price = predictor.predict(worst)['ensemble']['price_billion_vnd']
    diff = best_price - worst_price
    pct = (diff / worst_price * 100) if worst_price > 0 else 0
    
    status = "✅" if best_price > worst_price else "❌"
    print(f"  {status} {label:20s}: Tốt nhất={best_price:.2f} tỷ, Tệ nhất={worst_price:.2f} tỷ, Chênh={diff:+.2f} tỷ ({pct:+.1f}%)")

# ====================================================================
# TEST 4: Diện tích lớn hơn → giá cao hơn?
# ====================================================================
print("\n\n" + "=" * 75)
print("  TEST 4: DIỆN TÍCH — Cùng vị trí, diện tích tăng → giá tăng?")
print("=" * 75)

areas = [30, 50, 80, 100, 150, 200]
base = {
    'frontage': 5, 'access_road': 6,
    'floors': 3, 'bedrooms': 3, 'bathrooms': 2,
    'house_direction': 'Đông - Nam', 'balcony_direction': 'Nam',
    'legal_status': 'Have certificate', 'furniture_state': 'Full',
    'address': '123 ABC, Quận Cầu Giấy, Hà Nội',
}

prev_price = 0
area_pass = True
for area in areas:
    test_data = {**base, 'area': area}
    price = predictor.predict(test_data)['ensemble']['price_billion_vnd']
    status = "✅" if price >= prev_price else "❌"
    if price < prev_price:
        area_pass = False
    print(f"  {status} Diện tích {area:>5} m²  → {price:.2f} tỷ")
    prev_price = price

# ====================================================================
# TEST 5: Số tầng nhiều hơn → giá cao hơn?
# ====================================================================
print("\n\n" + "=" * 75)
print("  TEST 5: SỐ TẦNG — Cùng vị trí, tầng tăng → giá tăng?")
print("=" * 75)

floors_list = [1, 2, 3, 4, 5]
base = {
    'area': 80, 'frontage': 5, 'access_road': 6,
    'bedrooms': 3, 'bathrooms': 2,
    'house_direction': 'Đông - Nam', 'balcony_direction': 'Nam',
    'legal_status': 'Have certificate', 'furniture_state': 'Full',
    'address': '123 ABC, Quận Cầu Giấy, Hà Nội',
}

prev_price = 0
for fl in floors_list:
    test_data = {**base, 'floors': fl}
    price = predictor.predict(test_data)['ensemble']['price_billion_vnd']
    status = "✅" if price >= prev_price else "⚠️"
    print(f"  {status} {fl} tầng  → {price:.2f} tỷ")
    prev_price = price

# ====================================================================
# TEST 6: Quận trung tâm đắt hơn ngoại ô?
# ====================================================================
print("\n\n" + "=" * 75)
print("  TEST 6: VỊ TRÍ — Quận trung tâm đắt hơn ngoại ô?")
print("=" * 75)

base = {
    'area': 80, 'frontage': 5, 'access_road': 6,
    'floors': 3, 'bedrooms': 3, 'bathrooms': 2,
    'house_direction': 'Đông - Nam', 'balcony_direction': 'Nam',
    'legal_status': 'Have certificate', 'furniture_state': 'Full',
}

location_pairs = [
    ("Quận 1, Hồ Chí Minh", "Gò Vấp, Hồ Chí Minh", "Q1 vs Gò Vấp (HCM)"),
    ("Ba Đình, Hà Nội", "Hà Đông, Hà Nội", "Ba Đình vs Hà Đông (HN)"),
    ("Hoàn Kiếm, Hà Nội", "Quận Long Biên, Hà Nội", "Hoàn Kiếm vs Long Biên (HN)"),
]

for center_addr, suburb_addr, label in location_pairs:
    center = {**base, 'address': f'123 ABC, {center_addr}'}
    suburb = {**base, 'address': f'123 ABC, {suburb_addr}'}
    
    center_price = predictor.predict(center)['ensemble']['price_billion_vnd']
    suburb_price = predictor.predict(suburb)['ensemble']['price_billion_vnd']
    
    status = "✅" if center_price > suburb_price else "❌"
    print(f"  {status} {label}: Trung tâm={center_price:.2f} tỷ vs Ngoại ô={suburb_price:.2f} tỷ")

# ====================================================================
# SUMMARY
# ====================================================================
print("\n\n" + "=" * 75)
print("  📊 TÓM TẮT KẾT QUẢ TEST")
print("=" * 75)
print(f"  Pháp lý (Legal Status):    {legal_pass}/{legal_total} vị trí đúng thứ tự")
print(f"  Nội thất (Furniture):       {furn_pass}/{furn_total} vị trí đúng thứ tự")
print(f"  Diện tích tăng → giá tăng: {'✅ Đúng' if area_pass else '❌ Sai'}")
print("=" * 75)
