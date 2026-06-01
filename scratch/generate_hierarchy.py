import os
import sys
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocess import extract_location_features, normalize_district_name

def main():
    raw_path = os.path.join(BASE_DIR, 'data', 'raw', 'house_data_hcm.csv')
    hierarchy_path = os.path.join(BASE_DIR, 'static', 'location_hierarchy.json')
    
    print(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    print("Extracting location features...")
    df = extract_location_features(df)
    df['District'] = df['District'].apply(normalize_district_name)
    
    print("Building District -> Ward -> Street hierarchy...")
    hierarchy = {}
    
    # Group by District
    for district, dist_grp in df.groupby('District'):
        if district == 'Unknown' or len(district.strip()) < 2:
            continue
        hierarchy[district] = {}
        
        # Group by Ward
        for ward, ward_grp in dist_grp.groupby('Ward'):
            if ward == 'Unknown' or len(ward.strip()) < 2:
                continue
            
            # Get unique sorted streets for this ward
            streets = sorted([
                s for s in ward_grp['Street'].dropna().unique() 
                if s != 'Unknown' and len(s.strip()) > 2
            ])
            
            if streets:
                hierarchy[district][ward] = streets
                
    # Sort districts alphabetically, but Quận 1, 2, 3... first
    def district_sort_key(x):
        if x.startswith('Quận '):
            parts = x.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return (0, int(parts[1]), x)
        elif x == 'Thành phố Thủ Đức':
            return (1, 0, x)
        return (2, 0, x)
        
    sorted_districts = sorted(hierarchy.keys(), key=district_sort_key)
    sorted_hierarchy = {}
    for d in sorted_districts:
        # Sort wards alphabetically, but Phường 1, 2, 3... first
        def ward_sort_key(w):
            w_clean = w.replace('Phường ', '').strip()
            if w_clean.isdigit():
                return (0, int(w_clean), w)
            return (1, 0, w)
        sorted_wards = sorted(hierarchy[d].keys(), key=ward_sort_key)
        sorted_hierarchy[d] = {}
        for w in sorted_wards:
            sorted_hierarchy[d][w] = hierarchy[d][w]
            
    print(f"Saving hierarchy JSON to {hierarchy_path}...")
    os.makedirs(os.path.dirname(hierarchy_path), exist_ok=True)
    with open(hierarchy_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_hierarchy, f, ensure_ascii=False, indent=2)
        
    print("✅ Location hierarchy generated successfully!")
    print(f"  Total Districts: {len(sorted_hierarchy)}")
    total_wards = sum([len(sorted_hierarchy[d]) for d in sorted_hierarchy])
    print(f"  Total Wards: {total_wards}")

if __name__ == '__main__':
    main()
