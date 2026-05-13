"""
Geo Intelligence Module - Advanced Geographic Feature Engineering
Bổ sung geo features không phụ thuộc vào target encoding.

Key Features:
1. KMeans Location Clusters: Nhóm các vị trí có đặc điểm tương tự
2. Urban Development Index: Chỉ số phát triển đô thị
3. Amenity Density: Mật độ tiện ích xung quanh
4. Accessibility Score: Điểm tiếp cận tổng hợp
5. Directional Distance: Khoảng cách có hướng (N/S/E/W)
6. Premium Zone Detection: Nhận diện khu vực cao cấp
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Tuple, Optional
import warnings


class GeoIntelligence:
    """
    Tạo các geographic features nâng cao không dùng target encoding.
    
    Features tạo ra:
    - Location Clusters (KMeans)
    - Urban Development Index
    - Amenity Density
    - Accessibility Score
    - Directional Distances
    - Premium Zone Indicators
    """
    
    def __init__(
        self,
        n_clusters: int = 20,
        cluster_method: str = 'kmeans',
        use_advanced_features: bool = True,
    ):
        """
        Args:
            n_clusters: Số lượng clusters
            cluster_method: 'kmeans' hoặc 'quantile'
            use_advanced_features: Tính thêm advanced features
        """
        self.n_clusters = n_clusters
        self.cluster_method = cluster_method
        self.use_advanced_features = use_advanced_features
        
        self.scaler_ = StandardScaler()
        self.kmeans_ = None
        self.cluster_stats_ = {}
        self.urban_index_weights_ = None
        
    def fit(self, df: pd.DataFrame) -> 'GeoIntelligence':
        """
        Fit geo intelligence trên training data.
        
        Args:
            df: DataFrame với Lat/Lon và distance features
        """
        # 1. Fit KMeans clusters
        self._fit_clusters(df)
        
        # 2. Compute cluster statistics
        self._compute_cluster_stats(df)
        
        # 3. Fit urban index weights
        if self.use_advanced_features:
            self._fit_urban_index(df)
            
        return self
    
    def _fit_clusters(self, df: pd.DataFrame) -> None:
        """Fit KMeans clusters dựa trên Lat/Lon."""
        cluster_cols = ['Latitude', 'Longitude']
        
        if not all(c in df.columns for c in cluster_cols):
            warnings.warn("Latitude/Longitude not found. Skipping clustering.")
            return
            
        # Prepare data
        X = df[cluster_cols].copy()
        X = X.fillna(X.median())
        
        # Scale for clustering
        X_scaled = self.scaler_.fit_transform(X)
        
        # Fit KMeans
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
        )
        self.kmeans_.fit(X_scaled)
        
        print(f"    Fitted {self.n_clusters} location clusters")
    
    def _compute_cluster_stats(self, df: pd.DataFrame) -> None:
        """Compute statistics cho từng cluster."""
        if self.kmeans_ is None:
            return
            
        # Predict cluster cho data
        cluster_features = ['Latitude', 'Longitude']
        X = df[cluster_features].fillna(df[cluster_features].median())
        X_scaled = self.scaler_.transform(X)
        clusters = self.kmeans_.predict(X_scaled)
        
        # Compute stats per cluster
        for c in range(self.n_clusters):
            mask = clusters == c
            if mask.sum() > 0:
                self.cluster_stats_[c] = {
                    'n_samples': mask.sum(),
                    'lat_mean': df.loc[mask, 'Latitude'].mean(),
                    'lon_mean': df.loc[mask, 'Longitude'].mean(),
                    'price_mean': df.loc[mask, 'Price'].mean() if 'Price' in df.columns else None,
                    'land_price_mean': df.loc[mask, 'Land_Price_Per_M2'].mean() if 'Land_Price_Per_M2' in df.columns else None,
                }
    
    def _fit_urban_index(self, df: pd.DataFrame) -> None:
        """Fit weights cho urban development index."""
        # Urban index dựa trên các features đã có
        feature_cols = [
            'distance_to_cbd', 'distance_to_nearest_metro',
            'near_metro', 'near_hospital', 'near_school', 'near_market'
        ]
        
        available = [c for c in feature_cols if c in df.columns]
        if len(available) > 0:
            # Normalize weights
            self.urban_index_weights_ = {c: 1.0/len(available) for c in available}
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply geo intelligence features.
        
        Args:
            df: DataFrame cần transform
            
        Returns:
            DataFrame với geo features mới
        """
        df = df.copy()
        
        # 1. Location Clusters
        df = self._add_clusters(df)
        
        # 2. Cluster-level features (sử dụng statistics)
        df = self._add_cluster_features(df)
        
        # 3. Advanced features
        if self.use_advanced_features:
            df = self._add_urban_development_index(df)
            df = self._add_amenity_density(df)
            df = self._add_accessibility_score(df)
            df = self._add_directional_distance(df)
            df = self._add_premium_zone_indicators(df)
            
        return df
    
    def _add_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cluster assignments."""
        if self.kmeans_ is None:
            df['Location_Cluster'] = 0
            return df
            
        cluster_features = ['Latitude', 'Longitude']
        X = df[cluster_features].copy()
        X = X.fillna(X.median())
        
        X_scaled = self.scaler_.transform(X)
        df['Location_Cluster'] = self.kmeans_.predict(X_scaled)
        
        return df
    
    def _add_cluster_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cluster-level features ( KHÔNG dùng target encoding).
        
        Features:
        - Cluster_Size: Số lượng samples trong cluster
        - Cluster_Lat: Vĩ độ trung tâm cluster
        - Cluster_Lon: Kinh độ trung tâm cluster
        """
        df = df.copy()
        
        # Initialize
        df['Cluster_Size'] = 1
        df['Cluster_Lat'] = df['Latitude']
        df['Cluster_Lon'] = df['Longitude']
        
        # Get cluster stats
        if self.cluster_stats_:
            cluster_sizes = {c: s['n_samples'] for c, s in self.cluster_stats_.items()}
            cluster_lats = {c: s['lat_mean'] for c, s in self.cluster_stats_.items()}
            cluster_lons = {c: s['lon_mean'] for c, s in self.cluster_stats_.items()}
            
            df['Cluster_Size'] = df['Location_Cluster'].map(cluster_sizes).fillna(1)
            df['Cluster_Lat'] = df['Location_Cluster'].map(cluster_lats).fillna(df['Latitude'])
            df['Cluster_Lon'] = df['Location_Cluster'].map(cluster_lons).fillna(df['Longitude'])
        
        return df
    
    def _add_urban_development_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính Urban Development Index dựa trên proximity features.
        
        Index = f(distance_to_cbd, metro, hospital, school, market)
        """
        df = df.copy()
        
        # Component features
        components = {}
        
        # CBD proximity (inverse distance - closer = higher)
        if 'distance_to_cbd' in df.columns:
            components['cbd_proximity'] = 1 / (df['distance_to_cbd'] + 0.1)
            
        # Metro proximity
        if 'distance_to_nearest_metro' in df.columns:
            components['metro_proximity'] = 1 / (df['distance_to_nearest_metro'] + 0.1)
            
        # Binary amenities (weighted sum)
        binary_cols = ['near_metro', 'near_hospital', 'near_school', 'near_market']
        components['amenity_score'] = sum(
            df[c].fillna(0) for c in binary_cols if c in df.columns
        )
        
        # Combine components
        if components:
            # Normalize each component
            for name, values in components.items():
                max_val = values.max()
                if max_val > 0:
                    components[name] = values / max_val
            
            # Weighted average
            df['Urban_Development_Index'] = (
                0.4 * components.get('cbd_proximity', 0) +
                0.3 * components.get('metro_proximity', 0) +
                0.3 * components.get('amenity_score', 0)
            )
            
            # Scale to 0-1
            df['Urban_Development_Index'] = df['Urban_Development_Index'].clip(0, 1)
        else:
            df['Urban_Development_Index'] = 0.5
            
        return df
    
    def _add_amenity_density(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính Amenity Density - mật độ tiện ích trong bán kính.
        
        Features:
        - Amenity_Count_1km: Số tiện ích trong 1km
        - Amenity_Count_3km: Số tiện ích trong 3km
        """
        df = df.copy()
        
        # Count amenities within different radii
        # (sử dụng near_* features đã có)
        
        if all(c in df.columns for c in ['near_metro', 'near_hospital', 'near_school', 'near_market']):
            df['Amenity_Count_1km'] = df['near_metro'].fillna(0).astype(int)
            df['Amenity_Count_3km'] = (
                df['near_hospital'].fillna(0).astype(int) +
                df['near_school'].fillna(0).astype(int) +
                df['near_market'].fillna(0).astype(int)
            )
        else:
            df['Amenity_Count_1km'] = 0
            df['Amenity_Count_3km'] = 0
            
        # Total amenity score
        df['Total_Amenity_Score'] = df['Amenity_Count_1km'] + df['Amenity_Count_3km'] * 0.5
        
        return df
    
    def _add_accessibility_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính Accessibility Score - điểm tiếp cận tổng hợp.
        
        Features:
        - Transit_Score: Điểm giao thông công cộng
        - Walkability_Score: Điểm đi bộ
        - Overall_Accessibility: Điểm tổng hợp
        """
        df = df.copy()
        
        # Transit Score (dựa trên metro)
        if 'near_metro' in df.columns and 'distance_to_nearest_metro' in df.columns:
            df['Transit_Score'] = (
                df['near_metro'].fillna(0) * 1.0 +
                (1 - (df['distance_to_nearest_metro'] / df['distance_to_nearest_metro'].max())).fillna(0) * 0.5
            )
        else:
            df['Transit_Score'] = 0.5
            
        # Walkability Score (dựa trên market và school)
        if 'near_market' in df.columns and 'near_school' in df.columns:
            df['Walkability_Score'] = (
                df['near_market'].fillna(0) * 0.6 +
                df['near_school'].fillna(0) * 0.4
            )
        else:
            df['Walkability_Score'] = 0.5
            
        # Overall Accessibility
        df['Overall_Accessibility'] = (
            df['Transit_Score'] * 0.5 +
            df['Walkability_Score'] * 0.5
        )
        
        return df
    
    def _add_directional_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính directional distances - khoảng cách theo hướng.
        
        Features:
        - Distance_North: Khoảng cách về phía Bắc (từ CBD)
        - Distance_East: Khoảng cách về phía Đông (từ CBD)
        - Distance_From_CBD_Normalized: Khoảng cách đã normalize
        """
        df = df.copy()
        
        # CBD center
        CBD_LAT = 10.7758
        CBD_LON = 106.7004
        
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            # Distance components
            df['Distance_North'] = (df['Latitude'] - CBD_LAT).clip(-10, 10)
            df['Distance_East'] = (df['Longitude'] - CBD_LON).clip(-10, 10)
            
            # Quadrant indicator (which direction from CBD)
            df['Quadrant_NE'] = ((df['Latitude'] >= CBD_LAT) & (df['Longitude'] >= CBD_LON)).astype(int)
            df['Quadrant_NW'] = ((df['Latitude'] >= CBD_LAT) & (df['Longitude'] < CBD_LON)).astype(int)
            df['Quadrant_SE'] = ((df['Latitude'] < CBD_LAT) & (df['Longitude'] >= CBD_LON)).astype(int)
            df['Quadrant_SW'] = ((df['Latitude'] < CBD_LAT) & (df['Longitude'] < CBD_LON)).astype(int)
            
            # Normalized distance from CBD
            if 'distance_to_cbd' in df.columns:
                max_dist = df['distance_to_cbd'].max()
                df['Distance_From_CBD_Normalized'] = df['distance_to_cbd'] / max_dist if max_dist > 0 else 0
        else:
            df['Distance_North'] = 0
            df['Distance_East'] = 0
            df['Quadrant_NE'] = 0
            df['Quadrant_NW'] = 0
            df['Quadrant_SE'] = 0
            df['Quadrant_SW'] = 0
            df['Distance_From_CBD_Normalized'] = 0.5
            
        return df
    
    def _add_premium_zone_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nhận diện premium zones dựa trên features KHÔNG dùng target.
        
        Premium Zone Indicators:
        - Near_CBD: Gần trung tâm
        - Near_Metro_Core: Gần metro core
        - High_Amenity: Nhiều tiện ích
        - Premium_Location: Tổng hợp premium indicators
        """
        df = df.copy()
        
        # Near CBD (within 2km)
        if 'distance_to_cbd' in df.columns:
            df['Near_CBD'] = (df['distance_to_cbd'] <= 2.0).astype(int)
            df['Very_Near_CBD'] = (df['distance_to_cbd'] <= 1.0).astype(int)
        else:
            df['Near_CBD'] = 0
            df['Very_Near_CBD'] = 0
            
        # Near Metro Core (Ben Thanh, Saigon stations)
        if 'distance_to_nearest_metro' in df.columns:
            df['Near_Metro_Core'] = (df['distance_to_nearest_metro'] <= 0.5).astype(int)
        else:
            df['Near_Metro_Core'] = 0
            
        # High Amenity (multiple amenities nearby)
        if 'Total_Amenity_Score' in df.columns:
            df['High_Amenity'] = (df['Total_Amenity_Score'] >= 3).astype(int)
        else:
            df['High_Amenity'] = 0
            
        # Premium Location Score
        df['Premium_Location_Score'] = (
            df['Near_CBD'] * 0.4 +
            df['Near_Metro_Core'] * 0.2 +
            df['High_Amenity'] * 0.2 +
            df['Urban_Development_Index'].fillna(0.5) * 0.2
        )
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit và transform."""
        self.fit(df)
        return self.transform(df)


class PremiumLocationDetector:
    """
    Nhận diện và định giá premium locations mà KHÔNG dùng target encoding.
    
    Approach:
    1. Premium streets được identify bằng tên (rule-based)
    2. Premium districts được identify bằng tên (rule-based)
    3. Premium bonus được calculate từ land prices và distance features
    """
    
    # Premium streets (known from domain knowledge)
    PREMIUM_STREETS = {
        'Lê Lợi', 'Đồng Khởi', 'Nguyễn Huệ', 'Lê Duẩn', 'Hàm Nghi', 
        'Pasteur', 'Nam Kỳ Khởi Nghĩa', 'Hàn Thuyên', 'Công Xã Paris',
        'Lý Tự Trọng', 'Tràng Tiền', 'Lý Thái Tổ', 'Hàng Khay',
        'Điện Biên Phủ', 'Phan Chu Trinh', 'Hai Bà Trưng', 'Nguyễn Trãi',
        'Võ Văn Tần', 'Trần Hưng Đạo', 'Nguyễn Thị Minh Khai',
        ' Pasteur', 'Đakao', 'Cao Thắng', 'Đặng Dung',
    }
    
    PREMIUM_DISTRICTS = {'Quận 1', 'Quận 3', 'Hoàn Kiếm', 'Ba Đình'}
    
    PREMIUM_WARDS = {
        'Phường Bến Nghé', 'Phường Đa Kao', 'Phường Tân Định',  # Q1
        'Phường 1', 'Phường 2', 'Phường 3',  # Q3
        'Phường Trúc Bạch', 'Phường Điện Biên',  # Ba Đình
    }
    
    def __init__(
        self,
        street_bonus: float = 0.20,  # 20% bonus for premium streets
        district_bonus: float = 0.10,  # 10% bonus for premium districts
        metro_bonus: float = 0.05,  # 5% bonus for near metro
    ):
        """
        Args:
            street_bonus: Bonus % cho premium streets
            district_bonus: Bonus % cho premium districts
            metro_bonus: Bonus % cho near metro
        """
        self.street_bonus = street_bonus
        self.district_bonus = district_bonus
        self.metro_bonus = metro_bonus
        
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect premium locations và add features.
        
        Args:
            df: DataFrame với Street, District columns
            
        Returns:
            DataFrame với premium features
        """
        df = df.copy()
        
        # 1. Is Premium Street (rule-based)
        if 'Street' in df.columns:
            df['Is_Premium_Street'] = df['Street'].isin(self.PREMIUM_STREETS).astype(int)
        else:
            df['Is_Premium_Street'] = 0
            
        # 2. Is Premium District (rule-based)
        if 'District' in df.columns:
            df['Is_Premium_District'] = df['District'].isin(self.PREMIUM_DISTRICTS).astype(int)
        else:
            df['Is_Premium_District'] = 0
            
        # 3. Is Premium Ward
        if 'Ward' in df.columns:
            df['Is_Premium_Ward'] = df['Ward'].isin(self.PREMIUM_WARDS).astype(int)
        else:
            df['Is_Premium_Ward'] = 0
            
        # 4. Premium Count (tổng số premium indicators)
        df['Premium_Indicators_Count'] = (
            df['Is_Premium_Street'] + 
            df['Is_Premium_District'] + 
            df['Is_Premium_Ward']
        )
        
        # 5. Premium Location Flag
        df['Is_Premium_Location'] = (df['Premium_Indicators_Count'] >= 2).astype(int)
        
        # 6. Premium Bonus Multiplier (tính trực tiếp, không dùng target)
        df['Premium_Bonus_Multiplier'] = (
            1.0 +
            df['Is_Premium_Street'] * self.street_bonus +
            df['Is_Premium_District'] * self.district_bonus +
            (df['near_metro'] if 'near_metro' in df.columns else 0) * self.metro_bonus
        )
        
        return df
    
    def get_premium_info(self, street: str, district: str, ward: str = None) -> Dict:
        """
        Get premium info cho một location.
        
        Returns:
            Dict với premium indicators
        """
        info = {
            'is_premium_street': street in self.PREMIUM_STREETS,
            'is_premium_district': district in self.PREMIUM_DISTRICTS,
            'is_premium_ward': ward in self.PREMIUM_WARDS if ward else False,
            'premium_level': 0,
            'bonus_multiplier': 1.0,
        }
        
        # Calculate premium level
        info['premium_level'] = sum([
            info['is_premium_street'],
            info['is_premium_district'],
            info['is_premium_ward'],
        ])
        
        # Calculate bonus
        info['bonus_multiplier'] = (
            1.0 +
            (self.street_bonus if info['is_premium_street'] else 0) +
            (self.district_bonus if info['is_premium_district'] else 0)
        )
        
        return info


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def add_geo_features(df: pd.DataFrame, n_clusters: int = 20) -> pd.DataFrame:
    """
    Add all geo features trong một function.
    
    Args:
        df: DataFrame cần transform
        n_clusters: Số clusters cho KMeans
        
    Returns:
        DataFrame với tất cả geo features
    """
    # 1. Geo Intelligence
    geo = GeoIntelligence(n_clusters=n_clusters)
    df = geo.fit_transform(df)
    
    # 2. Premium Location Detector
    detector = PremiumLocationDetector()
    df = detector.detect(df)
    
    return df


def get_location_score(
    distance_to_cbd: float,
    distance_to_metro: float,
    near_amenities: int,
    urban_index: float = None,
) -> float:
    """
    Calculate overall location score từ các features.
    
    Returns:
        Score từ 0-1 (1 = best location)
    """
    # Distance score (inverse - closer is better)
    if distance_to_cbd > 0:
        dist_score = max(0, 1 - distance_to_cbd / 20)  # 20km as max reference
    else:
        dist_score = 0.5
        
    # Metro score
    if distance_to_metro > 0:
        metro_score = max(0, 1 - distance_to_metro / 5)
    else:
        metro_score = 0.5
        
    # Amenity score
    amenity_score = min(1, near_amenities / 5)
    
    # Combine
    score = (
        dist_score * 0.4 +
        metro_score * 0.3 +
        amenity_score * 0.3
    )
    
    if urban_index is not None:
        score = score * 0.7 + urban_index * 0.3
        
    return round(score, 3)


if __name__ == "__main__":
    print("=" * 60)
    print("  GEO INTELLIGENCE MODULE - Test")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n = 500
    
    sample_df = pd.DataFrame({
        'Latitude': np.random.uniform(10.7, 10.9, n),
        'Longitude': np.random.uniform(106.6, 106.8, n),
        'Price': np.random.exponential(10, n),
        'Land_Price_Per_M2': np.random.uniform(30, 200, n),
        'distance_to_cbd': np.random.uniform(0, 15, n),
        'distance_to_nearest_metro': np.random.uniform(0, 5, n),
        'near_metro': np.random.choice([0, 1], n),
        'near_hospital': np.random.choice([0, 1], n),
        'near_school': np.random.choice([0, 1], n),
        'near_market': np.random.choice([0, 1], n),
        'Street': np.random.choice(['Lê Lợi', 'Nguyễn Huệ', 'Võ Văn Tần', 'Unknown'], n),
        'District': np.random.choice(['Quận 1', 'Quận 3', 'Quận 7', 'Bình Thạnh'], n),
        'Ward': np.random.choice(['Phường 1', 'Phường 2', 'Phường 3'], n),
    })
    
    print(f"\n  Sample data: {len(sample_df)} rows")
    
    # Test GeoIntelligence
    print("\n  Testing GeoIntelligence:")
    geo = GeoIntelligence(n_clusters=10)
    df_geo = geo.fit_transform(sample_df)
    
    new_cols = [c for c in df_geo.columns if c not in sample_df.columns]
    print(f"    New columns added: {len(new_cols)}")
    print(f"    Columns: {new_cols}")
    
    # Test PremiumLocationDetector
    print("\n  Testing PremiumLocationDetector:")
    detector = PremiumLocationDetector()
    df_premium = detector.detect(df_geo)
    
    premium_cols = [c for c in df_premium.columns if 'Premium' in c or 'Is_Premium' in c]
    print(f"    Premium columns: {premium_cols}")
    print(f"    Premium locations: {df_premium['Is_Premium_Location'].sum()}")
    
    # Test combined
    print("\n  Testing add_geo_features:")
    df_final = add_geo_features(sample_df, n_clusters=10)
    print(f"    Final columns: {len(df_final.columns)}")
    print(f"    Original: {len(sample_df.columns)}, Added: {len(df_final.columns) - len(sample_df.columns)}")
    
    print("\n" + "=" * 60)
    print("  Test Complete!")
    print("=" * 60)
