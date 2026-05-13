"""
Smoothed Target Encoding Module - Anti-Leakage Geographic Encoding
Giải quyết vấn đề target encoding leakage trong house price prediction.

Key Improvements:
1. Smoothed Target Encoding: Giảm overfitting cho categories có ít samples
2. K-Fold Target Encoding: Ngăn chặn data leakage khi train/validate
3. Minimum Sample Threshold: Bỏ qua categories có quá ít samples
4. Geographic Hierarchy: Street < Ward < District để giảm trọng số Street

Formula:
TE = (n * category_mean + m * global_mean) / (n + m)

Where:
- n = number of samples in category
- m = smoothing factor (higher = more regularization)
- global_mean = global average house price
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Dict, Tuple, Optional
import warnings


class SmoothedTargetEncoder:
    """
    Smoothed Target Encoding với anti-leakage protections.
    
    Features:
    - Smoothed encoding: Giảm overfitting cho small categories
    - K-Fold encoding: Ngăn data leakage trong cross-validation
    - Minimum threshold: Bỏ qua categories có ít samples
    - Weight decay: Giảm trọng số Street/Ward encoding
    """
    
    def __init__(
        self,
        smoothing_district: int = 10,
        smoothing_ward: int = 30,
        smoothing_street: int = 50,
        smoothing_cluster: int = 20,
        min_samples: int = 5,
        encode_kfold: bool = True,
        n_folds: int = 5,
        random_state: int = 42,
        reduce_street_importance: bool = True,
    ):
        """
        Args:
            smoothing_district: Smoothing factor cho District (có nhiều data)
            smoothing_ward: Smoothing factor cho Ward
            smoothing_street: Smoothing factor cho Street (cao nhất - ít data nhất)
            smoothing_cluster: Smoothing factor cho Location Cluster
            min_samples: Ngưỡng tối thiểu samples để encode
            encode_kfold: Nếu True, dùng K-Fold encoding cho training
            n_folds: Số folds cho K-Fold encoding
            random_state: Random seed
            reduce_street_importance: Nếu True, giảm 30% importance của Street encoding
        """
        self.smoothing = {
            'District': smoothing_district,
            'Ward': smoothing_ward,
            'Street': smoothing_street,
            'Cluster': smoothing_cluster,
        }
        self.min_samples = min_samples
        self.encode_kfold = encode_kfold
        self.n_folds = n_folds
        self.random_state = random_state
        self.reduce_street_importance = reduce_street_importance
        
        # Storage cho encodings
        self.encodings_: Dict[str, Dict] = {}
        self.global_mean_: float = 0.0
        self.encoding_stats_: Dict[str, pd.DataFrame] = {}
        
    def fit(self, df: pd.DataFrame, target_col: str = 'Price') -> 'SmoothedTargetEncoder':
        """
        Fit encoder trên training data.
        
        Args:
            df: DataFrame với target column
            target_col: Tên cột target
            
        Returns:
            Self
        """
        self.global_mean_ = df[target_col].mean()
        
        # Compute encodings cho từng level
        for level in ['District', 'Ward', 'Street']:
            self._compute_encoding(df, level, target_col)
        
        # Compute cluster encoding
        if 'Location_Cluster' in df.columns:
            self._compute_encoding(df, 'Cluster', target_col)
            
        return self
    
    def _compute_encoding(
        self, 
        df: pd.DataFrame, 
        level: str, 
        target_col: str
    ) -> None:
        """
        Compute smoothed target encoding cho một level.
        
        Formula: TE = (n * mean + m * global_mean) / (n + m)
        """
        col = level if level != 'Cluster' else 'Location_Cluster'
        
        if col not in df.columns:
            return
            
        # Compute stats per category
        stats = df.groupby(col)[target_col].agg(['mean', 'count', 'std']).reset_index()
        stats.columns = [col, 'cat_mean', 'cat_count', 'cat_std']
        
        # Apply minimum sample threshold
        stats['valid'] = stats['cat_count'] >= self.min_samples
        
        # Smoothed encoding
        smoothing = self.smoothing.get(level, 20)
        stats['encoded'] = (
            stats['cat_count'] * stats['cat_mean'] + smoothing * self.global_mean_
        ) / (stats['cat_count'] + smoothing)
        
        # For categories with too few samples, use global mean
        stats.loc[~stats['valid'], 'encoded'] = self.global_mean_
        
        # Reduce Street importance if requested
        if self.reduce_street_importance and level == 'Street':
            # Blend Street encoding 30% towards global mean
            blend_factor = 0.3
            stats['encoded'] = (
                blend_factor * self.global_mean_ + 
                (1 - blend_factor) * stats['encoded']
            )
        
        # Create encoding dictionary
        self.encodings_[level] = dict(zip(stats[col], stats['encoded']))
        self.encoding_stats_[level] = stats
        
        print(f"    {level}: {len(stats)} categories, "
              f"{stats['valid'].sum()} valid (≥{self.min_samples} samples)")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply encoding to DataFrame.
        
        Args:
            df: DataFrame cần encode
            
        Returns:
            DataFrame với các cột encoded mới
        """
        df = df.copy()
        
        for level in ['District', 'Ward', 'Street']:
            col = level if level != 'Cluster' else 'Location_Cluster'
            if col not in df.columns:
                continue
                
            encoded_col = f'{level}_Encoded'
            enc_dict = self.encodings_.get(level, {})
            
            df[encoded_col] = df[col].map(enc_dict).fillna(self.global_mean_)
        
        return df
    
    def fit_transform(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'Price'
    ) -> Tuple[pd.DataFrame, 'SmoothedTargetEncoder']:
        """
        Fit và transform trong một bước (cho training data).
        
        Returns:
            Tuple of (encoded DataFrame, fitted encoder)
        """
        self.fit(df, target_col)
        df_encoded = self.transform(df)
        return df_encoded, self
    
    def get_encoding(self, level: str) -> Dict:
        """Get encoding dictionary cho một level."""
        return self.encodings_.get(level, {})
    
    def get_global_mean(self) -> float:
        """Get global mean của target."""
        return self.global_mean_
    
    def get_encoding_stats(self, level: str) -> pd.DataFrame:
        """Get detailed encoding statistics."""
        return self.encoding_stats_.get(level, pd.DataFrame())
    
    def summary(self) -> Dict:
        """Get summary của tất cả encodings."""
        summary = {
            'global_mean': self.global_mean_,
            'levels': {}
        }
        for level, stats in self.encoding_stats_.items():
            valid_stats = stats[stats['valid']]
            summary['levels'][level] = {
                'total_categories': len(stats),
                'valid_categories': len(valid_stats),
                'mean_samples_per_category': stats['cat_count'].mean(),
                'encoding_range': (stats['encoded'].min(), stats['encoded'].max()),
            }
        return summary


class KFoldTargetEncoder:
    """
    K-Fold Target Encoding để ngăn chặn data leakage trong training.
    
    Khi train model với cross-validation, target encoding phải được tính
    trên fold hiện tại để tránh leakage từ validation set.
    
    Algorithm:
    1. Split data into K folds
    2. For each fold i:
       - Compute encoding từ folds khác (1...K except i)
       - Apply encoding cho fold i
    3. Final encoding (for inference) được compute trên toàn bộ training data
    """
    
    def __init__(
        self,
        smoothing_district: int = 10,
        smoothing_ward: int = 30,
        smoothing_street: int = 50,
        smoothing_cluster: int = 20,
        min_samples: int = 5,
        n_folds: int = 5,
        random_state: int = 42,
        reduce_street_importance: float = 0.3,
    ):
        """
        Args:
            smoothing_*: Smoothing factors cho từng level
            min_samples: Ngưỡng tối thiểu samples
            n_folds: Số folds cho K-Fold encoding
            random_state: Random seed
            reduce_street_importance: % blend towards global mean (0.3 = 30%)
        """
        self.smoothing = {
            'District': smoothing_district,
            'Ward': smoothing_ward,
            'Street': smoothing_street,
            'Cluster': smoothing_cluster,
        }
        self.min_samples = min_samples
        self.n_folds = n_folds
        self.random_state = random_state
        self.reduce_street_importance = reduce_street_importance
        
        self.final_encoder_: Optional[SmoothedTargetEncoder] = None
        self.global_mean_: float = 0.0
        
    def fit_transform_kfold(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'Price'
    ) -> pd.DataFrame:
        """
        Fit với K-Fold encoding trên training data.
        
        Args:
            df: Training DataFrame
            target_col: Tên cột target
            
        Returns:
            DataFrame với encoded columns (OOF encoding)
        """
        df = df.copy()
        self.global_mean_ = df[target_col].mean()
        
        # Khởi tạo encoded columns với global mean
        for level in ['District', 'Ward', 'Street']:
            col = level if level != 'Cluster' else 'Location_Cluster'
            if col in df.columns:
                df[f'{level}_Encoded'] = self.global_mean_
        
        # K-Fold encoding
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df)):
            # Training fold để compute encoding
            df_train_fold = df.iloc[train_idx]
            
            # Compute encodings từ training fold
            fold_encodings = self._compute_fold_encodings(df_train_fold, target_col)
            
            # Apply cho validation fold
            for level, enc_dict in fold_encodings.items():
                col = level if level != 'Cluster' else 'Location_Cluster'
                if col in df.columns:
                    # Apply encoding với blending để giảm Street importance
                    blend_factor = self.reduce_street_importance if level == 'Street' else 0.0
                    
                    for idx in val_idx:
                        cat_val = df.iloc[idx][col]
                        encoded_val = enc_dict.get(cat_val, self.global_mean_)
                        
                        # Blend với global mean cho Street
                        if blend_factor > 0:
                            encoded_val = (
                                blend_factor * self.global_mean_ + 
                                (1 - blend_factor) * encoded_val
                            )
                        
                        df.iloc[idx, df.columns.get_loc(f'{level}_Encoded')] = encoded_val
        
        # Fit final encoder trên toàn bộ data (cho inference)
        self.final_encoder_ = SmoothedTargetEncoder(
            smoothing_district=self.smoothing['District'],
            smoothing_ward=self.smoothing['Ward'],
            smoothing_street=self.smoothing['Street'],
            smoothing_cluster=self.smoothing['Cluster'],
            min_samples=self.min_samples,
            reduce_street_importance=True,
        )
        self.final_encoder_.fit(df, target_col)
        
        return df
    
    def _compute_fold_encodings(
        self, 
        df_fold: pd.DataFrame, 
        target_col: str
    ) -> Dict[str, Dict]:
        """
        Compute smoothed encodings từ một fold.
        
        Returns:
            Dict mapping level -> {category: encoded_value}
        """
        encodings = {}
        global_mean = df_fold[target_col].mean()
        
        for level in ['District', 'Ward', 'Street']:
            col = level if level != 'Cluster' else 'Location_Cluster'
            
            if col not in df_fold.columns:
                continue
                
            # Compute stats
            stats = df_fold.groupby(col)[target_col].agg(['mean', 'count'])
            
            # Smoothed encoding
            smoothing = self.smoothing.get(level, 20)
            encoded = {}
            
            for cat, row in stats.iterrows():
                count = row['count']
                mean = row['mean']
                
                if count >= self.min_samples:
                    encoded[cat] = (count * mean + smoothing * global_mean) / (count + smoothing)
                else:
                    encoded[cat] = global_mean
            
            encodings[level] = encoded
            
        return encodings
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply final encoding (fitted on all training data).
        
        Args:
            df: DataFrame cần encode
            
        Returns:
            DataFrame với encoded columns
        """
        if self.final_encoder_ is None:
            raise ValueError("Encoder chưa được fit! Gọi fit_transform_kfold trước.")
        
        return self.final_encoder_.transform(df)
    
    def get_final_encoder(self) -> SmoothedTargetEncoder:
        """Get final encoder (đã fit trên toàn bộ training data)."""
        return self.final_encoder_


class GeoWeightedEncoder:
    """
    Geographic Hierarchy Weighted Encoding.
    
    Giảm trọng số của Street encoding vì:
    1. Streets có ít samples hơn District/Ward
    2. Streets dễ bị overfitting hơn
    3. Premium streets đã được capture bởi Land_Price và distance features
    
    Strategy:
    - District: Weight = 1.0 (có nhiều data, reliable)
    - Ward: Weight = 0.7 (medium reliability)
    - Street: Weight = 0.4 (ít data, high risk overfitting)
    - Cluster: Weight = 0.6 (computed from Lat/Lon)
    """
    
    def __init__(
        self,
        district_weight: float = 1.0,
        ward_weight: float = 0.7,
        street_weight: float = 0.4,
        cluster_weight: float = 0.6,
        **kwargs
    ):
        """
        Args:
            district_weight: Trọng số cho District_Encoded
            ward_weight: Trọng số cho Ward_Encoded
            street_weight: Trọng số cho Street_Encoded
            cluster_weight: Trọng số cho Cluster_Encoded
            **kwargs: Arguments cho SmoothedTargetEncoder
        """
        self.weights = {
            'District': district_weight,
            'Ward': ward_weight,
            'Street': street_weight,
            'Cluster': cluster_weight,
        }
        self.encoder_ = SmoothedTargetEncoder(**kwargs)
        
    def fit(self, df: pd.DataFrame, target_col: str = 'Price') -> 'GeoWeightedEncoder':
        """Fit encoder."""
        self.encoder_.fit(df, target_col)
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform với weighted encoding.
        
        Tạo thêm các cột weighted version:
        - District_Encoded_Weighted
        - Ward_Encoded_Weighted
        - Street_Encoded_Weighted
        - Cluster_Encoded_Weighted
        """
        df = self.encoder_.transform(df)
        
        # Tạo weighted versions
        for level, weight in self.weights.items():
            orig_col = f'{level}_Encoded'
            if orig_col in df.columns:
                df[f'{level}_Encoded_Weighted'] = df[orig_col] * weight
                
        return df
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'Price') -> pd.DataFrame:
        """Fit và transform."""
        self.fit(df, target_col)
        return self.transform(df)
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return self.weights.copy()
    
    def summary(self) -> Dict:
        """Get summary với weights."""
        summary = self.encoder_.summary()
        summary['weights'] = self.weights
        return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_leakage_free_encoder(
    smoothing_district: int = 10,
    smoothing_ward: int = 30,
    smoothing_street: int = 50,
    use_kfold: bool = True,
    reduce_street_importance: float = 0.3,
) -> Tuple[SmoothedTargetEncoder | KFoldTargetEncoder, pd.DataFrame]:
    """
    Factory function để tạo encoder với anti-leakage settings tối ưu.
    
    Returns:
        Tuple of (encoder, encoded_df) nếu fit_transform
    """
    if use_kfold:
        encoder = KFoldTargetEncoder(
            smoothing_district=smoothing_district,
            smoothing_ward=smoothing_ward,
            smoothing_street=smoothing_street,
            reduce_street_importance=reduce_street_importance,
        )
    else:
        encoder = SmoothedTargetEncoder(
            smoothing_district=smoothing_district,
            smoothing_ward=smoothing_ward,
            smoothing_street=smoothing_street,
            reduce_street_importance=True,
        )
    
    return encoder


def get_premium_street_bonus(
    street: str,
    district: str,
    global_mean: float,
    land_price_per_m2: float,
    area: float,
) -> float:
    """
    Calculate premium bonus cho premium streets mà không dùng target encoding.
    
    Logic:
    - Premium streets có land price cao hơn (đã reflect trong land_price_per_m2)
    - Thêm bonus factor cho các streets đặc biệt premium
    
    Args:
        street: Tên đường
        district: Tên quận
        global_mean: Global mean price
        land_price_per_m2: Land price per m2
        area: Diện tích
        
    Returns:
        Bonus multiplier (1.0 = no bonus)
    """
    PREMIUM_STREETS = {
        'Lê Lợi', 'Đồng Khởi', 'Nguyễn Huệ', 'Lê Duẩn', 'Hàm Nghi', 
        'Pasteur', 'Nam Kỳ Khởi Nghĩa', 'Hàn Thuyên', 'Công Xã Paris',
        'Lý Tự Trọng', 'Tràng Tiền', 'Lý Thái Tổ', 'Hàng Khay',
        'Điện Biên Phủ', 'Phan Chu Trinh', 'Hai Bà Trưng', 'Nguyễn Trãi',
    }
    
    PREMIUM_DISTRICTS = {'Quận 1', 'Quận 3', 'Hoàn Kiếm', 'Ba Đình'}
    
    bonus = 1.0
    
    # Street bonus (không dùng target encoding)
    if street in PREMIUM_STREETS:
        # Base bonus: 10-20% tùy đường
        street_bonus = 1.15
        bonus *= street_bonus
        
    # District bonus (không dùng target encoding)
    if district in PREMIUM_DISTRICTS:
        district_bonus = 1.10
        bonus *= district_bonus
        
    return bonus


if __name__ == "__main__":
    # Test module
    print("=" * 60)
    print("  SMOOTHED TARGET ENCODING MODULE - Test")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    sample_df = pd.DataFrame({
        'District': np.random.choice(['Quận 1', 'Quận 3', 'Quận 7', 'Bình Thạnh'], n),
        'Ward': np.random.choice([f'Phường {i}' for i in range(1, 20)], n),
        'Street': np.random.choice(['Lê Lợi', 'Nguyễn Huệ', 'Võ Văn Tần', 'Unknown'], n),
        'Price': np.random.exponential(10, n),  # triệu VND
        'Area': np.random.uniform(50, 200, n),
    })
    
    print(f"\n  Sample data: {len(sample_df)} rows")
    print(f"  Global mean price: {sample_df['Price'].mean():.2f}")
    
    # Test SmoothedTargetEncoder
    print("\n  Testing SmoothedTargetEncoder:")
    encoder = SmoothedTargetEncoder(
        smoothing_district=10,
        smoothing_ward=30,
        smoothing_street=50,
        reduce_street_importance=True,
    )
    
    df_encoded = encoder.fit_transform(sample_df, target_col='Price')
    
    print("\n  Encoding Summary:")
    summary = encoder.summary()
    for level, stats in summary['levels'].items():
        print(f"    {level}: {stats['valid_categories']} valid categories")
    
    # Test KFoldTargetEncoder
    print("\n  Testing KFoldTargetEncoder:")
    kfold_encoder = KFoldTargetEncoder(
        smoothing_district=10,
        smoothing_ward=30,
        smoothing_street=50,
        reduce_street_importance=0.3,
    )
    
    df_kfold = kfold_encoder.fit_transform_kfold(sample_df, target_col='Price')
    print(f"    OOF encoded: {len(df_kfold)} rows")
    print(f"    Columns: {[c for c in df_kfold.columns if 'Encoded' in c]}")
    
    print("\n" + "=" * 60)
    print("  Test Complete!")
    print("=" * 60)
