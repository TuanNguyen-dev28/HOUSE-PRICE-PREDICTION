"""
Comprehensive Data Cleaning Script
Cleans house price data based on evaluation findings.
"""

import pandas as pd
import numpy as np
from scipy import stats

def load_data(path):
    return pd.read_csv(path)

def remove_duplicates(df):
    """Remove near-duplicates based on Address + Area + Price"""
    print("\n[STEP 1] REMOVING DUPLICATES")
    print("-" * 50)
    
    initial_count = len(df)
    
    # Remove exact duplicates
    df = df.drop_duplicates()
    exact_dup_removed = initial_count - len(df)
    
    # Remove near-duplicates (same Address + Area + Price)
    df_nodup = df.drop_duplicates(subset=['Address', 'Area', 'Price'], keep='first')
    near_dup_removed = len(df) - len(df_nodup)
    
    print(f"  Exact duplicates removed: {exact_dup_removed}")
    print(f"  Near-duplicates removed: {near_dup_removed}")
    print(f"  Records remaining: {len(df_nodup)}")
    
    return df_nodup

def handle_outliers(df, method='winsorize'):
    """Handle outliers using winsorization or removal"""
    print("\n[STEP 2] HANDLING OUTLIERS")
    print("-" * 50)
    
    # Define outlier thresholds
    outlier_config = {
        'Area': {'lower': 10, 'upper': 500},        # Min 10m2, Max 500m2
        'Frontage': {'lower': 2, 'upper': 20},       # Min 2m, Max 20m
        'Access Road': {'lower': 1, 'upper': 50},    # Min 1m, Max 50m
        'Floors': {'lower': 1, 'upper': 15},         # Min 1 floor, Max 15 floors
        'Bedrooms': {'lower': 0, 'upper': 10},       # Min 0, Max 10 bedrooms
        'Bathrooms': {'lower': 0, 'upper': 8},       # Min 0, Max 8 bathrooms
        'Price': {'lower': 0.5, 'upper': 500},       # Min 0.5 ty, Max 500 ty
    }
    
    df_clean = df.copy()
    total_outliers = 0
    
    for col, bounds in outlier_config.items():
        if col in df_clean.columns:
            before = len(df_clean)
            
            # Handle NaN values first - don't count them as outliers
            mask = df_clean[col].notna()
            
            # Cap outliers instead of removing
            df_clean.loc[mask & (df_clean[col] < bounds['lower']), col] = bounds['lower']
            df_clean.loc[mask & (df_clean[col] > bounds['upper']), col] = bounds['upper']
            
            outliers = before - df_clean[col].notna().sum()
            total_outliers += outliers
            
            print(f"  {col}: capped to [{bounds['lower']}, {bounds['upper']}]")
    
    print(f"  Total outlier values capped: {total_outliers}")
    print(f"  Records remaining: {len(df_clean)}")
    
    return df_clean

def impute_missing_values(df):
    """Impute missing values based on feature type"""
    print("\n[STEP 3] IMPUTING MISSING VALUES")
    print("-" * 50)
    
    df_clean = df.copy()
    
    # Categorical features - use mode
    categorical_cols = ['House direction', 'Balcony direction', 'Legal status', 'Furniture state']
    
    print("  Categorical features (imputing with mode):")
    for col in categorical_cols:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()
            if len(mode_val) > 0:
                mode_val = mode_val[0]
                null_count = df_clean[col].isnull().sum()
                df_clean[col] = df_clean[col].fillna(mode_val)
                print(f"    {col}: {null_count} missing -> filled with '{mode_val}'")
    
    # Numerical features - use median by district
    numerical_cols = ['Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms']
    
    print("\n  Numerical features (imputing with median):")
    
    # Extract district for group-based imputation
    df_clean['District'] = df_clean['Address'].str.split(',').str[-2].str.strip()
    
    for col in numerical_cols:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            null_count = df_clean[col].isnull().sum()
            
            # Impute with district median if possible
            district_median = df_clean.groupby('District')[col].transform('median')
            df_clean[col] = df_clean[col].fillna(district_median)
            
            # Fill remaining with overall median
            overall_median = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(overall_median)
            
            print(f"    {col}: {null_count} missing -> filled with median")
    
    # Drop temporary District column
    df_clean = df_clean.drop(columns=['District'])
    
    return df_clean

def remove_low_quality_records(df):
    """Remove records with too many missing features"""
    print("\n[STEP 4] REMOVING LOW-QUALITY RECORDS")
    print("-" * 50)
    
    initial_count = len(df)
    
    # Calculate missing features per row
    feature_cols = ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms', 
                    'House direction', 'Balcony direction', 'Legal status', 'Furniture state']
    
    df['missing_count'] = df[feature_cols].isnull().sum(axis=1)
    
    # Remove records with > 5 missing features
    threshold = 5
    df_clean = df[df['missing_count'] <= threshold].copy()
    df_clean = df_clean.drop(columns=['missing_count'])
    
    removed = initial_count - len(df_clean)
    print(f"  Records with > {threshold} missing features removed: {removed}")
    print(f"  Records remaining: {len(df_clean)}")
    
    return df_clean

def validate_cleaned_data(df):
    """Validate cleaned data quality"""
    print("\n[STEP 5] DATA VALIDATION")
    print("=" * 50)
    
    print(f"\n  Final Dataset Statistics:")
    print(f"    Total records: {len(df):,}")
    print(f"    Total features: {len(df.columns)}")
    
    # Missing values check
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    print(f"\n  Missing Values (should all be 0):")
    any_missing = False
    for col in df.columns:
        if missing[col] > 0:
            print(f"    {col}: {missing[col]} ({missing_pct[col]}%)")
            any_missing = True
    if not any_missing:
        print("    All columns complete!")
    
    # Price distribution
    print(f"\n  Price Distribution:")
    print(f"    Min: {df['Price'].min():.2f}")
    print(f"    Max: {df['Price'].max():.2f}")
    print(f"    Mean: {df['Price'].mean():.2f}")
    print(f"    Median: {df['Price'].median():.2f}")
    print(f"    Std: {df['Price'].std():.2f}")
    
    # Skewness
    skewness = stats.skew(df['Price'])
    print(f"    Skewness: {skewness:.3f}")
    if abs(skewness) > 2:
        print("    WARNING: High skewness - consider log transformation")
    
    # Duplicate check
    duplicates = df.duplicated(subset=['Address', 'Area', 'Price']).sum()
    print(f"\n  Duplicate Check:")
    print(f"    Near-duplicates: {duplicates}")
    
    return df

def main():
    print("=" * 60)
    print("           HOUSE PRICE DATA CLEANING PIPELINE")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df = load_data("data/raw/house_data.csv")
    print(f"  Initial records: {len(df):,}")
    print(f"  Initial features: {len(df.columns)}")
    
    # Step 1: Remove duplicates
    df = remove_duplicates(df)
    
    # Step 2: Handle outliers
    df = handle_outliers(df)
    
    # Step 3: Impute missing values
    df = impute_missing_values(df)
    
    # Step 4: Remove low-quality records
    df = remove_low_quality_records(df)
    
    # Step 5: Validate
    df = validate_cleaned_data(df)
    
    # Save cleaned data
    output_path = "data/raw/house_data_cleaned.csv"
    df.to_csv(output_path, index=False)
    print(f"\n  Cleaned data saved to: {output_path}")
    
    # Also update the main file
    backup_path = "data/raw/house_data_backup.csv"
    df.to_csv(backup_path, index=False)
    print(f"  Backup saved to: {backup_path}")
    
    print("\n" + "=" * 60)
    print("           CLEANING COMPLETE!")
    print("=" * 60)
    
    return df

if __name__ == "__main__":
    df = main()
