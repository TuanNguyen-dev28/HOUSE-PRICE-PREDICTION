"""
Comprehensive Data Evaluation Script
Analyzes data quality, distributions, outliers, and provides recommendations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def load_and_explore_data():
    """Load data and basic exploration"""
    print("=" * 70)
    print("                    DATA QUALITY EVALUATION REPORT")
    print("=" * 70)
    
    df = pd.read_csv("data/raw/house_data.csv")
    print(f"\n[1] BASIC OVERVIEW")
    print(f"    Total records: {len(df):,}")
    print(f"    Total features: {len(df.columns)}")
    print(f"    Features: {df.columns.tolist()}")
    
    return df

def evaluate_missing_values(df):
    """Analyze missing values"""
    print(f"\n{'='*70}")
    print("[2] MISSING VALUES ANALYSIS")
    print("=" * 70)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    }).sort_values('Missing %', ascending=False)
    
    # Only show columns with missing values
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    if len(missing_df) > 0:
        print("\n    Columns with missing values:")
        for col, row in missing_df.iterrows():
            status = "HIGH" if row['Missing %'] > 50 else ("MEDIUM" if row['Missing %'] > 20 else "LOW")
            print(f"      {col:<20} {row['Missing Count']:>6,} ({row['Missing %']:>5.1f}%) [{status}]")
    else:
        print("    No missing values found!")
    
    # Summary
    total_cells = df.shape[0] * df.shape[1]
    total_missing = missing.sum()
    print(f"\n    Overall data completeness: {((total_cells - total_missing) / total_cells * 100):.2f}%")
    
    return missing_df

def evaluate_duplicates(df):
    """Check for duplicate records"""
    print(f"\n{'='*70}")
    print("[3] DUPLICATE ANALYSIS")
    print("=" * 70)
    
    duplicates = df.duplicated().sum()
    print(f"\n    Exact duplicate rows: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")
    
    # Check for near-duplicates based on key columns
    key_cols = ['Address', 'Area', 'Price']
    existing_cols = [c for c in key_cols if c in df.columns]
    if existing_cols:
        near_dup = df.duplicated(subset=existing_cols).sum()
        print(f"    Near-duplicates (same Address+Area+Price): {near_dup:,} ({near_dup/len(df)*100:.2f}%)")
    
    return duplicates

def evaluate_price_distribution(df):
    """Analyze price distribution"""
    print(f"\n{'='*70}")
    print("[4] PRICE DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    prices = df['Price'].dropna()
    
    print(f"\n    Price Statistics (ty):")
    print(f"      Min:        {prices.min():.2f}")
    print(f"      Max:        {prices.max():.2f}")
    print(f"      Mean:       {prices.mean():.2f}")
    print(f"      Median:     {prices.median():.2f}")
    print(f"      Std Dev:    {prices.std():.2f}")
    print(f"      Skewness:   {prices.skew():.3f}")
    
    # Percentiles
    print(f"\n    Percentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"      {p}th:         {np.percentile(prices, p):.2f}")
    
    # Price ranges
    price_ranges = {
        '< 5 ty': (prices < 5).sum(),
        '5-10 ty': ((prices >= 5) & (prices < 10)).sum(),
        '10-20 ty': ((prices >= 10) & (prices < 20)).sum(),
        '20-50 ty': ((prices >= 20) & (prices < 50)).sum(),
        '50-100 ty': ((prices >= 50) & (prices < 100)).sum(),
        '> 100 ty': (prices >= 100).sum()
    }
    
    print(f"\n    Price Range Distribution:")
    for range_name, count in price_ranges.items():
        pct = count / len(prices) * 100
        bar = "#" * int(pct / 2)
        print(f"      {range_name:<15} {count:>6,} ({pct:>5.2f}%) {bar}")
    
    return prices

def evaluate_outliers(df):
    """Detect and analyze outliers"""
    print(f"\n{'='*70}")
    print("[5] OUTLIER DETECTION")
    print("=" * 70)
    
    numeric_cols = ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms', 'Price']
    
    print(f"\n    Using IQR method (values outside Q1-1.5*IQR to Q3+1.5*IQR)")
    print(f"\n    Outliers by feature:")
    print("-" * 60)
    print(f"    {'Feature':<15} {'Lower':<10} {'Upper':<10} {'Outliers':<12} {'%':<8}")
    print("-" * 60)
    
    outlier_summary = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        data = df[col].dropna()
        if len(data) == 0:
            continue
            
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = ((data < lower) | (data > upper)).sum()
        pct = outliers / len(data) * 100
        
        print(f"    {col:<15} {lower:<10.2f} {upper:<10.2f} {outliers:<12,} {pct:<8.2f}")
        
        outlier_summary[col] = {
            'lower': lower, 'upper': upper,
            'count': outliers, 'pct': pct
        }
    
    return outlier_summary

def evaluate_district_analysis(df):
    """Analyze prices by district"""
    print(f"\n{'='*70}")
    print("[6] DISTRICT/CITY PRICE ANALYSIS")
    print("=" * 70)
    
    # Extract district
    df['District'] = df['Address'].str.split(',').str[-2].str.strip()
    df['City'] = df['Address'].str.split(',').str[-1].str.strip()
    
    # HCM districts
    hcm_df = df[df['City'].str.contains('HCM|Ho Chi', case=False, na=False)]
    hanoi_df = df[df['City'].str.contains('Hanoi|Ha Noi', case=False, na=False)]
    
    print(f"\n    Ho Chi Minh City:")
    print(f"      Total records: {len(hcm_df):,}")
    if len(hcm_df) > 0:
        district_stats = hcm_df.groupby('District')['Price'].agg(['count', 'mean', 'median', 'min', 'max'])
        district_stats = district_stats.sort_values('mean', ascending=False)
        print(f"\n      Top 10 districts by avg price:")
        print("      " + "-" * 65)
        print(f"      {'District':<25} {'Count':>8} {'Mean':>10} {'Median':>10} {'Max':>10}")
        print("      " + "-" * 65)
        for district, row in district_stats.head(10).iterrows():
            d = str(district).encode('ascii', 'replace').decode('ascii')
            print(f"      {d:<25} {row['count']:>8.0f} {row['mean']:>10.2f} {row['median']:>10.2f} {row['max']:>10.2f}")
    
    print(f"\n    Hanoi:")
    print(f"      Total records: {len(hanoi_df):,}")
    if len(hanoi_df) > 0:
        district_stats = hanoi_df.groupby('District')['Price'].agg(['count', 'mean', 'median', 'min', 'max'])
        district_stats = district_stats.sort_values('mean', ascending=False)
        print(f"\n      Top 10 districts by avg price:")
        print("      " + "-" * 65)
        print(f"      {'District':<25} {'Count':>8} {'Mean':>10} {'Median':>10} {'Max':>10}")
        print("      " + "-" * 65)
        for district, row in district_stats.head(10).iterrows():
            d = str(district).encode('ascii', 'replace').decode('ascii')
            print(f"      {d:<25} {row['count']:>8.0f} {row['mean']:>10.2f} {row['median']:>10.2f} {row['max']:>10.2f}")
    
    return df

def evaluate_feature_quality(df):
    """Evaluate feature quality and correlations"""
    print(f"\n{'='*70}")
    print("[7] FEATURE QUALITY ANALYSIS")
    print("=" * 70)
    
    numeric_cols = ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms', 'Price']
    
    # Correlation with price
    print(f"\n    Correlation with Price:")
    print("    " + "-" * 40)
    
    correlations = {}
    for col in numeric_cols:
        if col in df.columns and col != 'Price':
            corr = df[col].corr(df['Price'])
            correlations[col] = corr
            bar = "#" * int(abs(corr) * 30)
            sign = "+" if corr > 0 else "-"
            print(f"      {col:<15} {sign}{abs(corr):.4f} {bar}")
    
    # Cardinality of categorical columns
    cat_cols = ['House direction', 'Balcony direction', 'Legal status', 'Furniture state']
    print(f"\n    Categorical Feature Cardinality:")
    print("    " + "-" * 40)
    for col in cat_cols:
        if col in df.columns:
            n_unique = df[col].nunique()
            print(f"      {col:<20} {n_unique:>3} unique values")
            print(f"        Values: {df[col].value_counts().head(5).to_dict()}".encode('ascii', 'replace').decode('ascii'))
    
    return correlations

def generate_recommendations(df, missing_df, outlier_summary, correlations):
    """Generate data cleaning and improvement recommendations"""
    print(f"\n{'='*70}")
    print("[8] RECOMMENDATIONS & ACTION ITEMS")
    print("=" * 70)
    
    print("\n    A. DATA QUALITY ISSUES:")
    
    # Missing values
    if len(missing_df) > 0:
        high_missing = missing_df[missing_df['Missing %'] > 30]
        if len(high_missing) > 0:
            print("    WARNING: HIGH MISSING VALUES (>30%):")
            for col in high_missing.index:
                print(f"        - {col}: {missing_df.loc[col, 'Missing %']}% missing")
            print("        -> Consider: Dropping column OR imputing with domain knowledge")
    
    # Outliers
    high_outlier_cols = [col for col, vals in outlier_summary.items() if vals['pct'] > 5]
    if high_outlier_cols:
        print("    WARNING: HIGH OUTLIER PERCENTAGE (>5%):")
        for col in high_outlier_cols:
            print(f"        - {col}: {outlier_summary[col]['pct']:.2f}% outliers")
        print("        -> Consider: Capping (winsorization) OR investigating source")
    
    print("\n    B. FEATURE ENGINEERING SUGGESTIONS:")
    
    # Low correlation features
    low_corr = [col for col, corr in correlations.items() if abs(corr) < 0.1]
    if low_corr:
        print("    INFO: LOW CORRELATION WITH PRICE:")
        for col in low_corr:
            print(f"        - {col}: {correlations[col]:.4f}")
        print("        -> Consider: Creating interaction features OR removing")
    
    # High correlation features
    high_corr = [col for col, corr in correlations.items() if abs(corr) > 0.5]
    if high_corr:
        print("    SUCCESS: HIGH CORRELATION WITH PRICE:")
        for col in high_corr:
            print(f"        - {col}: {correlations[col]:.4f}")
        print("        -> Good predictors, consider for model features")
    
    print("\n    C. DATA BALANCE:")
    
    # Price distribution balance
    prices = df['Price'].dropna()
    low_price = (prices < 10).sum()
    high_price = (prices >= 20).sum()
    total = len(prices)
    
    print(f"    Price distribution:")
    print(f"        Low price (<10 ty):    {low_price:,} ({low_price/total*100:.1f}%)")
    print(f"        Medium (10-20 ty):    {((prices >= 10) & (prices < 20)).sum():,} ({((prices >= 10) & (prices < 20)).sum()/total*100:.1f}%)")
    print(f"        High price (>=20 ty):  {high_price:,} ({high_price/total*100:.1f}%)")
    
    if high_price / total < 0.05:
        print("    WARNING: HIGH-VALUE DATA UNDERREPRESENTED!")
        print("        -> Consider: Collecting more high-value property data")
    
    print("\n" + "=" * 70)
    print("                    EVALUATION COMPLETE")
    print("=" * 70)

def create_visualizations(df):
    """Create visualization plots"""
    print("\n    Creating visualization plots...")
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # 1. Price Distribution
        ax1 = axes[0, 0]
        prices = df['Price'].dropna()
        ax1.hist(prices, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Price (ty)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Price Distribution')
        ax1.axvline(prices.median(), color='red', linestyle='--', label=f'Median: {prices.median():.2f}')
        ax1.legend()
        
        # 2. Log Price Distribution
        ax2 = axes[0, 1]
        log_prices = np.log1p(prices)
        ax2.hist(log_prices, bins=50, color='coral', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Log(Price + 1)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Log-Transformed Price Distribution')
        
        # 3. Area vs Price
        ax3 = axes[0, 2]
        df_clean = df[['Area', 'Price']].dropna()
        ax3.scatter(df_clean['Area'], df_clean['Price'], alpha=0.3, s=5)
        ax3.set_xlabel('Area (m2)')
        ax3.set_ylabel('Price (ty)')
        ax3.set_title('Area vs Price')
        
        # 4. Price by City
        ax4 = axes[1, 0]
        df['City'] = df['Address'].str.split(',').str[-1].str.strip()
        city_prices = df.groupby('City')['Price'].median().sort_values(ascending=False)
        city_prices.plot(kind='bar', ax=ax4, color=['steelblue', 'coral', 'green'][:len(city_prices)])
        ax4.set_xlabel('City')
        ax4.set_ylabel('Median Price (ty)')
        ax4.set_title('Median Price by City')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Missing Values Heatmap
        ax5 = axes[1, 1]
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
        missing_pct = missing_pct[missing_pct > 0]
        if len(missing_pct) > 0:
            ax5.barh(missing_pct.index, missing_pct.values, color='orange')
        ax5.set_xlabel('Missing %')
        ax5.set_title('Missing Values by Feature')
        
        # 6. Box Plot by District (HCM)
        ax6 = axes[1, 2]
        df['District'] = df['Address'].str.split(',').str[-2].str.strip()
        hcm_df = df[df['City'].str.contains('HCM|Ho Chi', case=False, na=False)]
        top_districts = hcm_df.groupby('District')['Price'].median().nlargest(8).index
        box_data = [hcm_df[hcm_df['District'] == d]['Price'].values for d in top_districts]
        ax6.boxplot(box_data, labels=top_districts)
        ax6.set_xlabel('District')
        ax6.set_ylabel('Price (ty)')
        ax6.set_title('Price Distribution by Top Districts (HCM)')
        ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('data_evaluation_plots.png', dpi=150, bbox_inches='tight')
        print(f"    Plots saved to: data_evaluation_plots.png")
        
    except Exception as e:
        print(f"    Warning: Could not create plots: {e}")

def main():
    # Load and explore data
    df = load_and_explore_data()
    
    # Evaluate missing values
    missing_df = evaluate_missing_values(df)
    
    # Evaluate duplicates
    duplicates = evaluate_duplicates(df)
    
    # Evaluate price distribution
    prices = evaluate_price_distribution(df)
    
    # Evaluate outliers
    outlier_summary = evaluate_outliers(df)
    
    # Evaluate district analysis
    df = evaluate_district_analysis(df)
    
    # Evaluate feature quality
    correlations = evaluate_feature_quality(df)
    
    # Generate recommendations
    generate_recommendations(df, missing_df, outlier_summary, correlations)
    
    # Create visualizations
    create_visualizations(df)

if __name__ == "__main__":
    main()
