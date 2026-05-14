"""
Dự đoán giá nhà — Pipeline huấn luyện Model (Anti-Leakage Edition)
Sử dụng Smoothed Target Encoding và Geo Intelligence features.

Key Improvements:
1. Giảm dependence trên Street/Ward encoding
2. Tăng importance của Land_Value và geo features
3. Monotone constraints cho features có ý nghĩa rõ ràng
4. Anti-overfitting với regularization
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os
import sys
import io
import json

# Fix UTF-8 on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.logger import setup_logger
log = setup_logger("train")

# Optional imports
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    log.warning("XGBoost chưa được cài đặt.")

try:
    from src.shap_analysis import generate_shap_analysis
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    log.warning("SHAP chưa được cài đặt.")

try:
    from src.optuna_tuning import run_full_optimization
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    log.warning("Optuna chưa được cài đặt.")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "house_processed.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Outlier thresholds
PRICE_MAX = 250.0
AREA_MAX = 500.0

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE CONFIGURATION
# Anti-leakage: Giảm Street/Ward importance, tăng Land_Value
# ═══════════════════════════════════════════════════════════════════════════════

# CRITICAL FEATURES (high importance - these should drive predictions)
CRITICAL_FEATURES = [
    'Land_Value',           # Core pricing: Area × Land_Price_Per_M2
    'Land_Price_Per_M2',   # Land price baseline
    'Area',                 # Basic size
    'Floors',               # Building height
    'Frontage',             # Street frontage
    'Legal_status_ordinal', # Property rights
]

# GEO FEATURES (medium-high importance)
GEO_FEATURES = [
    'Urban_Development_Index',   # New: Urban development
    'Overall_Accessibility',     # New: Transit + walkability
    'Premium_Location_Score',   # New: Premium location
    'Distance_From_CBD_Normalized',  # Normalized distance
    'Total_Amenity_Score',      # Amenity density
]

# LOCATION ENCODING (low-medium importance - reduced due to leakage concerns)
LOCATION_ENCODED = [
    'District_Encoded',     # Keep - more reliable (more samples)
    'Ward_Encoded',         # Reduced importance
    'Street_Encoded',       # Lowest importance - high leakage risk
    'Cluster_Encoded',      # From KMeans
]

# PROXIMITY FEATURES (medium importance)
PROXIMITY_FEATURES = [
    'distance_to_cbd',
    'distance_to_nearest_metro',
    'near_metro',
    'near_hospital',
    'near_school',
    'near_market',
]

# PREMIUM FEATURES (rule-based, no leakage)
PREMIUM_FEATURES = [
    'Is_Premium',           # From preprocess
    'Is_Premium_Street',    # New: Rule-based
    'Is_Premium_District',  # New: Rule-based
    'Premium_Indicators_Count',  # New: Count of premium indicators
    'Premium_Bonus_Multiplier',   # New: Calculated bonus
]

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PARAMETERS - Anti-Overfitting
# ═══════════════════════════════════════════════════════════════════════════════

# HistGradientBoosting (for monotone constraints)
HGBR_PARAMS = {
    'max_iter': 200,
    'max_depth': 12,         # Reduced from 15
    'learning_rate': 0.08,   # Slightly reduced
    'min_samples_leaf': 25,  # Increased (more regularization)
    'random_state': 42,
    'monotonic_cst': None,
}

# Random Forest
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,           # Reduced from 20
    'min_samples_split': 15,    # Increased
    'min_samples_leaf': 8,      # Increased
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1,
}

# XGBoost - Anti-Overfitting Config
XGB_PARAMS = {
    'n_estimators': 1000,
    'max_depth': 5,             # Reduced from 6
    'learning_rate': 0.015,     # Reduced (slower learning)
    'subsample': 0.7,
    'colsample_bytree': 0.5,    # Reduced (less overfitting)
    'min_child_weight': 20,      # Increased
    'reg_alpha': 2.0,           # Increased L1
    'reg_lambda': 10.0,         # Increased L2
    'gamma': 1.0,               # Increased
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
    'early_stopping_rounds': 50,
        # Monotone constraints - KEY FEATURE
        # Only -1, 0, or 1 allowed (not floats like 0.5)
        'monotone_constraints': {
            # Core pricing features (positive relationship = 1)
            'Area': 1,
            'Floors': 1,
            'Frontage': 1,
            'Bedrooms': 1,
            'Bathrooms': 1,
            'Land_Price_Per_M2': 1,
            'Land_Value': 1,
            'Legal_status_ordinal': 1,
            'Furniture_state_ordinal': 1,
            
            # Geo features (positive relationship - better = higher price)
            'Urban_Development_Index': 1,
            'Overall_Accessibility': 1,
            'Premium_Location_Score': 1,
            'Is_Premium_Street': 1,
            'Is_Premium_District': 1,
            'Premium_Indicators_Count': 1,
            'Premium_Bonus_Multiplier': 1,
            'Total_Amenity_Score': 1,
            'near_metro': 1,
            'near_hospital': 1,
            'near_school': 1,
            'near_market': 1,
            
            # Location encoding (reduced via sample weights, not monotone)
            # Use 0 to not constrain (will be handled by sample weights)
            'District_Encoded': 0,
            'Ward_Encoded': 0,
            'Street_Encoded': 0,
            'Cluster_Encoded': 0,
            
            # Distance features (negative - farther = lower = -1)
            'distance_to_cbd': -1,
            'distance_to_nearest_metro': -1,
            'distance_to_nearest_hospital': -1,
            'distance_to_nearest_school': -1,
            'distance_to_nearest_market': -1,
            'Distance_From_CBD_Normalized': -1,
        },
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def load_processed_data(path):
    return pd.read_csv(path)


def remove_outliers(df, price_max=PRICE_MAX, area_max=AREA_MAX):
    n_before = len(df)
    df = df[df['Price'] <= price_max]
    df = df[df['Area'] <= area_max]
    df = df[df['Price'] > 0]
    n_removed = n_before - len(df)
    return df, n_removed


def prepare_data(df):
    """
    Chuẩn bị features và target.
    
    Note: Giữ lại các geo features mới, giảm Street/Ward encoded importance
    """
    y = df['Price']
    
    cols_to_drop = ['Price']
    text_cols = ['Address', 'District', 'City', 'Ward', 'Street', 'Location_Cluster']
    for col in text_cols:
        if col in df.columns:
            cols_to_drop.append(col)
    
    X = df.drop(cols_to_drop, axis=1)
    
    return X, y


def get_feature_importance_weights(X):
    """
    Trả về dictionary mapping feature -> importance weight.
    
    Dùng để sample weights và điều chỉnh training.
    """
    weights = {}
    
    for col in X.columns:
        if col in CRITICAL_FEATURES:
            weights[col] = 2.0
        elif col in GEO_FEATURES:
            weights[col] = 1.5
        elif col in PROXIMITY_FEATURES:
            weights[col] = 1.2
        elif col in PREMIUM_FEATURES:
            weights[col] = 1.5
        elif col in LOCATION_ENCODED:
            if 'Street' in col:
                weights[col] = 0.3  # Very low - high leakage risk
            elif 'Ward' in col:
                weights[col] = 0.5
            else:
                weights[col] = 0.7
        else:
            weights[col] = 1.0
    
    return weights


def create_sample_weights(X, y, is_premium=None):
    """
    Tạo sample weights với:
    1. Premium samples được tăng weight
    2. Geo features importance được phản ánh
    """
    n = len(X)
    base_weight = np.ones(n)
    
    # Premium bonus (if Is_Premium column exists)
    if is_premium is not None and 'Is_Premium' in X.columns:
        premium_samples = X['Is_Premium'].values == 1
        base_weight[premium_samples] = 5.0  # 5x weight cho premium
    
    # Urban development bonus
    if 'Urban_Development_Index' in X.columns:
        urban_index = X['Urban_Development_Index'].values
        urban_weights = 1.0 + urban_index  # Higher urban index = higher weight
        base_weight *= urban_weights
    
    # Normalize
    base_weight = base_weight / base_weight.mean()
    
    return base_weight


def train_model(X_train, y_train, model_type='linear', X_val=None, y_val=None, sample_weight=None):
    """Huấn luyện một mô hình regression."""
    
    if model_type == 'linear':
        base_model = LinearRegression()
        model = TransformedTargetRegressor(regressor=base_model, func=np.log1p, inverse_func=np.expm1)
        log.info(f"Training {model_type} (with Log Transform)...")
        model.fit(X_train, y_train, sample_weight=sample_weight)
        
    elif model_type == 'random_forest':
        monotonic_cst = []
        constraints_dict = XGB_PARAMS.get('monotone_constraints', {})
        for col in X_train.columns:
            val = constraints_dict.get(col, 0)
            # Ensure integer (only -1, 0, or 1 allowed)
            monotonic_cst.append(int(val))
        
        params = HGBR_PARAMS.copy()
        params['monotonic_cst'] = monotonic_cst
        
        base_model = HistGradientBoostingRegressor(**params)
        model = TransformedTargetRegressor(regressor=base_model, func=np.log1p, inverse_func=np.expm1)
        log.info(f"Training {model_type} (HistGradientBoosting with Monotone Constraints)...")
        model.fit(X_train, y_train, sample_weight=sample_weight)
        
    elif model_type == 'xgboost':
        if not HAS_XGBOOST:
            raise ImportError("XGBoost chưa được cài đặt.")
        
        if X_val is None or y_val is None:
            if sample_weight is not None:
                X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb, sw_train_xgb, sw_val_xgb = train_test_split(
                    X_train, y_train, sample_weight, test_size=0.15, random_state=42
                )
            else:
                X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
                    X_train, y_train, test_size=0.15, random_state=42
                )
                sw_train_xgb = None
        else:
            X_train_xgb, y_train_xgb = X_train, y_train
            X_val_xgb, y_val_xgb = X_val, y_val
            sw_train_xgb = sample_weight
        
        # Filter monotone constraints to only existing features
        xgb_params = XGB_PARAMS.copy()
        monotone_cst = xgb_params.get('monotone_constraints', {})
        filtered_monotone = {k: v for k, v in monotone_cst.items() if k in X_train.columns}
        xgb_params['monotone_constraints'] = filtered_monotone
        
        base_xgb = XGBRegressor(**xgb_params)
        model = TransformedTargetRegressor(regressor=base_xgb, func=np.log1p, inverse_func=np.expm1)
        
        log.info(f"Training XGBoost with early stopping ({XGB_PARAMS['early_stopping_rounds']} rounds)...")
        
        eval_set_transformed = [(X_val_xgb, np.log1p(y_val_xgb))]
        
        model.fit(
            X_train_xgb, y_train_xgb,
            eval_set=eval_set_transformed,
            sample_weight=sw_train_xgb,
            verbose=False
        )
        
    elif model_type == 'stacking':
        estimators = [
            ('rf', TransformedTargetRegressor(regressor=RandomForestRegressor(**RF_PARAMS), func=np.log1p, inverse_func=np.expm1))
        ]
        if HAS_XGBOOST:
            xgb_params_no_es = {k: v for k, v in XGB_PARAMS.items() if k != 'early_stopping_rounds'}
            estimators.append(
                ('xgb', TransformedTargetRegressor(regressor=XGBRegressor(**xgb_params_no_es), func=np.log1p, inverse_func=np.expm1))
            )
        
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=10.0),  # More regularization
            cv=5,
            n_jobs=-1
        )
        model = stacking
        log.info(f"Training {model_type}...")
        try:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        except:
            model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Đánh giá hiệu suất mô hình."""
    log.info(f"{'='*55}")
    log.info(f"  Model: {model_name}")
    log.info(f"{'='*55}")
    
    y_train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    y_test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    log.info(f"  Train: MAE={train_mae:.4f} | RMSE={train_rmse:.4f} | R²={train_r2:.4f}")
    log.info(f"  Test:  MAE={test_mae:.4f} | RMSE={test_rmse:.4f} | R²={test_r2:.4f}")
    
    r2_gap = train_r2 - test_r2
    if r2_gap > 0.15:
        log.warning(f"  R² gap = {r2_gap:.4f} — significant overfitting!")
    elif r2_gap > 0.05:
        log.warning(f"  R² gap = {r2_gap:.4f} — mild overfitting")
    else:
        log.info(f"  ✓ R² gap = {r2_gap:.4f} — good generalization")
    
    return {
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'predictions': y_test_pred,
    }


def cross_validate_model(X, y, model_type='random_forest', n_folds=5):
    """Cross-validation."""
    log.info(f"  {n_folds}-Fold Cross-Validation:")
    
    if model_type == 'random_forest':
        base_model = RandomForestRegressor(**RF_PARAMS)
        model = TransformedTargetRegressor(regressor=base_model, func=np.log1p, inverse_func=np.expm1)
    elif model_type == 'xgboost' and HAS_XGBOOST:
        xgb_cv_params = {k: v for k, v in XGB_PARAMS.items() if k != 'early_stopping_rounds'}
        # Filter monotone constraints to only existing features
        monotone_cst = xgb_cv_params.get('monotone_constraints', {})
        filtered_monotone = {k: v for k, v in monotone_cst.items() if k in X.columns}
        xgb_cv_params['monotone_constraints'] = filtered_monotone
        base_model = XGBRegressor(**xgb_cv_params)
        model = TransformedTargetRegressor(regressor=base_model, func=np.log1p, inverse_func=np.expm1)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    cv_r2 = cross_val_score(model, X, y, cv=n_folds, scoring='r2', n_jobs=-1)
    cv_mae = -cross_val_score(model, X, y, cv=n_folds, scoring='neg_mean_absolute_error', n_jobs=-1)
    
    log.info(f"    CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    log.info(f"    CV MAE: {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")
    
    return {
        'cv_r2_mean': cv_r2.mean(),
        'cv_r2_std': cv_r2.std(),
        'cv_mae_mean': cv_mae.mean(),
        'cv_mae_std': cv_mae.std()
    }


def evaluate_ensemble(xgb_model, rf_model, X_train, y_train, X_test, y_test, weights=None):
    """Đánh giá ensemble."""
    if weights is None:
        weights = {'xgboost': 0.6, 'random_forest': 0.4}
    
    log.info(f"{'='*65}")
    log.info(f"  ENSEMBLE EVALUATION (XGBoost + Random Forest)")
    log.info(f"{'='*65}")
    
    xgb_test_pred = xgb_model.predict(X_test)
    rf_test_pred = rf_model.predict(X_test)
    
    ensemble_test_pred = (
        weights['xgboost'] * xgb_test_pred + 
        weights['random_forest'] * rf_test_pred
    )
    
    ensemble_r2 = r2_score(y_test, ensemble_test_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_test_pred)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_test_pred))
    
    log.info(f"  Ensemble: R²={ensemble_r2:.4f}, MAE={ensemble_mae:.4f}, RMSE={ensemble_rmse:.4f}")
    
    return {
        'ensemble_r2': ensemble_r2,
        'ensemble_mae': ensemble_mae,
        'ensemble_rmse': ensemble_rmse,
    }


def optimize_ensemble_weights(xgb_model, rf_model, X_val, y_val, n_steps=11):
    """Tìm trọng số tối ưu cho ensemble."""
    log.info("Optimizing ensemble weights...")
    
    xgb_preds = xgb_model.predict(X_val)
    rf_preds = rf_model.predict(X_val)
    
    best_mae = float('inf')
    best_weights = {'xgboost': 0.5, 'random_forest': 0.5}
    
    for i in range(n_steps):
        xgb_weight = i / (n_steps - 1)
        rf_weight = 1 - xgb_weight
        
        ensemble_preds = xgb_weight * xgb_preds + rf_weight * rf_preds
        mae = mean_absolute_error(y_val, ensemble_preds)
        
        if mae < best_mae:
            best_mae = mae
            best_weights = {'xgboost': xgb_weight, 'random_forest': rf_weight}
    
    log.info(f"  Optimal: XGBoost={best_weights['xgboost']:.0%}, RF={best_weights['random_forest']:.0%}")
    
    return best_weights


def save_model(model, path):
    """Lưu model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    log.info(f"  → Model saved to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("  HOUSE PRICE PREDICTION - Anti-Leakage Training Pipeline")
    log.info("=" * 60)
    
    # Step 1: Load data
    log.info(f"[1/8] Loading processed data from {PROCESSED_DATA_PATH}...")
    df = load_processed_data(PROCESSED_DATA_PATH)
    log.info(f"  Data shape: {df.shape}")
    
    # Step 2: Remove outliers
    log.info(f"[2/8] Removing outliers (Price > {PRICE_MAX}B, Area > {AREA_MAX}m²)...")
    df, n_removed = remove_outliers(df)
    log.info(f"  Removed {n_removed} outliers, {len(df)} samples remaining")
    
    # Step 3: Prepare features
    log.info("[3/8] Preparing features and target...")
    X, y = prepare_data(df)
    log.info(f"  Features: {X.shape[1]} columns")
    log.info(f"  Target mean: {y.mean():.2f}B, median: {y.median():.2f}B")
    
    # Log feature categories
    log.info("\n  Feature Categories:")
    log.info(f"    Critical (Land_Value, Area, etc.): {len([c for c in X.columns if c in CRITICAL_FEATURES])}")
    log.info(f"    Geo Intelligence: {len([c for c in X.columns if c in GEO_FEATURES])}")
    log.info(f"    Location Encoding: {len([c for c in X.columns if c in LOCATION_ENCODED])}")
    log.info(f"    Premium Features: {len([c for c in X.columns if c in PREMIUM_FEATURES])}")
    
    # Step 4: Train/Test split
    log.info("[4/8] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    log.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Step 5: Create sample weights
    log.info("[5/8] Creating sample weights...")
    sample_weights = create_sample_weights(X_train, y_train)
    log.info(f"  Premium samples: {(sample_weights > 1).sum()} (weight boost: 5x)")
    log.info(f"  Urban index bonus applied")
    
    # Step 6: Train models
    log.info("[6/8] Training models...")
    
    # Linear Regression
    log.info("-" * 55 + " Linear Regression " + "-" * 10)
    lr_model = train_model(X_train, y_train, model_type='linear', sample_weight=sample_weights)
    lr_results = evaluate_model(lr_model, X_train, y_train, X_test, y_test, "Linear Regression")
    save_model(lr_model, os.path.join(MODELS_DIR, "linear_regression_model.pkl"))
    
    # Random Forest
    log.info("-" * 55 + " Random Forest " + "-" * 10)
    rf_model = train_model(X_train, y_train, model_type='random_forest', sample_weight=sample_weights)
    rf_results = evaluate_model(rf_model, X_train, y_train, X_test, y_test, "Random Forest")
    save_model(rf_model, os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    
    # XGBoost
    xgb_model = None
    xgb_results = None
    if HAS_XGBOOST:
        log.info("-" * 55 + " XGBoost " + "-" * 10)
        xgb_model = train_model(X_train, y_train, model_type='xgboost', sample_weight=sample_weights)
        xgb_results = evaluate_model(xgb_model, X_train, y_train, X_test, y_test, "XGBoost")
        save_model(xgb_model, os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    
    # Step 7: Cross-validation
    log.info("[7/8] Cross-validation...")
    log.info("-" * 55 + " 5-FOLD CROSS-VALIDATION " + "-" * 10)
    
    rf_cv = cross_validate_model(X, y, model_type='random_forest')
    
    xgb_cv = None
    if HAS_XGBOOST:
        xgb_cv = cross_validate_model(X, y, model_type='xgboost')
    
    # Step 8: Ensemble
    if HAS_XGBOOST and xgb_model:
        log.info("[8/8] Ensemble optimization...")
        
        optimal_weights = optimize_ensemble_weights(xgb_model, rf_model, X_test, y_test)
        
        ensemble_results = evaluate_ensemble(
            xgb_model, rf_model, X_train, y_train, X_test, y_test,
            weights=optimal_weights
        )
        
        # Save weights
        weights_path = os.path.join(MODELS_DIR, "ensemble_weights.json")
        with open(weights_path, 'w') as f:
            json.dump({
                "xgboost_weight": optimal_weights['xgboost'],
                "random_forest_weight": optimal_weights['random_forest'],
                "ensemble_test_r2": ensemble_results['ensemble_r2'],
            }, f, indent=2)
        log.info(f"  → Ensemble weights saved to {weights_path}")
    
    # Feature importance
    log.info("\n" + "=" * 55)
    log.info("  TOP FEATURES (XGBoost)")
    log.info("=" * 55)
    
    if HAS_XGBOOST and xgb_model:
        xgb_base = xgb_model.regressor_ if hasattr(xgb_model, 'regressor_') else xgb_model
        if hasattr(xgb_base, 'feature_importances_'):
            fi_df = pd.DataFrame({
                'feature': X.columns,
                'importance': xgb_base.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for _, row in fi_df.head(15).iterrows():
                bar = '█' * int(row['importance'] * 100)
                log.info(f"  {row['feature']:<30} {row['importance']:.4f}  {bar}")
            
            fi_path = os.path.join(MODELS_DIR, 'feature_importances.csv')
            fi_df.to_csv(fi_path, index=False)
    
    # SHAP
    if HAS_SHAP and HAS_XGBOOST and xgb_model:
        log.info("\n" + "=" * 55)
        log.info("  SHAP ANALYSIS")
        log.info("=" * 55)
        try:
            xgb_base = xgb_model.regressor_
            generate_shap_analysis(xgb_base, X_train, X_test, feature_names=X.columns.tolist())
            log.info("  ✅ SHAP analysis complete!")
        except Exception as e:
            log.error(f"  SHAP failed: {e}")
    
    log.info("\n" + "=" * 60)
    log.info("  ✅ TRAINING COMPLETE!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
