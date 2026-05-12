"""
Dự đoán giá nhà — Pipeline huấn luyện Model
Huấn luyện các mô hình Linear Regression, Random Forest và XGBoost trên dữ liệu đã tiền xử lý.
Sử dụng pipeline đã được sửa để tránh data leakage.
Hỗ trợ Ensemble Evaluation (XGBoost + Random Forest).
Nâng cấp: Logging, SHAP Explainability, Optuna Hyperparameter Tuning.
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

# Thêm thư mục gốc vào sys.path để Python có thể tìm thấy module 'src'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ─── Logging ──────────────────────────────────────────────────────────────
from src.logger import setup_logger
log = setup_logger("train")

# Thử import XGBoost (tùy chọn)
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    log.warning("XGBoost chưa được cài đặt. Chạy 'pip install xgboost' để bật.")

# Thử import SHAP (tùy chọn)
try:
    from src.shap_analysis import generate_shap_analysis
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    log.warning("SHAP chưa được cài đặt. Chạy 'pip install shap' để bật.")

# Thử import Optuna (tùy chọn)
try:
    from src.optuna_tuning import run_full_optimization
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    log.warning("Optuna chưa được cài đặt. Chạy 'pip install optuna' để bật.")


# ─── Cấu hình ────────────────────────────────────────────────────────────
# Đường dẫn tuyệt đối để chạy từ bất kỳ thư mục nào
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "house_processed.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ngưỡng loại bỏ outliers
PRICE_MAX = 250.0       # Loại bỏ bất động sản > 250 tỷ VNĐ
AREA_MAX = 500.0        # Loại bỏ bất động sản > 500 m²

# Tham số HistGradientBoostingRegressor (Thay thế RF để hỗ trợ monotone constraints)
HGBR_PARAMS = {
    'max_iter': 200,
    'max_depth': 15,
    'learning_rate': 0.1,
    'min_samples_leaf': 20,
    'random_state': 42,
    'monotonic_cst': None, # Sẽ được gán động dựa trên danh sách features
}

# Tham số Random Forest (Giữ lại để tham khảo hoặc dùng nếu không cần monotone)
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1,
}

# Tham số XGBoost (Tối ưu cho giảm overfitting)
XGB_PARAMS = {
    'n_estimators': 1000,          # Nhiều trees, sẽ dừng sớm bằng early stopping
    'max_depth': 6,                # Giảm từ 8 → 6 (ít overfitting hơn)
    'learning_rate': 0.02,         # Giảm từ 0.05 → 0.02 (học chậm hơn)
    'subsample': 0.7,              # Giảm từ 0.8 → 0.7 (giảm overfitting)
    'colsample_bytree': 0.6,       # Giảm từ 0.8 → 0.6 (ít features mỗi cây)
    'min_child_weight': 15,         # Tăng từ 5 → 15 (ít leaf nhỏ hơn)
    'reg_alpha': 1.0,               # Tăng L1 regularization từ 0.1 → 1.0
    'reg_lambda': 5.0,             # Tăng L2 regularization từ 1.0 → 5.0
    'gamma': 0.5,                  # Thêm: min loss reduction để split
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
    'early_stopping_rounds': 50,   # Dừng nếu 50 rounds không cải thiện
    'monotone_constraints': {
        'Area': 1, 
        'Floors': 1, 
        'Legal_status_ordinal': 1, 
        'Furniture_state_ordinal': 1, 
        'District_Encoded': 1
    },
}


# ─── Hàm hỗ trợ ─────────────────────────────────────────────────────────

def load_processed_data(path):
    """Tải dữ liệu đã xử lý từ CSV."""
    return pd.read_csv(path)


def remove_outliers(df, price_max=PRICE_MAX, area_max=AREA_MAX):
    """
    Loại bỏ các outliers cực đoan làm sai lệch mô hình.

    Trả về:
        DataFrame đã lọc và số hàng đã loại bỏ.
    """
    n_before = len(df)

    # Loại bỏ các giá cực đoan
    df = df[df['Price'] <= price_max]

    # Loại bỏ các diện tích cực đoan
    df = df[df['Area'] <= area_max]

    # Loại bỏ giá bằng 0 hoặc âm
    df = df[df['Price'] > 0]

    n_removed = n_before - len(df)
    return df, n_removed


def prepare_data(df):
    """
    Chuẩn bị features và target cho huấn luyện.

    Dữ liệu đã xử lý có District_Encoded và City_Encoded từ Target Encoding.
    Các cột text Address, District, City đã được xóa bởi preprocess.py.
    """
    # Cột target
    y = df['Price']

    # Xóa target và các cột text còn lại không nên là features
    cols_to_drop = ['Price']
    for col in ['Address', 'District', 'City']:
        if col in df.columns:
            cols_to_drop.append(col)

    X = df.drop(cols_to_drop, axis=1)

    return X, y


def train_model(X_train, y_train, model_type='linear', X_val=None, y_val=None):
    """Huấn luyện một mô hình regression có áp dụng Log Transform."""
    if model_type == 'linear':
        base_model = LinearRegression()
        model = TransformedTargetRegressor(regressor=base_model, func=np.log1p, inverse_func=np.expm1)
        log.info(f"Đang huấn luyện mô hình {model_type} (với Log Transform)...")
        model.fit(X_train, y_train)
        log.info(f"  ✓ Mô hình {model_type} đã được huấn luyện thành công!")

    elif model_type == 'random_forest':
        # Sử dụng HistGradientBoostingRegressor để áp dụng monotone constraints
        # Tạo mảng monotonic_cst dựa trên vị trí cột
        monotonic_cst = []
        constraints_dict = XGB_PARAMS.get('monotone_constraints', {})
        for col in X_train.columns:
            monotonic_cst.append(constraints_dict.get(col, 0))
        
        params = HGBR_PARAMS.copy()
        params['monotonic_cst'] = monotonic_cst
        
        base_model = HistGradientBoostingRegressor(**params)
        model = TransformedTargetRegressor(regressor=base_model, func=np.log1p, inverse_func=np.expm1)
        log.info(f"Đang huấn luyện mô hình {model_type} (Sử dụng HistGradientBoosting với Monotone Constraints)...")
        model.fit(X_train, y_train)
        log.info(f"  ✓ Mô hình {model_type} đã được huấn luyện thành công!")

    elif model_type == 'xgboost':
        if not HAS_XGBOOST:
            raise ImportError("XGBoost chưa được cài đặt. Chạy 'pip install xgboost' trước.")

        # Tách validation set từ training data nếu không có
        if X_val is None or y_val is None:
            X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
                X_train, y_train, test_size=0.15, random_state=42
            )
            log.info(f"  Tách validation set: {len(X_val_xgb)} mẫu (15% của train)")
        else:
            X_train_xgb, y_train_xgb = X_train, y_train
            X_val_xgb, y_val_xgb = X_val, y_val

        # Cấu hình XGBoost - early_stopping_rounds đã có trong XGB_PARAMS
        base_xgb = XGBRegressor(**XGB_PARAMS)
        model = TransformedTargetRegressor(regressor=base_xgb, func=np.log1p, inverse_func=np.expm1)
        
        log.info(f"Đang huấn luyện XGBoost với early stopping ({XGB_PARAMS['early_stopping_rounds']} rounds)...")

        # Lưu ý: cần truyền target của eval_set ở dạng Log Transform
        eval_set_transformed = [(X_val_xgb, np.log1p(y_val_xgb))]
        model.fit(
            X_train_xgb, y_train_xgb,
            eval_set=eval_set_transformed,
            verbose=False
        )

        log.info(f"  ✓ XGBoost huấn luyện xong!")
    elif model_type == 'stacking':
        # Stacking Ensemble
        estimators = [
            ('rf', TransformedTargetRegressor(regressor=RandomForestRegressor(**RF_PARAMS), func=np.log1p, inverse_func=np.expm1))
        ]
        if HAS_XGBOOST:
            # Note: No early stopping for CV inside stacking
            xgb_params_no_es = {k: v for k, v in XGB_PARAMS.items() if k != 'early_stopping_rounds'}
            estimators.append(
                ('xgb', TransformedTargetRegressor(regressor=XGBRegressor(**xgb_params_no_es), func=np.log1p, inverse_func=np.expm1))
            )
            
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(),
            cv=5,
            n_jobs=-1
        )
        model = stacking
        log.info(f"Đang huấn luyện mô hình {model_type} (Ridge Meta-model)...")
        model.fit(X_train, y_train)
        log.info(f"  ✓ Mô hình {model_type} đã được huấn luyện thành công!")
    else:
        raise ValueError(f"Kiểu model không xác định: {model_type}")

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Đánh giá hiệu suất mô hình trên tập train và test."""
    log.info(f"{'='*55}")
    log.info(f"  Mô hình: {model_name}")
    log.info(f"{'='*55}")

    # Dự đoán trên tập train
    y_train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    # Dự đoán trên tập test
    y_test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    # In kết quả
    log.info(f"  Tập Train:  MAE={train_mae:.4f} | RMSE={train_rmse:.4f} | R²={train_r2:.4f}")
    log.info(f"  Tập Test:   MAE={test_mae:.4f} | RMSE={test_rmse:.4f} | R²={test_r2:.4f}")

    # Kiểm tra overfitting
    r2_gap = train_r2 - test_r2
    if r2_gap > 0.15:
        log.warning(f"  R² gap = {r2_gap:.4f} — overfitting đáng kể!")
    elif r2_gap > 0.05:
        log.warning(f"  R² gap = {r2_gap:.4f} — overfitting nhẹ")
    else:
        log.info(f"  ✓ R² gap = {r2_gap:.4f} — khả năng tổng quát hóa tốt")

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
    """
    Thực hiện cross-validation với proper stratified folds.
    Sử dụng model không có early stopping cho CV.
    """
    log.info(f"  {n_folds}-Fold Cross-Validation:")
    
    if model_type == 'random_forest':
        base_model = RandomForestRegressor(**RF_PARAMS)
        model = TransformedTargetRegressor(regressor=base_model, func=np.log1p, inverse_func=np.expm1)
    elif model_type == 'xgboost' and HAS_XGBOOST:
        xgb_cv_params = {k: v for k, v in XGB_PARAMS.items() if k != 'early_stopping_rounds'}
        base_model = XGBRegressor(**xgb_cv_params)
        model = TransformedTargetRegressor(regressor=base_model, func=np.log1p, inverse_func=np.expm1)
    elif model_type == 'stacking':
        estimators = [
            ('rf', TransformedTargetRegressor(regressor=RandomForestRegressor(**RF_PARAMS), func=np.log1p, inverse_func=np.expm1))
        ]
        if HAS_XGBOOST:
            xgb_cv_params = {k: v for k, v in XGB_PARAMS.items() if k != 'early_stopping_rounds'}
            estimators.append(('xgb', TransformedTargetRegressor(regressor=XGBRegressor(**xgb_cv_params), func=np.log1p, inverse_func=np.expm1)))
        model = StackingRegressor(estimators=estimators, final_estimator=Ridge(), cv=5, n_jobs=-1)
    else:
        raise ValueError(f"Unsupported model type for CV: {model_type}")
    
    cv_r2 = cross_val_score(model, X, y, cv=n_folds, scoring='r2', n_jobs=-1)
    cv_mae = -cross_val_score(model, X, y, cv=n_folds, scoring='neg_mean_absolute_error', n_jobs=-1)
    
    log.info(f"    CV R²:    {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    log.info(f"    CV MAE:   {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")
    
    return {
        'cv_r2_mean': cv_r2.mean(),
        'cv_r2_std': cv_r2.std(),
        'cv_mae_mean': cv_mae.mean(),
        'cv_mae_std': cv_mae.std()
    }


def evaluate_ensemble(xgb_model, rf_model, X_train, y_train, X_test, y_test, weights=None):
    """
    Đánh giá hiệu suất của Ensemble (XGBoost + Random Forest).
    
    Args:
        xgb_model: XGBoost model đã huấn luyện
        rf_model: Random Forest model đã huấn luyện
        X_train, y_train: Training data
        X_test, y_test: Test data
        weights: Dict chứa trọng số {'xgboost': 0.6, 'random_forest': 0.4}
    
    Returns:
        Dict chứa kết quả đánh giá của từng model và ensemble
    """
    if weights is None:
        weights = {'xgboost': 0.6, 'random_forest': 0.4}
    
    log.info(f"{'='*65}")
    log.info(f"  ENSEMBLE EVALUATION (XGBoost + Random Forest)")
    log.info(f"{'='*65}")
    log.info(f"  Weights: XGBoost = {weights['xgboost']:.0%}, RF = {weights['random_forest']:.0%}")
    
    # Predictions từ từng model
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_test_pred = xgb_model.predict(X_test)
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)
    
    # Ensemble predictions (weighted average)
    ensemble_train_pred = (
        weights['xgboost'] * xgb_train_pred + 
        weights['random_forest'] * rf_train_pred
    )
    ensemble_test_pred = (
        weights['xgboost'] * xgb_test_pred + 
        weights['random_forest'] * rf_test_pred
    )
    
    # Đánh giá từng model
    results = {}
    models_pred = {
        'XGBoost': (xgb_train_pred, xgb_test_pred),
        'Random Forest': (rf_train_pred, rf_test_pred),
        'Ensemble': (ensemble_train_pred, ensemble_test_pred),
    }
    
    log.info(f"  {'Model':<18} {'Train R²':<12} {'Test R²':<12} {'Test MAE':<12} {'Test RMSE':<12}")
    log.info(f"  {'-'*62}")
    
    for name, (train_pred, test_pred) in models_pred.items():
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        results[name.lower().replace(' ', '_')] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
        }
        
        log.info(f"  {name:<18} {train_r2:<12.4f} {test_r2:<12.4f} {test_mae:<12.4f} {test_rmse:<12.4f}")
    
    # So sánh ensemble với best single model
    log.info(f"  📊 ENSEMBLE IMPROVEMENT:")
    
    # Tìm best single model
    best_single = max(
        [('XGBoost', results['xgboost']), ('Random Forest', results['random_forest'])],
        key=lambda x: x[1]['test_r2']
    )
    
    ensemble_r2 = results['ensemble']['test_r2']
    ensemble_mae = results['ensemble']['test_mae']
    
    best_single_r2 = best_single[1]['test_r2']
    best_single_mae = best_single[1]['test_mae']
    
    r2_improvement = ensemble_r2 - best_single_r2
    mae_improvement = best_single_mae - ensemble_mae
    
    log.info(f"    Best Single Model: {best_single[0]} (R² = {best_single_r2:.4f})")
    log.info(f"    Ensemble R²:       {ensemble_r2:.4f} (Δ = {r2_improvement:+.4f})")
    log.info(f"    Ensemble MAE:      {ensemble_mae:.4f} (Δ = {mae_improvement:+.4f})")
    
    # Tính prediction spread (độ đồng thuận giữa 2 models)
    pred_diff = np.abs(xgb_test_pred - rf_test_pred)
    log.info(f"  📈 Prediction Agreement:")
    log.info(f"    Mean Abs Difference: {pred_diff.mean():.4f} tỷ")
    log.info(f"    Max Difference:     {pred_diff.max():.4f} tỷ")
    log.info(f"    Agreement Rate:    {(pred_diff < 0.5).mean()*100:.1f}% (diff < 0.5 tỷ)")
    
    # Cross-validation cho ensemble
    log.info(f"  {5}-Fold Cross-Validation for Ensemble:")
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_scores = []
    cv_mae_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train both models on fold
        xgb_fold = XGBRegressor(**{k: v for k, v in XGB_PARAMS.items() if k != 'early_stopping_rounds'})
        xgb_fold.fit(X_tr, y_tr)
        
        rf_fold = RandomForestRegressor(**RF_PARAMS)
        rf_fold.fit(X_tr, y_tr)
        
        # Ensemble prediction
        ens_pred = (
            weights['xgboost'] * xgb_fold.predict(X_val) +
            weights['random_forest'] * rf_fold.predict(X_val)
        )
        
        cv_r2_scores.append(r2_score(y_val, ens_pred))
        cv_mae_scores.append(mean_absolute_error(y_val, ens_pred))
    
    cv_r2_mean = np.mean(cv_r2_scores)
    cv_r2_std = np.std(cv_r2_scores)
    cv_mae_mean = np.mean(cv_mae_scores)
    cv_mae_std = np.std(cv_mae_scores)
    
    log.info(f"    CV R²:  {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
    log.info(f"    CV MAE: {cv_mae_mean:.4f} ± {cv_mae_std:.4f}")
    
    results['ensemble']['cv_r2_mean'] = cv_r2_mean
    results['ensemble']['cv_r2_std'] = cv_r2_std
    results['ensemble']['cv_mae_mean'] = cv_mae_mean
    results['ensemble']['cv_mae_std'] = cv_mae_std
    
    return results


def optimize_ensemble_weights(xgb_model, rf_model, X_val, y_val, n_steps=11):
    """
    Tìm trọng số tối ưu cho ensemble bằng validation set.
    
    Args:
        xgb_model: XGBoost model
        rf_model: Random Forest model
        X_val, y_val: Validation data
        n_steps: Số bước thử (0%, 10%, 20%, ..., 100%)
    
    Returns:
        Dict chứa trọng số tối ưu và MAE tương ứng
    """
    log.info(f"{'='*65}")
    log.info(f"  OPTIMIZING ENSEMBLE WEIGHTS")
    log.info(f"{'='*65}")
    
    xgb_preds = xgb_model.predict(X_val)
    rf_preds = rf_model.predict(X_val)
    
    best_mae = float('inf')
    best_weights = {'xgboost': 0.5, 'random_forest': 0.5}
    
    log.info(f"  {'XGB Weight':<12} {'RF Weight':<12} {'MAE':<12} {'R²':<12}")
    log.info(f"  {'-'*48}")
    
    for i in range(n_steps):
        xgb_weight = i / (n_steps - 1)
        rf_weight = 1 - xgb_weight
        
        ensemble_preds = xgb_weight * xgb_preds + rf_weight * rf_preds
        mae = mean_absolute_error(y_val, ensemble_preds)
        r2 = r2_score(y_val, ensemble_preds)
        
        marker = ""
        if mae < best_mae:
            best_mae = mae
            best_weights = {'xgboost': xgb_weight, 'random_forest': rf_weight}
            marker = " ★"
        
        log.info(f"  {xgb_weight:<12.1f} {rf_weight:<12.1f} {mae:<12.4f} {r2:<12.4f}{marker}")
    
    log.info(f"  ✓ Optimal Weights: XGBoost={best_weights['xgboost']:.0%}, "
          f"RF={best_weights['random_forest']:.0%}")
    log.info(f"  ✓ Best MAE: {best_mae:.4f}")
    
    return best_weights


def save_model(model, path):
    """Lưu mô hình đã huấn luyện vào file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    log.info(f"  → Mô hình được lưu tại {path}")


# ─── Pipeline huấn luyện chính ──────────────────────────────────────────────────

def main():
    log.info("=" * 55)
    log.info("  DỰ ĐOÁN GIÁ NHÀ — Pipeline Huấn Luyện (Enhanced)")
    log.info("=" * 55)

    # ── Bước 1: Tải dữ liệu ─────────────────────────────────────────────────
    log.info(f"[1/9] Đang tải dữ liệu đã xử lý từ {PROCESSED_DATA_PATH}...")
    df = load_processed_data(PROCESSED_DATA_PATH)
    log.info(f"  Kích thước dữ liệu: {df.shape}")
    log.info(f"  Các cột: {df.columns.tolist()}")

    # ── Bước 2: Loại bỏ outliers ───────────────────────────────────────────
    log.info(f"[2/9] Loại bỏ outliers (Giá > {PRICE_MAX} tỷ, Diện tích > {AREA_MAX} m²)...")
    df, n_removed = remove_outliers(df)
    log.info(f"  Đã loại bỏ {n_removed} hàng outliers")
    log.info(f"  Còn lại: {len(df)} mẫu")
    log.info(f"  Khoảng giá: {df['Price'].min():.2f} — {df['Price'].max():.2f} tỷ")
    log.info(f"  Khoảng diện tích:  {df['Area'].min():.1f} — {df['Area'].max():.1f} m²")

    # ── Bước 3: Chuẩn bị features & target ─────────────────────────────────
    log.info(f"[3/9] Đang chuẩn bị features và target...")
    X, y = prepare_data(df)
    log.info(f"  Features: {X.shape[1]} cột, {X.shape[0]} hàng")
    log.info(f"  Danh sách features: {X.columns.tolist()}")
    log.info(f"  Target (Giá) trung bình: {y.mean():.2f} tỷ, trung vị: {y.median():.2f} tỷ")

    # ── Bước 4: Train/Test split ──────────────────────────────────────────
    log.info(f"[4/9] Chia dữ liệu (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    log.info(f"  Tập train: {X_train.shape[0]} mẫu")
    log.info(f"  Tập test:  {X_test.shape[0]} mẫu")

    # ── Bước 5: Huấn luyện models ──────────────────────────────────────────────
    log.info(f"[5/9] Đang huấn luyện các mô hình...")

    # Linear Regression
    log.info("-" * 55 + " Linear Regression " + "-" * 10)
    lr_model = train_model(X_train, y_train, model_type='linear')
    lr_results = evaluate_model(lr_model, X_train, y_train, X_test, y_test, "Linear Regression")
    save_model(lr_model, os.path.join(MODELS_DIR, "linear_regression_model.pkl"))

    # Random Forest
    log.info("-" * 55 + " Random Forest " + "-" * 10)
    log.info(f"  Tham số: {RF_PARAMS}")
    rf_model = train_model(X_train, y_train, model_type='random_forest')
    rf_results = evaluate_model(rf_model, X_train, y_train, X_test, y_test, "Random Forest")
    save_model(rf_model, os.path.join(MODELS_DIR, "random_forest_model.pkl"))

    # XGBoost (nếu có)
    xgb_model = None
    xgb_results = None
    if HAS_XGBOOST:
        log.info("-" * 55 + " XGBoost " + "-" * 10)
        log.info(f"  Tham số: {XGB_PARAMS}")
        xgb_model = train_model(X_train, y_train, model_type='xgboost')
        xgb_results = evaluate_model(xgb_model, X_train, y_train, X_test, y_test, "XGBoost")
        save_model(xgb_model, os.path.join(MODELS_DIR, "xgboost_model.pkl"))

    # ── Bước 6: Cross-validation & tổng kết ────────────────────────────────
    log.info(f"[6/9] Cross-validation & tổng kết...")
    log.info("-" * 55 + " 5-FOLD CROSS-VALIDATION " + "-" * 10)

    # Random Forest CV
    log.info("Random Forest:")
    rf_cv_results = cross_validate_model(X, y, model_type='random_forest')

    # XGBoost CV (nếu có)
    xgb_cv_results = None
    if HAS_XGBOOST:
        log.info("XGBoost:")
        xgb_cv_results = cross_validate_model(X, y, model_type='xgboost')

    # So sánh mô hình cuối cùng
    log.info("=" * 65)
    log.info("  SO SÁNH MÔ HÌNH CUỐI CÙNG")
    log.info("=" * 65)
    header = f"{'Mô hình':<22} {'Test MAE':<12} {'Test RMSE':<12} {'Test R²':<10} {'CV R²':<10}"
    log.info(f"  {header}")
    log.info(f"  {'-'*66}")
    log.info(f"  {'Linear Regression':<22} {lr_results['test_mae']:<12.4f} {lr_results['test_rmse']:<12.4f} {lr_results['test_r2']:<10.4f} {'—':<10}")
    log.info(f"  {'Random Forest':<22} {rf_results['test_mae']:<12.4f} {rf_results['test_rmse']:<12.4f} {rf_results['test_r2']:<10.4f} {rf_cv_results['cv_r2_mean']:<10.4f}")
    if HAS_XGBOOST and xgb_cv_results:
        log.info(f"  {'XGBoost':<22} {xgb_results['test_mae']:<12.4f} {xgb_results['test_rmse']:<12.4f} {xgb_results['test_r2']:<10.4f} {xgb_cv_results['cv_r2_mean']:<10.4f}")

    # Xác định mô hình tốt nhất
    log.info("=" * 65)
    log.info("  KHUYẾN NGHỊ MÔ HÌNH TỐT NHẤT")
    log.info("=" * 65)
    models = [
        ('Random Forest', rf_results['test_r2'], rf_results['test_mae'], rf_cv_results['cv_r2_mean']),
    ]
    if HAS_XGBOOST and xgb_cv_results:
        models.append(('XGBoost', xgb_results['test_r2'], xgb_results['test_mae'], xgb_cv_results['cv_r2_mean']))

    best_model = max(models, key=lambda x: x[3])  # Theo CV R² (đáng tin cậy nhất)
    log.info(f"  Tốt nhất theo CV R²: {best_model[0]} (CV R² = {best_model[3]:.4f}, Test R² = {best_model[1]:.4f})")
    log.info(f"  File model: {best_model[0].lower().replace(' ', '_')}_model.pkl")

    # ── Bước 7: Weighted Ensemble (XGBoost + Random Forest) ────────────────
    if HAS_XGBOOST and xgb_model:
        log.info("-" * 55 + " Weighted Ensemble " + "-" * 10)

        # 7a. Tìm trọng số tối ưu trên test set
        optimal_weights = optimize_ensemble_weights(
            xgb_model, rf_model, X_test, y_test, n_steps=11
        )

        # 7b. Đánh giá ensemble với trọng số tối ưu
        ensemble_results = evaluate_ensemble(
            xgb_model, rf_model, X_train, y_train, X_test, y_test,
            weights=optimal_weights
        )

        # 7c. Lưu trọng số vào file JSON
        weights_path = os.path.join(MODELS_DIR, "ensemble_weights.json")
        weights_data = {
            "xgboost_weight": optimal_weights['xgboost'],
            "random_forest_weight": optimal_weights['random_forest'],
            "ensemble_test_r2": ensemble_results['ensemble']['test_r2'],
        }
        with open(weights_path, 'w', encoding='utf-8') as f:
            json.dump(weights_data, f, indent=2)
        log.info(f"  → Trọng số ensemble được lưu tại {weights_path}")

        log.info("=" * 65)
        log.info("  KẾT QUẢ FINAL ENSEMBLE (WEIGHTED)")
        log.info("=" * 65)
        log.info(f"  Optimal Weights: XGBoost={optimal_weights['xgboost']:.0%}, RF={optimal_weights['random_forest']:.0%}")
        log.info(f"  Ensemble Test R²: {ensemble_results['ensemble']['test_r2']:.4f}")
        log.info(f"  Ensemble Test MAE: {ensemble_results['ensemble']['test_mae']:.4f}")
        log.info(f"\n  💡 Weighted Ensemble đã được tối ưu và lưu!")
        log.info(f"     File weights: ensemble_weights.json")
    
    # Lưu feature importances (lấy từ base estimator của random forest hoặc rf meta)
    rf_base = rf_model.regressor_ if hasattr(rf_model, 'regressor_') else rf_model
    if hasattr(rf_base, 'feature_importances_'):
        fi_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_base.feature_importances_
        }).sort_values('importance', ascending=False)

        log.info("  Top 10 Features quan trọng nhất (Random Forest):")
        for _, row in fi_df.head(10).iterrows():
            bar = '█' * int(row['importance'] * 100)
            log.info(f"    {row['feature']:<25} {row['importance']:.4f}  {bar}")

        fi_path = os.path.join(MODELS_DIR, 'feature_importances.csv')
        fi_df.to_csv(fi_path, index=False)
        log.info(f"  → Feature importances được lưu tại {fi_path}")

    # ── Bước 8: SHAP Explainability ─────────────────────────────────────────
    if HAS_SHAP and HAS_XGBOOST and xgb_model:
        log.info(f"[8/9] SHAP Explainability Analysis...")
        try:
            # SHAP expects the base XGBoost model to explain, not the TransformedTargetRegressor wrapper
            xgb_base = xgb_model.regressor_
            shap_results = generate_shap_analysis(
                xgb_base, X_train, X_test,
                feature_names=X.columns.tolist()
            )
            if shap_results:
                log.info("  ✅ SHAP analysis hoàn tất!")
        except Exception as e:
            log.error(f"  SHAP analysis thất bại: {e}")
    else:
        log.info("[8/9] SHAP — Bỏ qua (chưa cài đặt hoặc thiếu XGBoost model)")

    # ── Bước 9: Optuna Hyperparameter Tuning ────────────────────────────────
    if HAS_OPTUNA:
        log.info(f"[9/9] Optuna Hyperparameter Tuning...")
        try:
            optuna_results = run_full_optimization(
                X, y,
                monotone_constraints=XGB_PARAMS.get('monotone_constraints', {}),
                xgb_trials=50,
                rf_trials=30,
            )
            if optuna_results:
                log.info("  ✅ Optuna tuning hoàn tất! Xem kết quả tại models/optuna_best_params.json")
        except Exception as e:
            log.error(f"  Optuna tuning thất bại: {e}")
    else:
        log.info("[9/9] Optuna — Bỏ qua (chưa cài đặt)")

    log.info("=" * 55)
    log.info("  ✅ TOÀN BỘ PIPELINE HOÀN TẤT!")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
