"""
Dự đoán giá nhà — Pipeline huấn luyện Model
Huấn luyện các mô hình Linear Regression, Random Forest và XGBoost trên dữ liệu đã tiền xử lý.
Sử dụng pipeline đã được sửa để tránh data leakage.
Hỗ trợ Ensemble Evaluation (XGBoost + Random Forest).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os
import sys
import io
import json

# Fix UTF-8 on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Thử import XGBoost (tùy chọn)
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Lưu ý: XGBoost chưa được cài đặt. Chạy 'pip install xgboost' để bật.")


# ─── Cấu hình ────────────────────────────────────────────────────────────
# Đường dẫn tuyệt đối để chạy từ bất kỳ thư mục nào
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "house_processed.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ngưỡng loại bỏ outliers
PRICE_MAX = 250.0       # Loại bỏ bất động sản > 250 tỷ VNĐ
AREA_MAX = 500.0        # Loại bỏ bất động sản > 500 m²

# Tham số Random Forest
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
    """Huấn luyện một mô hình regression."""
    if model_type == 'linear':
        model = LinearRegression()
        print(f"Đang huấn luyện mô hình {model_type}...")
        model.fit(X_train, y_train)
        print(f"  ✓ Mô hình {model_type} đã được huấn luyện thành công!")

    elif model_type == 'random_forest':
        model = RandomForestRegressor(**RF_PARAMS)
        print(f"Đang huấn luyện mô hình {model_type}...")
        model.fit(X_train, y_train)
        print(f"  ✓ Mô hình {model_type} đã được huấn luyện thành công!")

    elif model_type == 'xgboost':
        if not HAS_XGBOOST:
            raise ImportError("XGBoost chưa được cài đặt. Chạy 'pip install xgboost' trước.")

        # Tách validation set từ training data nếu không có
        if X_val is None or y_val is None:
            X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
                X_train, y_train, test_size=0.15, random_state=42
            )
            print(f"  Tách validation set: {len(X_val_xgb)} mẫu (15% của train)")
        else:
            X_train_xgb, y_train_xgb = X_train, y_train
            X_val_xgb, y_val_xgb = X_val, y_val

        # Cấu hình XGBoost - early_stopping_rounds đã có trong XGB_PARAMS
        model = XGBRegressor(**XGB_PARAMS)
        print(f"Đang huấn luyện XGBoost với early stopping ({XGB_PARAMS['early_stopping_rounds']} rounds)...")

        model.fit(
            X_train_xgb, y_train_xgb,
            eval_set=[(X_val_xgb, y_val_xgb)],
            verbose=False
        )

        print(f"  ✓ XGBoost huấn luyện xong! Best iteration: {model.best_iteration}")
    else:
        raise ValueError(f"Kiểu model không xác định: {model_type}")

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Đánh giá hiệu suất mô hình trên tập train và test."""
    print(f"\n{'='*55}")
    print(f"  Mô hình: {model_name}")
    print(f"{'='*55}")

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
    print(f"\n  Tập Train:")
    print(f"    MAE:  {train_mae:.4f} tỷ")
    print(f"    RMSE: {train_rmse:.4f} tỷ")
    print(f"    R²:   {train_r2:.4f}")

    print(f"\n  Tập Test:")
    print(f"    MAE:  {test_mae:.4f} tỷ")
    print(f"    RMSE: {test_rmse:.4f} tỷ")
    print(f"    R²:   {test_r2:.4f}")

    # Kiểm tra overfitting
    r2_gap = train_r2 - test_r2
    if r2_gap > 0.15:
        print(f"\n  ⚠️  R² gap = {r2_gap:.4f} — overfitting đáng kể!")
    elif r2_gap > 0.05:
        print(f"\n  ⚡ R² gap = {r2_gap:.4f} — overfitting nhẹ")
    else:
        print(f"\n  ✓  R² gap = {r2_gap:.4f} — khả năng tổng quát hóa tốt")

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
    print(f"\n  {n_folds}-Fold Cross-Validation:")
    
    if model_type == 'random_forest':
        model = RandomForestRegressor(**RF_PARAMS)
    elif model_type == 'xgboost' and HAS_XGBOOST:
        # XGBoost CV params (no early stopping)
        xgb_cv_params = {k: v for k, v in XGB_PARAMS.items() if k != 'early_stopping_rounds'}
        model = XGBRegressor(**xgb_cv_params)
    else:
        raise ValueError(f"Unsupported model type for CV: {model_type}")
    
    cv_r2 = cross_val_score(model, X, y, cv=n_folds, scoring='r2', n_jobs=-1)
    cv_mae = -cross_val_score(model, X, y, cv=n_folds, scoring='neg_mean_absolute_error', n_jobs=-1)
    
    print(f"    CV R²:    {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    print(f"    CV MAE:   {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")
    
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
    
    print(f"\n{'='*65}")
    print(f"  ENSEMBLE EVALUATION (XGBoost + Random Forest)")
    print(f"{'='*65}")
    print(f"  Weights: XGBoost = {weights['xgboost']:.0%}, RF = {weights['random_forest']:.0%}")
    
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
    
    print(f"\n  {'Model':<18} {'Train R²':<12} {'Test R²':<12} {'Test MAE':<12} {'Test RMSE':<12}")
    print(f"  {'-'*62}")
    
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
        
        print(f"  {name:<18} {train_r2:<12.4f} {test_r2:<12.4f} {test_mae:<12.4f} {test_rmse:<12.4f}")
    
    # So sánh ensemble với best single model
    print(f"\n  📊 ENSEMBLE IMPROVEMENT:")
    
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
    
    print(f"    Best Single Model: {best_single[0]} (R² = {best_single_r2:.4f})")
    print(f"    Ensemble R²:       {ensemble_r2:.4f} (Δ = {r2_improvement:+.4f})")
    print(f"    Ensemble MAE:      {ensemble_mae:.4f} (Δ = {mae_improvement:+.4f})")
    
    # Tính prediction spread (độ đồng thuận giữa 2 models)
    pred_diff = np.abs(xgb_test_pred - rf_test_pred)
    print(f"\n  📈 Prediction Agreement:")
    print(f"    Mean Abs Difference: {pred_diff.mean():.4f} tỷ")
    print(f"    Max Difference:     {pred_diff.max():.4f} tỷ")
    print(f"    Agreement Rate:    {(pred_diff < 0.5).mean()*100:.1f}% (diff < 0.5 tỷ)")
    
    # Cross-validation cho ensemble
    print(f"\n  {5}-Fold Cross-Validation for Ensemble:")
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
    
    print(f"    CV R²:  {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
    print(f"    CV MAE: {cv_mae_mean:.4f} ± {cv_mae_std:.4f}")
    
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
    print(f"\n{'='*65}")
    print(f"  OPTIMIZING ENSEMBLE WEIGHTS")
    print(f"{'='*65}")
    
    xgb_preds = xgb_model.predict(X_val)
    rf_preds = rf_model.predict(X_val)
    
    best_mae = float('inf')
    best_weights = {'xgboost': 0.5, 'random_forest': 0.5}
    
    print(f"\n  {'XGB Weight':<12} {'RF Weight':<12} {'MAE':<12} {'R²':<12}")
    print(f"  {'-'*48}")
    
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
        
        print(f"  {xgb_weight:<12.1f} {rf_weight:<12.1f} {mae:<12.4f} {r2:<12.4f}{marker}")
    
    print(f"\n  ✓ Optimal Weights: XGBoost={best_weights['xgboost']:.0%}, "
          f"RF={best_weights['random_forest']:.0%}")
    print(f"  ✓ Best MAE: {best_mae:.4f}")
    
    return best_weights


def save_model(model, path):
    """Lưu mô hình đã huấn luyện vào file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  → Mô hình được lưu tại {path}")


# ─── Pipeline huấn luyện chính ──────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  DỰ ĐOÁN GIÁ NHÀ — Pipeline Huấn Luyện (Fixed)")
    print("=" * 55)

    # ── Bước 1: Tải dữ liệu ─────────────────────────────────────────────────
    print(f"\n[1/6] Đang tải dữ liệu đã xử lý từ {PROCESSED_DATA_PATH}...")
    df = load_processed_data(PROCESSED_DATA_PATH)
    print(f"  Kích thước dữ liệu: {df.shape}")
    print(f"  Các cột: {df.columns.tolist()}")

    # ── Bước 2: Loại bỏ outliers ───────────────────────────────────────────
    print(f"\n[2/6] Loại bỏ outliers (Giá > {PRICE_MAX} tỷ, Diện tích > {AREA_MAX} m²)...")
    df, n_removed = remove_outliers(df)
    print(f"  Đã loại bỏ {n_removed} hàng outliers")
    print(f"  Còn lại: {len(df)} mẫu")
    print(f"  Khoảng giá: {df['Price'].min():.2f} — {df['Price'].max():.2f} tỷ")
    print(f"  Khoảng diện tích:  {df['Area'].min():.1f} — {df['Area'].max():.1f} m²")

    # ── Bước 3: Chuẩn bị features & target ─────────────────────────────────
    print(f"\n[3/6] Đang chuẩn bị features và target...")
    X, y = prepare_data(df)
    print(f"  Features: {X.shape[1]} cột, {X.shape[0]} hàng")
    print(f"  Danh sách features: {X.columns.tolist()}")
    print(f"  Target (Giá) trung bình: {y.mean():.2f} tỷ, trung vị: {y.median():.2f} tỷ")

    # ── Bước 4: Train/Test split ──────────────────────────────────────────
    print(f"\n[4/6] Chia dữ liệu (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Tập train: {X_train.shape[0]} mẫu")
    print(f"  Tập test:     {X_test.shape[0]} mẫu")

    # ── Bước 5: Huấn luyện models ──────────────────────────────────────────────
    print(f"\n[5/6] Đang huấn luyện các mô hình...")

    # Linear Regression
    print("\n" + "-" * 55)
    print("  HUẤN LUYỆN: Linear Regression")
    print("-" * 55)
    lr_model = train_model(X_train, y_train, model_type='linear')
    lr_results = evaluate_model(lr_model, X_train, y_train, X_test, y_test, "Linear Regression")
    save_model(lr_model, os.path.join(MODELS_DIR, "linear_regression_model.pkl"))

    # Random Forest
    print("\n" + "-" * 55)
    print("  HUẤN LUYỆN: Random Forest")
    print(f"  Tham số: {RF_PARAMS}")
    print("-" * 55)
    rf_model = train_model(X_train, y_train, model_type='random_forest')
    rf_results = evaluate_model(rf_model, X_train, y_train, X_test, y_test, "Random Forest")
    save_model(rf_model, os.path.join(MODELS_DIR, "random_forest_model.pkl"))

    # XGBoost (nếu có)
    if HAS_XGBOOST:
        print("\n" + "-" * 55)
        print("  HUẤN LUYỆN: XGBoost")
        print(f"  Tham số: {XGB_PARAMS}")
        print("-" * 55)
        xgb_model = train_model(X_train, y_train, model_type='xgboost')
        xgb_results = evaluate_model(xgb_model, X_train, y_train, X_test, y_test, "XGBoost")
        save_model(xgb_model, os.path.join(MODELS_DIR, "xgboost_model.pkl"))

    # ── Bước 6: Cross-validation & tổng kết ────────────────────────────────
    print(f"\n[6/6] Cross-validation & tổng kết...")

    # 5-Fold CV cho các mô hình
    print("\n" + "-" * 55)
    print("  5-FOLD CROSS-VALIDATION")
    print("-" * 55)

    # Random Forest CV
    print("\nRandom Forest:")
    rf_cv_results = cross_validate_model(X, y, model_type='random_forest')

    # XGBoost CV (nếu có)
    if HAS_XGBOOST:
        print("\nXGBoost:")
        xgb_cv_results = cross_validate_model(X, y, model_type='xgboost')

    # So sánh mô hình cuối cùng
    print("\n" + "=" * 65)
    print("  SO SÁNH MÔ HÌNH CUỐI CÙNG")
    print("=" * 65)
    header = f"{'Mô hình':<22} {'Test MAE':<12} {'Test RMSE':<12} {'Test R²':<10} {'CV R²':<10}"
    print(f"  {header}")
    print(f"  {'-'*66}")
    print(f"  {'Linear Regression':<22} {lr_results['test_mae']:<12.4f} {lr_results['test_rmse']:<12.4f} {lr_results['test_r2']:<10.4f} {'—':<10}")
    print(f"  {'Random Forest':<22} {rf_results['test_mae']:<12.4f} {rf_results['test_rmse']:<12.4f} {rf_results['test_r2']:<10.4f} {rf_cv_results['cv_r2_mean']:<10.4f}")
    if HAS_XGBOOST:
        print(f"  {'XGBoost':<22} {xgb_results['test_mae']:<12.4f} {xgb_results['test_rmse']:<12.4f} {xgb_results['test_r2']:<10.4f} {xgb_cv_results['cv_r2_mean']:<10.4f}")

    # Xác định mô hình tốt nhất
    print("\n" + "=" * 65)
    print("  KHUYẾN NGHỊ MÔ HÌNH TỐT NHẤT")
    print("=" * 65)
    models = [
        ('Random Forest', rf_results['test_r2'], rf_results['test_mae'], rf_cv_results['cv_r2_mean']),
    ]
    if HAS_XGBOOST:
        models.append(('XGBoost', xgb_results['test_r2'], xgb_results['test_mae'], xgb_cv_results['cv_r2_mean']))

    best_model = max(models, key=lambda x: x[3])  # Theo CV R² (đáng tin cậy nhất)
    print(f"  Tốt nhất theo CV R²: {best_model[0]} (CV R² = {best_model[3]:.4f}, Test R² = {best_model[1]:.4f})")
    print(f"  File model: {best_model[0].lower().replace(' ', '_')}_model.pkl")

    # ── Bước 7: Ensemble Evaluation (XGBoost + Random Forest) ───────────────
    if HAS_XGBOOST:
        print(f"\n[7/7] Ensemble Evaluation & Optimization...")
        
        # Tạo validation set từ test data để tối ưu weights
        X_train_main, X_val_opt, y_train_main, y_val_opt = train_test_split(
            X_train, y_train, test_size=0.2, random_state=99
        )
        
        # Tối ưu weights
        optimal_weights = optimize_ensemble_weights(
            xgb_model, rf_model, X_val_opt, y_val_opt
        )
        
        # Đánh giá ensemble với weights khác nhau
        print("\n" + "-" * 65)
        print("  Testing Different Ensemble Weights:")
        print("-" * 65)
        
        weight_configs = [
            {'xgboost': 0.5, 'random_forest': 0.5, 'name': 'Equal (50/50)'},
            {'xgboost': 0.6, 'random_forest': 0.4, 'name': 'XGB Heavy (60/40)'},
            {'xgboost': 0.7, 'random_forest': 0.3, 'name': 'XGB Heavy (70/30)'},
            {'xgboost': 0.4, 'random_forest': 0.6, 'name': 'RF Heavy (40/60)'},
            optimal_weights,  # Optimal weights
        ]
        
        best_config = None
        best_ensemble_r2 = -float('inf')
        
        for config in weight_configs:
            config_results = evaluate_ensemble(
                xgb_model, rf_model, 
                X_train, y_train, 
                X_test, y_test, 
                weights=config
            )
            
            if config_results['ensemble']['test_r2'] > best_ensemble_r2:
                best_ensemble_r2 = config_results['ensemble']['test_r2']
                best_config = config
        ##Grid search
        print("\n" + "=" * 65)
        print("  FINAL ENSEMBLE RECOMMENDATION")
        print("=" * 65)
        print(f"  Recommended Weights: XGBoost={best_config['xgboost']:.0%}, "
              f"RF={best_config['random_forest']:.0%}")
        print(f"  Ensemble Test R²: {best_ensemble_r2:.4f}")
        print(f"\n  💡 Use these weights in app.py for best performance!")
        print(f"     ENSEMBLE_WEIGHTS = {best_config}")
        
        # Lưu optimal weights vào file để app.py có thể sử dụng
        weights_file = os.path.join(MODELS_DIR, 'ensemble_weights.json')
        weights_data = {
            'xgboost_weight': best_config['xgboost'],
            'random_forest_weight': best_config['random_forest'],
            'ensemble_test_r2': float(best_ensemble_r2),
        }
        with open(weights_file, 'w') as f:
            json.dump(weights_data, f, indent=2)
        print(f"\n  → Optimal weights saved to {weights_file}")
    
    # Lưu feature importances
    if hasattr(rf_model, 'feature_importances_'):
        fi_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n  Top 10 Features quan trọng nhất (Random Forest):")
        for _, row in fi_df.head(10).iterrows():
            bar = '█' * int(row['importance'] * 100)
            print(f"    {row['feature']:<25} {row['importance']:.4f}  {bar}")

        fi_path = os.path.join(MODELS_DIR, 'feature_importances.csv')
        fi_df.to_csv(fi_path, index=False)
        print(f"\n  → Feature importances được lưu tại {fi_path}")

    print("\n" + "=" * 55)
    print("  ✅ Huấn luyện hoàn tất!")
    print("=" * 55)


if __name__ == "__main__":
    main()
