"""
Optuna Hyperparameter Optimization Module.
Sử dụng Bayesian Optimization để tìm tham số tối ưu cho XGBoost và Random Forest.
Kết quả tốt nhất sẽ được lưu thành JSON để train.py có thể sử dụng.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from src.logger import get_logger

logger = get_logger("optuna_tuning")

# Lazy import
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("Optuna chưa được cài đặt. Chạy 'pip install optuna' để bật.")

try:
    from optuna.integration.mlflow import MLflowCallback
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logger.warning("MLflow integration cho Optuna chưa khả dụng.")

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def _xgb_objective(trial, X, y, monotone_constraints, n_folds=5):
    """Optuna objective function cho XGBoost."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 30),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 2.0),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        'monotone_constraints': monotone_constraints,
    }

    model = XGBRegressor(**params)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    scores = cross_val_score(
        model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1
    )
    rmse = np.sqrt(-scores.mean())
    return rmse  # Optuna minimizes by default


def _rf_objective(trial, X, y, n_folds=5):
    """Optuna objective function cho Random Forest."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 10, 40),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
        'random_state': 42,
        'n_jobs': -1,
    }

    model = RandomForestRegressor(**params)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    scores = cross_val_score(
        model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1
    )
    rmse = np.sqrt(-scores.mean())
    return rmse


def optimize_xgboost(X, y, monotone_constraints, n_trials=50, n_folds=5):
    """
    Tối ưu hyperparameters cho XGBoost bằng Optuna.

    Args:
        X: DataFrame features.
        y: Series target.
        monotone_constraints: Dict ràng buộc đơn điệu.
        n_trials: Số lần thử (nhiều hơn = tốt hơn nhưng chậm hơn).
        n_folds: Số folds cho cross-validation.

    Returns:
        dict chứa best_params, best_rmse, study object.
    """
    if not HAS_OPTUNA or not HAS_XGBOOST:
        logger.error("Cần cài đặt optuna và xgboost để chạy tối ưu.")
        return None

    logger.info("=" * 65)
    logger.info("  OPTUNA — Tối Ưu Hyperparameters XGBoost")
    logger.info(f"  Trials: {n_trials} | Folds: {n_folds}")
    logger.info("=" * 65)

    study = optuna.create_study(
        direction='minimize',
        study_name='xgboost_optimization',
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    callbacks = []
    if HAS_MLFLOW:
        mlflow_cb = MLflowCallback(
            tracking_uri="file://" + os.path.join(BASE_DIR, "mlruns"),
            metric_name="rmse",
        )
        callbacks.append(mlflow_cb)

    study.optimize(
        lambda trial: _xgb_objective(trial, X, y, monotone_constraints, n_folds),
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=callbacks,
    )

    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['verbosity'] = 0
    best_params['monotone_constraints'] = monotone_constraints
    best_rmse = study.best_value

    logger.info(f"\n  ✓ Tối ưu XGBoost hoàn tất!")
    logger.info(f"  Best RMSE (CV): {best_rmse:.4f}")
    logger.info(f"  Best Params:")
    for k, v in study.best_params.items():
        logger.info(f"    {k}: {v}")

    return {
        'best_params': best_params,
        'best_rmse': best_rmse,
        'study': study,
    }


def optimize_random_forest(X, y, n_trials=30, n_folds=5):
    """
    Tối ưu hyperparameters cho Random Forest bằng Optuna.

    Args:
        X: DataFrame features.
        y: Series target.
        n_trials: Số lần thử.
        n_folds: Số folds cho cross-validation.

    Returns:
        dict chứa best_params, best_rmse, study object.
    """
    if not HAS_OPTUNA:
        logger.error("Cần cài đặt optuna để chạy tối ưu.")
        return None

    logger.info("\n" + "=" * 65)
    logger.info("  OPTUNA — Tối Ưu Hyperparameters Random Forest")
    logger.info(f"  Trials: {n_trials} | Folds: {n_folds}")
    logger.info("=" * 65)

    study = optuna.create_study(
        direction='minimize',
        study_name='rf_optimization',
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    callbacks = []
    if HAS_MLFLOW:
        mlflow_cb = MLflowCallback(
            tracking_uri="file://" + os.path.join(BASE_DIR, "mlruns"),
            metric_name="rmse",
        )
        callbacks.append(mlflow_cb)

    study.optimize(
        lambda trial: _rf_objective(trial, X, y, n_folds),
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=callbacks,
    )

    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_rmse = study.best_value

    logger.info(f"\n  ✓ Tối ưu Random Forest hoàn tất!")
    logger.info(f"  Best RMSE (CV): {best_rmse:.4f}")
    logger.info(f"  Best Params:")
    for k, v in study.best_params.items():
        logger.info(f"    {k}: {v}")

    return {
        'best_params': best_params,
        'best_rmse': best_rmse,
        'study': study,
    }


def run_full_optimization(X, y, monotone_constraints, xgb_trials=50, rf_trials=30):
    """
    Chạy tối ưu cho cả XGBoost và Random Forest, lưu kết quả.

    Args:
        X, y: Dữ liệu huấn luyện.
        monotone_constraints: Dict ràng buộc đơn điệu cho XGBoost.
        xgb_trials: Số trials cho XGBoost.
        rf_trials: Số trials cho Random Forest.

    Returns:
        dict chứa kết quả tối ưu cho cả hai model.
    """
    results = {}

    # ── XGBoost ──
    xgb_result = optimize_xgboost(X, y, monotone_constraints, n_trials=xgb_trials)
    if xgb_result:
        results['xgboost'] = {
            'best_params': {k: v for k, v in xgb_result['best_params'].items()
                           if k != 'monotone_constraints'},
            'best_rmse': xgb_result['best_rmse'],
        }

    # ── Random Forest ──
    rf_result = optimize_random_forest(X, y, n_trials=rf_trials)
    if rf_result:
        results['random_forest'] = {
            'best_params': rf_result['best_params'],
            'best_rmse': rf_result['best_rmse'],
        }

    # ── Lưu kết quả ──
    output_path = os.path.join(MODELS_DIR, "optuna_best_params.json")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Convert numpy types to Python native for JSON serialization
    serializable = {}
    for model_name, data in results.items():
        serializable[model_name] = {
            'best_params': {k: (float(v) if isinstance(v, (np.floating,)) else
                               int(v) if isinstance(v, (np.integer,)) else v)
                           for k, v in data['best_params'].items()},
            'best_rmse': float(data['best_rmse']),
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    logger.info(f"\n  → Kết quả Optuna được lưu tại: {output_path}")

    # ── So sánh ──
    logger.info("\n" + "=" * 65)
    logger.info("  OPTUNA — TỔNG KẾT TỐI ƯU")
    logger.info("=" * 65)
    for model_name, data in results.items():
        logger.info(f"  {model_name}: RMSE = {data['best_rmse']:.4f}")
    logger.info("=" * 65)

    return {
        'xgboost': xgb_result,
        'random_forest': rf_result,
        'output_path': output_path,
    }
