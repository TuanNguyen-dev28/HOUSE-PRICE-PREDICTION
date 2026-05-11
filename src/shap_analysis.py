"""
SHAP (SHapley Additive exPlanations) Analysis Module.
Generates feature importance explanations for XGBoost model predictions.
Produces global summary plots and per-sample waterfall explanations.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from src.logger import get_logger

logger = get_logger("shap_analysis")

# Lazy import SHAP (heavy dependency)
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP chưa được cài đặt. Chạy 'pip install shap' để bật.")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "models", "shap_plots")


def generate_shap_analysis(xgb_model, X_train, X_test, feature_names=None, max_display=15):
    """
    Phân tích SHAP toàn diện cho mô hình XGBoost.

    Args:
        xgb_model: Mô hình XGBoost đã huấn luyện.
        X_train: DataFrame features tập train (dùng cho background).
        X_test: DataFrame features tập test (dùng để giải thích).
        feature_names: Danh sách tên features (nếu None, lấy từ X_train).
        max_display: Số features tối đa hiển thị trên biểu đồ.

    Returns:
        dict chứa shap_values, expected_value, và đường dẫn tới các plot.
    """
    if not HAS_SHAP:
        logger.error("SHAP không khả dụng. Bỏ qua phân tích.")
        return None

    os.makedirs(PLOTS_DIR, exist_ok=True)
    logger.info("Đang tính SHAP values cho XGBoost...")

    # ── Tạo explainer ──
    explainer = shap.TreeExplainer(xgb_model)

    # Lấy mẫu con nếu tập test quá lớn (SHAP nặng tính toán)
    if len(X_test) > 2000:
        logger.info(f"  Lấy mẫu 2000/{len(X_test)} dòng từ tập test để tăng tốc.")
        X_explain = X_test.sample(n=2000, random_state=42)
    else:
        X_explain = X_test

    shap_values = explainer(X_explain)
    logger.info(f"  ✓ SHAP values đã tính xong cho {len(X_explain)} mẫu.")

    results = {
        'shap_values': shap_values,
        'expected_value': explainer.expected_value,
        'plots': {},
    }

    # ── 1. Summary Bar Plot (Global Feature Importance) ──
    logger.info("  Đang tạo SHAP Summary Bar Plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=max_display, show=False, ax=ax)
    ax.set_title("SHAP — Mức Độ Ảnh Hưởng Trung Bình Của Từng Feature", fontsize=13)
    bar_path = os.path.join(PLOTS_DIR, "shap_feature_importance.png")
    fig.tight_layout()
    fig.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    results['plots']['bar'] = bar_path
    logger.info(f"    → Lưu tại {bar_path}")

    # ── 2. Beeswarm Plot (Impact direction per feature) ──
    logger.info("  Đang tạo SHAP Beeswarm Plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title("SHAP — Phân Bố Tác Động Của Từng Feature Lên Giá", fontsize=13)
    beeswarm_path = os.path.join(PLOTS_DIR, "shap_beeswarm.png")
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=150, bbox_inches='tight')
    plt.close()
    results['plots']['beeswarm'] = beeswarm_path
    logger.info(f"    → Lưu tại {beeswarm_path}")

    # ── 3. Waterfall Plots (Top 3 samples) ──
    logger.info("  Đang tạo Waterfall Plots cho 3 mẫu đại diện...")
    # Chọn 3 mẫu: giá thấp, giá trung bình, giá cao (theo predicted SHAP value)
    pred_sums = shap_values.values.sum(axis=1)
    sample_indices = [
        int(np.argmin(pred_sums)),   # Giá thấp nhất
        int(np.argsort(pred_sums)[len(pred_sums) // 2]),  # Trung vị
        int(np.argmax(pred_sums)),   # Giá cao nhất
    ]
    labels = ["low_price", "median_price", "high_price"]
    label_vi = ["Giá Thấp", "Giá Trung Bình", "Giá Cao"]

    for idx, label, label_text in zip(sample_indices, labels, label_vi):
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[idx], max_display=12, show=False)
        plt.title(f"SHAP Waterfall — Giải Thích Dự Đoán ({label_text})", fontsize=13)
        wf_path = os.path.join(PLOTS_DIR, f"shap_waterfall_{label}.png")
        plt.tight_layout()
        plt.savefig(wf_path, dpi=150, bbox_inches='tight')
        plt.close()
        results['plots'][f'waterfall_{label}'] = wf_path
        logger.info(f"    → {label_text}: {wf_path}")

    # ── 4. Lưu SHAP feature importance dạng CSV ──
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    if feature_names is None:
        feature_names = X_explain.columns.tolist()
    shap_importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
    }).sort_values('mean_abs_shap', ascending=False)

    csv_path = os.path.join(PLOTS_DIR, "shap_feature_importance.csv")
    shap_importance_df.to_csv(csv_path, index=False)
    results['importance_csv'] = csv_path
    logger.info(f"  → SHAP importance CSV: {csv_path}")

    # ── In top features ──
    logger.info("\n  📊 SHAP Feature Importance (Top 10):")
    for _, row in shap_importance_df.head(10).iterrows():
        bar = '█' * int(row['mean_abs_shap'] / shap_importance_df['mean_abs_shap'].max() * 30)
        logger.info(f"    {row['feature']:<25} {row['mean_abs_shap']:.4f}  {bar}")

    logger.info(f"\n  ✅ SHAP analysis hoàn tất! Plots lưu tại: {PLOTS_DIR}")
    return results
