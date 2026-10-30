"""
Train the clinical XGBoost HCC risk model.

Pipeline
    1. Load data/hcc_xgboost/hcc_synthetic_dataset.xlsx (5,000 patients).
    2. Stratified 70/15/15 split, saved to outputs/splits/clinical_split.csv.
    3. Fit the preprocessor on the training split only.
    4. 5-fold grid search on the training split.
    5. Final fit with early stopping on the validation split.
    6. Evaluate once on the test split and save the model, preprocessor,
       plots and outputs/metrics/clinical_metrics.txt.

Usage
    python train_clinical_xgb.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)

from src.clinical_xgb_data import (
    load_dataset, split_dataset, build_preprocessor, fit_preprocessor,
    transform_to_frame, save_preprocessor, TARGET_COLUMN,
)
from src.clinical_xgb_model import create_model, cross_validated_search

CONFIG = {
    "data_path": os.path.join("data", "hcc_xgboost", "hcc_synthetic_dataset.xlsx"),
    "seed": 42,
    "run_hyperparam_search": True,
    "cv_folds": 5,
    "early_stopping_rounds": 20,
    "save_dir": "outputs",
}


def _plot_roc_curve(y_true, y_proba, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#3498db")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("Clinical XGBoost Model — ROC Curve (Test Split)", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] ROC curve saved to: {save_path} (AUC={auc_score:.4f})")
    return auc_score


def _plot_loss_curve(evals_result, save_path):
    train_loss = evals_result["validation_0"]["logloss"]
    val_loss = evals_result["validation_1"]["logloss"]
    rounds = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rounds, train_loss, color="#2ecc71", lw=1.8, label="Train logloss")
    ax.plot(rounds, val_loss, color="#e74c3c", lw=1.8, label="Validation logloss")
    ax.set_xlabel("Boosting Round", fontsize=13)
    ax.set_ylabel("Log Loss", fontsize=13)
    ax.set_title("Clinical XGBoost Model — Training / Eval Loss", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Loss curve saved to: {save_path}")


def _plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No HCC", "HCC"])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Clinical XGBoost Model — Confusion Matrix (Test Split)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved to: {save_path}")
    return cm


def _write_metrics_file(metrics, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("CLINICAL XGBOOST MODEL — TEST SET RESULTS\n")
        f.write("=" * 50 + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key:14s}: {value:.4f}\n")
    print(f"[INFO] Metrics saved to: {save_path}")


def main():
    cfg = CONFIG
    plots_dir = os.path.join(cfg["save_dir"], "plots")
    models_dir = os.path.join(cfg["save_dir"], "models")
    metrics_dir = os.path.join(cfg["save_dir"], "metrics")
    for d in (plots_dir, models_dir, metrics_dir):
        os.makedirs(d, exist_ok=True)

    # Split before any preprocessing
    df = load_dataset(cfg["data_path"])
    train_df, val_df, test_df = split_dataset(df, seed=cfg["seed"])

    preprocessor = build_preprocessor()
    fit_preprocessor(preprocessor, train_df)
    save_preprocessor(preprocessor, os.path.join(models_dir, "clinical_xgb_preprocessor.joblib"))

    X_train = transform_to_frame(preprocessor, train_df)
    X_val = transform_to_frame(preprocessor, val_df)
    X_test = transform_to_frame(preprocessor, test_df)
    y_train = train_df[TARGET_COLUMN].to_numpy()
    y_val = val_df[TARGET_COLUMN].to_numpy()
    y_test = test_df[TARGET_COLUMN].to_numpy()

    best_params = {}
    if cfg["run_hyperparam_search"]:
        search = cross_validated_search(X_train, y_train, seed=cfg["seed"], n_splits=cfg["cv_folds"])
        best_params = search.best_params_

    model = create_model(
        **best_params,
        early_stopping_rounds=cfg["early_stopping_rounds"],
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True,
    )

    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    cm = _plot_confusion_matrix(y_test, y_test_pred, os.path.join(plots_dir, "clinical_xgb_confusion_matrix.png"))
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_test_proba),
    }
    _write_metrics_file(metrics, os.path.join(metrics_dir, "clinical_metrics.txt"))

    _plot_roc_curve(y_test, y_test_proba, os.path.join(plots_dir, "clinical_xgb_roc_curve.png"))
    _plot_loss_curve(model.evals_result(), os.path.join(plots_dir, "clinical_xgb_loss_curve.png"))

    model_path = os.path.join(models_dir, "clinical_xgb_model.json")
    model.save_model(model_path)
    print(f"[INFO] Model saved to: {model_path}")

    with open(os.path.join(models_dir, "clinical_xgb_meta.json"), "w") as f:
        json.dump({
            "feature_columns_transformed": list(X_train.columns),
            "best_params": best_params,
            "n_estimators": int(model.n_estimators),
            "seed": cfg["seed"],
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("  CLINICAL XGBOOST TRAINING COMPLETE")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"  {key:14s}: {value:.4f}")


if __name__ == "__main__":
    main()
