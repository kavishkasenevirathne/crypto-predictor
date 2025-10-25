#!/usr/bin/env python3
"""
train_lgb.py

Train a LightGBM classifier on preprocessed crypto features (time-series).

Usage:
    python src/models/train_lgb.py
    python src/models/train_lgb.py --train data/processed/train.parquet --val data/processed/val.parquet
"""
import argparse
from pathlib import Path
import pandas as pd
import joblib
import json
from datetime import datetime, timezone
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix
import warnings
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# ------------------------
# Utility functions
# ------------------------
def load_data(path: Path):
    """Load CSV or Parquet into a DataFrame."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    return df

def prepare_features(df: pd.DataFrame, target_col="target_dir"):
    """Return X (numeric features) and y (target). Drop timestamp and lookahead columns."""
    df = df.dropna(subset=[target_col])
    y = df[target_col].astype(int).reset_index(drop=True)
    X = df.drop(columns=[target_col, "ret_next"], errors="ignore")
    X = X.drop(columns=["timestamp"], errors="ignore")
    X = X.select_dtypes(include=[np.number]).reset_index(drop=True)
    mask = ~X.isna().any(axis=1)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    if len(X) == 0:
        raise ValueError("No valid numeric rows left after cleaning.")
    return X, y

# ------------------------
# Main training function
# ------------------------
def train_lgb(train_path, val_path, params_path=None, output_dir="models", quiet=False):
    train_df = load_data(Path(train_path))
    val_df = load_data(Path(val_path))

    if not quiet:
        print(f"Training set: {train_df.shape}, Validation set: {val_df.shape}")

    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Train/val sets must contain data after cleaning.")
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        print("Warning: train or val has a single class — some metrics (AUC) will be skipped.")

    # Default LGBMClassifier parameters
    clf_params = {
        "objective": "binary",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "random_state": 42,
        "verbose": -1
    }

    # Override with user params if provided
    if params_path:
        with open(params_path, "r", encoding="utf-8") as f:
            clf_params.update(json.load(f))

    # ------------------------
    # Train the model
    # ------------------------
    model = LGBMClassifier(**clf_params)
    callbacks = [early_stopping(50)]
    if not quiet:
        callbacks.append(log_evaluation(50))

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=callbacks
    )

    # ------------------------
    # Save model and parameters
    # ------------------------
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = Path(output_dir) / f"lgb_model_{timestamp}.pkl"
    joblib.dump(model, model_path)
    if not quiet:
        print(f"✅ Model saved to: {model_path}")

    params_path_out = Path(output_dir) / f"lgb_params_{timestamp}.json"
    with open(params_path_out, "w", encoding="utf-8") as pf:
        json.dump(clf_params, pf, indent=2)
    if not quiet:
        print(f"✅ Params saved to: {params_path_out}")

    # ------------------------
    # Validation metrics
    # ------------------------
    preds_prob = model.predict_proba(X_val)[:, 1]
    preds_label = (preds_prob > 0.5).astype(int)

    metrics = {}
    try:
        metrics["val_accuracy"] = float(accuracy_score(y_val, preds_label))
        metrics["val_auc"] = float(roc_auc_score(y_val, preds_prob)) if len(np.unique(y_val)) > 1 else None
        metrics["val_logloss"] = float(log_loss(y_val, preds_prob))
        metrics["confusion_matrix"] = confusion_matrix(y_val, preds_label).tolist()
    except Exception as e:
        metrics["eval_error"] = str(e)

    # Baselines
    baseline = {}
    if "ret1" in X_val.columns:
        base_pred = (X_val["ret1"] > 0).astype(int).values
        baseline["naive_last_dir_acc"] = float(accuracy_score(y_val, base_pred))
    if {"sma_12", "sma_30"}.issubset(set(X_val.columns)):
        mac_pred = (X_val["sma_12"] > X_val["sma_30"]).astype(int).values
        baseline["ma_crossover_acc"] = float(accuracy_score(y_val, mac_pred))
    metrics["baselines"] = baseline

    # Save metrics
    metrics_path = Path(output_dir) / f"lgb_metrics_{timestamp}.json"
    out_metrics = {"validation": metrics}
    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(out_metrics, mf, indent=2)
    if not quiet:
        print(f"✅ Metrics saved to: {metrics_path}")

    # Feature importance
    fi_path = Path(output_dir) / f"lgb_feature_importance_{timestamp}.csv"
    fi_df = pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_})
    fi_df.sort_values("importance", ascending=False).to_csv(fi_path, index=False)
    if not quiet:
        print(f"✅ Feature importance saved to: {fi_path}")
        print("Validation metrics summary:")
        print(json.dumps(metrics, indent=2))

    return model, out_metrics

# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM on crypto features")
    parser.add_argument("--train", "-tr", default="data/processed/train.parquet", help="Train split path")
    parser.add_argument("--val", "-v", default="data/processed/val.parquet", help="Validation split path")
    parser.add_argument("--params", "-p", help="Optional JSON file with LightGBM hyperparameters")
    parser.add_argument("--output-dir", "-o", default="models", help="Directory to save model and metrics")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")
    args = parser.parse_args()

    train_lgb(args.train, args.val, args.params, args.output_dir, quiet=args.quiet)
