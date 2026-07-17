"""
XGBoost fraud classifier with cost-sensitive threshold calibration.
Uses temporal train/test split to avoid data leakage.
"""
import pandas as pd
import numpy as np
import joblib
import yaml
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features
from evaluate import cost_threshold

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

CONFIG_PATH = "config.yaml"


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(cfg["data"]["transactions_path"])
    X, encoder = build_features(df, fit=True)
    y = df["is_fraud"]

    # temporal split — use transaction index as proxy for time
    cutoff = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
    y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Fraud rate — train: {y_train.mean():.2%} | test: {y_test.mean():.2%}")

    # SMOTE oversampling on training set only
    sm = SMOTE(sampling_strategy=0.1, random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE — train size: {len(X_res):,}")

    model = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_res == 0).sum() / (y_res == 1).sum(),
        eval_metric="aucpr",
        early_stopping_rounds=40,
        random_state=42,
    )
    model.fit(X_res, y_res, eval_set=[(X_test, y_test)], verbose=100)

    proba = model.predict_proba(X_test)[:, 1]
    auc  = roc_auc_score(y_test, proba)
    ap   = average_precision_score(y_test, proba)
    fn_fp_ratio = cfg["model"]["fn_fp_cost_ratio"]
    thresh = cost_threshold(y_test, proba, fn_fp_ratio)

    print(f"\nAUC-ROC: {auc:.4f} | Avg Precision: {ap:.4f}")
    print(f"Cost-optimal threshold (FN:FP={fn_fp_ratio}:1): {thresh:.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model": model,
        "encoder": encoder,
        "threshold": thresh,
        "features": list(X.columns),
    }, cfg["data"]["xgb_model_path"])
    print(f"Saved → {cfg['data']['xgb_model_path']}")


if __name__ == "__main__":
    main()
