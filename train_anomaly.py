"""
Ensemble scorer: blends XGBoost + Isolation Forest outputs.
Writes fraud alerts to CSV (Snowflake / monitoring table in production).
"""
import pandas as pd
import numpy as np
import joblib
import yaml
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features


def normalize_if_scores(scores):
    """Map IF decision function (lower=anomalous) to [0,1] probability-like score."""
    mn, mx = scores.min(), scores.max()
    normalized = (scores - mn) / (mx - mn + 1e-9)
    return 1 - normalized   # invert: high score = high anomaly


def score_batch(input_path, output_path, config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    xgb_bundle = joblib.load(cfg["data"]["xgb_model_path"])
    iso_bundle  = joblib.load(cfg["data"]["iso_model_path"])

    df = pd.read_csv(input_path)
    print(f"[{datetime.now():%H:%M:%S}] Scoring {len(df):,} transactions...")

    # XGBoost scores
    X_xgb, _ = build_features(df, encoder=xgb_bundle["encoder"], fit=False)
    xgb_proba = xgb_bundle["model"].predict_proba(X_xgb)[:, 1]

    # Isolation Forest scores
    X_iso, _ = build_features(df, encoder=iso_bundle["encoder"], fit=False)
    iso_scores = iso_bundle["iso"].decision_function(X_iso)
    iso_proba  = normalize_if_scores(iso_scores)

    # ensemble blend
    w = cfg["model"]["ensemble_xgb_weight"]
    df["xgb_score"]     = xgb_proba
    df["anomaly_score"] = iso_proba
    df["fraud_score"]   = w * xgb_proba + (1 - w) * iso_proba
    df["alert_flag"]    = (df["fraud_score"] >= xgb_bundle["threshold"]).astype(int)

    alerts = df[df["alert_flag"] == 1].copy()
    alerts = alerts.sort_values("fraud_score", ascending=False)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    alerts.to_csv(output_path, index=False)

    print(f"Total transactions: {len(df):,}")
    print(f"Alerts generated:   {len(alerts):,} ({len(alerts)/len(df):.2%})")
    if "is_fraud" in df.columns:
        prec = alerts["is_fraud"].mean()
        rec  = df[df["is_fraud"]==1]["alert_flag"].mean()
        print(f"Precision: {prec:.2%} | Recall: {rec:.2%}")
    print(f"Output → {output_path}")


if __name__ == "__main__":
    score_batch("data/transactions.csv", "data/fraud_alerts.csv")
