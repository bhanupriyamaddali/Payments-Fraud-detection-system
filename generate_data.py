"""
Isolation Forest anomaly detector trained on legitimate transactions only.
Catches novel fraud patterns not present in the supervised training data.
"""
import pandas as pd
import numpy as np
import joblib
import yaml
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features
from sklearn.ensemble import IsolationForest


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(cfg["data"]["transactions_path"])
    # train ONLY on legitimate transactions — model learns "normal" behavior
    legit = df[df["is_fraud"] == 0].copy()
    X_legit, encoder = build_features(legit, fit=True)

    print(f"Training Isolation Forest on {len(X_legit):,} legitimate transactions...")

    iso = IsolationForest(
        n_estimators=200,
        contamination=cfg["model"]["if_contamination"],
        max_samples=0.8,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_legit)

    # sanity check: how well does IF separate fraud from legit on full dataset?
    X_all, _ = build_features(df, encoder=encoder, fit=False)
    scores = iso.decision_function(X_all)   # lower = more anomalous
    df["if_score"] = scores

    # at optimal threshold, what precision do we get on fraud?
    thresh_q = np.percentile(scores[df["is_fraud"] == 0], cfg["model"]["if_contamination"] * 100)
    flagged = df[scores <= thresh_q]
    precision = flagged["is_fraud"].mean()
    recall = flagged["is_fraud"].sum() / df["is_fraud"].sum()

    print(f"Isolation Forest sanity check:")
    print(f"  Flags: {len(flagged):,} transactions ({len(flagged)/len(df):.1%})")
    print(f"  Fraud precision: {precision:.2%}")
    print(f"  Fraud recall:    {recall:.2%}")

    os.makedirs("models", exist_ok=True)
    joblib.dump({"iso": iso, "encoder": encoder, "threshold": thresh_q},
                cfg["data"]["iso_model_path"])
    print(f"Saved → {cfg['data']['iso_model_path']}")


if __name__ == "__main__":
    main()
