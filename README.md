# Fraud Detection System with Real-Time Alert Scoring

Automated fraud detection pipeline combining supervised gradient boosting with unsupervised anomaly detection. Designed for financial transaction data. Achieves 92% recall on fraudulent transactions while maintaining an 85% precision rate — tuned around the cost reality that missing fraud is 15x more expensive than investigating a false alarm.

---

## Problem

Fraud detection is not a standard classification problem. Three things make it hard:

1. **Extreme class imbalance** — fraud is typically 0.5–2% of transactions. Naive models predict "not fraud" for everything and achieve 99% accuracy while catching zero fraud.
2. **Adversarial drift** — fraudsters adapt. A model trained only on historical labeled patterns will miss novel schemes. You need an anomaly detection layer that flags anything statistically unusual, regardless of whether it matches prior fraud.
3. **Cost asymmetry** — a missed fraud (FN) is ~15x more expensive than a false alert (FP). The decision threshold must reflect this, not default to 0.5.

---

## Approach

**Two-model ensemble:**

*Model 1 — XGBoost Classifier (supervised)*
Trained on labeled fraud/non-fraud transactions. Handles known fraud patterns well. Uses SMOTE + scale_pos_weight to address imbalance. Features are engineered to capture fraud behavioral signatures: late-night transactions, velocity bursts, high-amount foreign transactions with new accounts.

*Model 2 — Isolation Forest (unsupervised)*
Trained exclusively on legitimate transactions — it learns what "normal" looks like. Anything that deviates significantly from normal behavior triggers a high anomaly score, regardless of whether it resembles known fraud. This catches novel fraud patterns that the supervised model hasn't seen.

*Ensemble:*
`final_score = 0.7 × xgb_score + 0.3 × anomaly_score`

Blend weight tuned on a labeled validation set. XGBoost dominates for known patterns; anomaly detection provides the safety net.

**Threshold selection:**
Cost-minimizing threshold search using estimated FN/FP cost ratio. At FN:FP = 15:1, optimal threshold is ~0.28 (much lower than the typical 0.5 default).

**Temporal validation:**
Models trained on months 1-18, validated on 19-21, tested on 22-24. Prevents data leakage and produces realistic performance estimates.

---

## Results

Evaluated on 500K synthetic financial transactions (1.5% fraud rate):

| Metric | XGBoost only | Isolation Forest only | Ensemble |
|--------|--------------|----------------------|---------|
| Recall | 88% | 71% | **92%** |
| Precision | 89% | 43% | **85%** |
| AUC-ROC | 0.97 | 0.81 | **0.97** |
| Novel fraud recall* | 61% | 79% | **84%** |

*Novel fraud recall: performance on held-out fraud patterns not seen in training.

Ensemble improves especially on novel fraud — the anomaly model catches what the supervised model misses.

---

## Technical Highlights

- **Two-model ensemble** — supervised + unsupervised addresses both known and novel fraud
- **Cost-sensitive threshold** — threshold calibrated to FN:FP cost ratio, not arbitrary 0.5
- **Temporal validation** — no data leakage; realistic production performance estimates
- **SHAP explanations** — each alert includes top contributing features for investigator context
- **Batch pipeline** — scores daily transaction batches, writes alerts to monitoring table

---

## Stack

Python, XGBoost, scikit-learn (IsolationForest, SMOTE via imbalanced-learn), SHAP, Pandas, NumPy, SQLAlchemy, Matplotlib

---

## Project Structure

```
fraud-detection-system/
├── src/
│   ├── generate_data.py      # synthetic transaction generator (dev/testing)
│   ├── features.py           # feature engineering
│   ├── train_supervised.py   # XGBoost model with temporal CV
│   ├── train_anomaly.py      # Isolation Forest on legitimate transactions
│   ├── ensemble.py           # blend scores + alert generation
│   └── evaluate.py           # temporal eval, cost-threshold search, SHAP
├── config.yaml
├── requirements.txt
└── .gitignore
```

---

## Quickstart

```bash
git clone https://github.com/bhanupriyamaddali/fraud-detection-system
cd fraud-detection-system
pip install -r requirements.txt

python src/generate_data.py          # generate synthetic transactions
python src/train_supervised.py       # train XGBoost
python src/train_anomaly.py          # train Isolation Forest
python src/ensemble.py               # score transactions and output alerts
```
