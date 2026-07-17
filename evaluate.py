"""
Synthetic transaction generator for dev and testing.
Production version connects to the data warehouse via SQLAlchemy.
"""
import pandas as pd
import numpy as np
import os

SEED = 42
N = 500_000
FRAUD_RATE = 0.015


def generate(n=N, fraud_rate=FRAUD_RATE, seed=SEED):
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    def legit(size):
        return {
            "amount":             rng.lognormal(4.0, 1.2, size),
            "hour_of_day":        rng.integers(7, 22, size),
            "day_of_week":        rng.integers(0, 7, size),
            "merchant_category":  rng.choice(["grocery","gas","restaurant","retail","online"], size),
            "is_foreign":         rng.binomial(1, 0.05, size),
            "txn_count_1h":       rng.poisson(1.1, size),
            "txn_count_24h":      rng.poisson(4.2, size),
            "account_age_days":   rng.integers(90, 3650, size),
            "is_fraud":           np.zeros(size, dtype=int),
        }

    def fraud(size):
        return {
            "amount":             rng.lognormal(5.8, 1.9, size),    # higher amounts
            "hour_of_day":        rng.choice([0,1,2,3,23], size),   # late night
            "day_of_week":        rng.integers(0, 7, size),
            "merchant_category":  rng.choice(["online","electronics","jewelry"], size),
            "is_foreign":         rng.binomial(1, 0.50, size),      # more foreign
            "txn_count_1h":       rng.poisson(6.5, size),           # velocity burst
            "txn_count_24h":      rng.poisson(18.0, size),
            "account_age_days":   rng.integers(1, 60, size),        # newer accounts
            "is_fraud":           np.ones(size, dtype=int),
        }

    df = pd.DataFrame({**legit(n_legit)}).append(
        pd.DataFrame({**fraud(n_fraud)}), ignore_index=True
    ) if False else pd.concat([
        pd.DataFrame(legit(n_legit)),
        pd.DataFrame(fraud(n_fraud))
    ], ignore_index=True)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["transaction_id"] = ["txn_" + str(i).zfill(8) for i in range(len(df))]

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/transactions.csv", index=False)
    print(f"Generated {len(df):,} transactions | fraud: {df['is_fraud'].sum():,} ({fraud_rate:.1%})")
    return df


if __name__ == "__main__":
    generate()
