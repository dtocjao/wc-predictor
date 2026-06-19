#!/usr/bin/env python3
"""
Training script for the WC 2026 XGBoost match predictor.

Downloads 45,000+ historical international match results, builds dynamic
Elo ratings, computes rolling form features, and trains XGBoost.

Run once at build time:  python train.py
"""

import json
from collections import defaultdict, deque
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
ELO_PATH   = MODEL_DIR / "elo_ratings.json"

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
LOOKBACK = 8    # rolling form window
MIN_YEAR = 1993 # ignore pre-1993 (pre-modern football)

K_FACTORS = {
    "FIFA World Cup": 40,
    "UEFA Euro": 35,
    "Copa América": 35,
    "Africa Cup of Nations": 32,
    "Asian Cup": 30,
    "Gold Cup": 28,
    "UEFA Nations League": 25,
    "Confederation Cup": 30,
    "FIFA World Cup qualification": 28,
    "UEFA Euro qualification": 22,
    "Copa América qualification": 22,
    "Friendly": 10,
}

def get_k(tournament: str) -> float:
    for key, k in K_FACTORS.items():
        if key in tournament:
            return float(k)
    return 20.0


def download_data() -> pd.DataFrame:
    print("Downloading match data...")
    r = requests.get(DATA_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df[df["date"].dt.year >= MIN_YEAR].sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df):,} matches ({df['date'].min().date()} to {df['date'].max().date()})")
    return df


def build_features(df: pd.DataFrame):
    """Single O(n) pass — no data leakage."""
    elo       = defaultdict(lambda: 1500.0)
    team_hist = defaultdict(lambda: deque(maxlen=LOOKBACK))
    h2h_hist  = defaultdict(lambda: deque(maxlen=6))

    rows   = []
    labels = []

    def form(hist):
        if not hist:
            return {"wr": 0.33, "dr": 0.33, "gf": 1.15, "ga": 1.15, "gd": 0.0, "n": 0}
        n  = len(hist)
        w  = sum(1 for m in hist if m["r"] == "W")
        d  = sum(1 for m in hist if m["r"] == "D")
        gf = sum(m["gf"] for m in hist) / n
        ga = sum(m["ga"] for m in hist) / n
        return {"wr": w/n, "dr": d/n, "gf": gf, "ga": ga, "gd": gf-ga, "n": n}

    for _, row in df.iterrows():
        h = row["home_team"]; a = row["away_team"]
        hs = int(row["home_score"]); as_ = int(row["away_score"])
        t  = str(row.get("tournament", ""))

        eh = elo[h]; ea = elo[a]
        exp_h = 1.0 / (1.0 + 10.0 ** ((ea - eh) / 400.0))

        hf = form(team_hist[h]); af = form(team_hist[a])

        key   = tuple(sorted([h, a]))
        h2h   = list(h2h_hist[key])
        h2h_n = len(h2h)
        h2h_w = sum(1 for m in h2h if m["w"] == h)

        is_wc    = 1 if ("FIFA World Cup" in t and "qualification" not in t.lower()) else 0
        is_major = 1 if any(x in t for x in ["World Cup","Euro","Copa América","Nations Cup","Gold Cup","African Cup","Asian Cup"]) else 0
        is_fr    = 1 if "Friendly" in t else 0

        rows.append({
            "elo_diff":           (eh - ea) / 400.0,
            "elo_home_norm":      eh / 2000.0,
            "elo_away_norm":      ea / 2000.0,
            "elo_expected_home":  exp_h,
            "h_win_rate":  hf["wr"], "h_draw_rate": hf["dr"],
            "h_avg_gf":    hf["gf"], "h_avg_ga":    hf["ga"],
            "h_avg_gd":    hf["gd"], "h_form_n":    hf["n"] / LOOKBACK,
            "a_win_rate":  af["wr"], "a_draw_rate": af["dr"],
            "a_avg_gf":    af["gf"], "a_avg_ga":    af["ga"],
            "a_avg_gd":    af["gd"], "a_form_n":    af["n"] / LOOKBACK,
            "gf_diff":       hf["gf"] - af["gf"],
            "ga_diff":       hf["ga"] - af["ga"],
            "win_rate_diff": hf["wr"] - af["wr"],
            "h2h_win_rate": h2h_w / max(h2h_n, 1),
            "h2h_played":   min(h2h_n, 6) / 6.0,
            "is_wc":               is_wc,
            "is_major_tournament": is_major,
            "is_friendly":         is_fr,
        })

        labels.append(0 if hs > as_ else (1 if hs == as_ else 2))

        # update Elo
        k     = get_k(t)
        score = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        delta = k * (score - exp_h)
        elo[h] = eh + delta
        elo[a] = ea - delta

        # update histories
        hr = "W" if hs > as_ else ("D" if hs == as_ else "L")
        ar = "L" if hr == "W" else ("D" if hr == "D" else "W")
        team_hist[h].append({"r": hr, "gf": hs, "ga": as_})
        team_hist[a].append({"r": ar, "gf": as_, "ga": hs})
        winner = h if hs > as_ else (a if as_ > hs else "draw")
        h2h_hist[key].append({"w": winner})

    feat_df   = pd.DataFrame(rows)
    final_elo = {t: float(v) for t, v in elo.items()}
    return feat_df, np.array(labels), final_elo


def train(feat_df, labels):
    split = int(len(feat_df) * 0.85)
    X = feat_df.values.astype(np.float32)
    y = labels
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    dist = np.bincount(y) / len(y)
    print(f"Training {len(X_tr):,} | Validation {len(X_val):,}")
    print(f"Labels: home={dist[0]:.2f} draw={dist[1]:.2f} away={dist[2]:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=800, max_depth=5, learning_rate=0.015,
        subsample=0.80, colsample_bytree=0.80, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        early_stopping_rounds=30, random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)

    probs = model.predict_proba(X_val)
    print(f"Accuracy: {accuracy_score(y_val, np.argmax(probs,1)):.4f}")
    print(f"Log-loss: {log_loss(y_val, probs):.4f}")
    print(f"Best iteration: {model.best_iteration}")

    names = list(feat_df.columns)
    imp   = sorted(zip(names, model.feature_importances_), key=lambda x: -x[1])
    print("\nTop features:")
    for n, s in imp[:10]:
        print(f"  {n:<28} {s:.4f}")

    return model, names


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = download_data()
    print("Building features...")
    feat_df, labels, final_elo = build_features(df)
    model, feat_names = train(feat_df, labels)
    joblib.dump({"model": model, "feature_names": feat_names}, MODEL_PATH)
    with open(ELO_PATH, "w") as f:
        json.dump(final_elo, f)
    top = sorted(final_elo.items(), key=lambda x: -x[1])[:15]
    print("\nTop 15 by Elo:")
    for team, elo in top:
        print(f"  {team:<30} {elo:.0f}")
    print(f"\nDone. {MODEL_PATH} | {len(final_elo)} teams")


if __name__ == "__main__":
    main()