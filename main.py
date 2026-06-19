#!/usr/bin/env python3
"""
FastAPI service — WC 2026 XGBoost + Poisson + Elo predictor.
Blend: 60% XGBoost (learned patterns) + 40% Poisson (statistical model).
"""

import json
import math
import os
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
ELO_PATH   = MODEL_DIR / "elo_ratings.json"
WC_AVG     = 1.28   # historical WC goals per team per game
XGB_WEIGHT = 0.60   # blend: 60% XGBoost + 40% Poisson

state: dict = {}


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        print("Model not found — running train.py...")
        subprocess.run(["python", "train.py"], check=True)

    print("Loading model...")
    payload = joblib.load(MODEL_PATH)
    state["model"]         = payload["model"]
    state["feature_names"] = payload["feature_names"]

    print("Loading Elo ratings...")
    with open(ELO_PATH) as f:
        state["elo"] = json.load(f)

    print(f"Ready — {len(state['elo'])} teams in Elo database.")
    yield
    state.clear()


app = FastAPI(
    title="WC 2026 Predictor",
    description="XGBoost + Poisson + Elo football match predictor",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class FormStats(BaseModel):
    wins:   int   = 0
    draws:  int   = 0
    losses: int   = 0
    gf:     float = 0.0
    ga:     float = 0.0
    played: int   = 0

class H2HStats(BaseModel):
    played: int = 0
    wins1:  int = 0
    draws:  int = 0
    wins2:  int = 0

class PredictRequest(BaseModel):
    team1: str = Field(..., example="Mexico")
    team2: str = Field(..., example="South Korea")
    form1: Optional[FormStats] = None
    form2: Optional[FormStats] = None
    h2h:  Optional[H2HStats]  = None
    is_wc:              bool = True
    is_major_tournament: bool = True

class PredictResponse(BaseModel):
    team1:       str
    team2:       str
    prob_team1:  float
    prob_draw:   float
    prob_team2:  float
    lambda1:     float
    lambda2:     float
    prob_over25: float
    prob_btts:   float
    confidence:  str
    elo1:        float
    elo2:        float
    xgb_probs:   list[float]   # raw XGBoost [win, draw, loss]
    poisson_probs: list[float] # raw Poisson  [win, draw, loss]
    model_used:  str


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_elo(name: str) -> float:
    db = state.get("elo", {})
    if name in db:
        return db[name]
    nl = name.lower().strip()
    for k, v in db.items():
        if k.lower() == nl:
            return v
    for k, v in db.items():
        if nl in k.lower() or k.lower() in nl:
            return v
    return 1600.0   # fallback for unknown teams


def build_features(req: PredictRequest, e1: float, e2: float) -> dict:
    """Convert request into the same feature vector used during training."""
    exp_h = 1.0 / (1.0 + 10.0 ** ((e2 - e1) / 400.0))

    f1 = req.form1
    if f1 and f1.played > 0:
        h_wr = f1.wins   / f1.played
        h_dr = f1.draws  / f1.played
        h_gf = f1.gf     / f1.played
        h_ga = f1.ga     / f1.played
        h_gd = (f1.gf - f1.ga) / f1.played
        h_n  = min(f1.played, 8) / 8.0
    else:
        h_wr = h_dr = 0.33
        h_gf = h_ga = 1.15
        h_gd = 0.0
        h_n  = 0.0

    f2 = req.form2
    if f2 and f2.played > 0:
        a_wr = f2.wins   / f2.played
        a_dr = f2.draws  / f2.played
        a_gf = f2.gf     / f2.played
        a_ga = f2.ga     / f2.played
        a_gd = (f2.gf - f2.ga) / f2.played
        a_n  = min(f2.played, 8) / 8.0
    else:
        a_wr = a_dr = 0.33
        a_gf = a_ga = 1.15
        a_gd = 0.0
        a_n  = 0.0

    h2h = req.h2h
    if h2h and h2h.played > 0:
        h2h_wr = h2h.wins1   / h2h.played
        h2h_n  = min(h2h.played, 6) / 6.0
    else:
        h2h_wr = 0.33
        h2h_n  = 0.0

    return {
        "elo_diff":           (e1 - e2) / 400.0,
        "elo_home_norm":      e1 / 2000.0,
        "elo_away_norm":      e2 / 2000.0,
        "elo_expected_home":  exp_h,
        "h_win_rate":  h_wr, "h_draw_rate": h_dr,
        "h_avg_gf":    h_gf, "h_avg_ga":    h_ga,
        "h_avg_gd":    h_gd, "h_form_n":    h_n,
        "a_win_rate":  a_wr, "a_draw_rate": a_dr,
        "a_avg_gf":    a_gf, "a_avg_ga":    a_ga,
        "a_avg_gd":    a_gd, "a_form_n":    a_n,
        "gf_diff":       h_gf - a_gf,
        "ga_diff":       h_ga - a_ga,
        "win_rate_diff": h_wr - a_wr,
        "h2h_win_rate": h2h_wr,
        "h2h_played":   h2h_n,
        "is_wc":               1 if req.is_wc else 0,
        "is_major_tournament": 1 if req.is_major_tournament else 0,
        "is_friendly":         0,
    }


def compute_lambda(e1: float, e2: float,
                   f1: Optional[FormStats], f2: Optional[FormStats]) -> tuple[float, float]:
    """Expected goals via Poisson λ = att × def × WC_avg, adjusted by Elo."""
    att1 = (f1.gf / f1.played / WC_AVG) if (f1 and f1.played > 0) else 1.0
    def1 = (f1.ga / f1.played / WC_AVG) if (f1 and f1.played > 0) else 1.0
    att2 = (f2.gf / f2.played / WC_AVG) if (f2 and f2.played > 0) else 1.0
    def2 = (f2.ga / f2.played / WC_AVG) if (f2 and f2.played > 0) else 1.0

    elo_adj = (e1 - e2) / 800.0
    no_data = (not f1 or f1.played == 0) and (not f2 or f2.played == 0)

    if no_data:
        L1 = WC_AVG * (1 + elo_adj * 0.5)
        L2 = WC_AVG * (1 - elo_adj * 0.5)
    else:
        L1 = att1 * def2 * WC_AVG * (1 + elo_adj * 0.15)
        L2 = att2 * def1 * WC_AVG * (1 - elo_adj * 0.15)

    L1 = max(0.30, min(4.0, L1))
    L2 = max(0.30, min(4.0, L2))
    return L1, L2


def poisson_probs(L1: float, L2: float, max_g: int = 8) -> tuple:
    """Compute P(win), P(draw), P(loss), P(over2.5), P(BTTS) from Poisson."""
    def lf(n):
        r = 0.0
        for i in range(2, n + 1): r += math.log(i)
        return r

    def pmf(lam, k):
        return math.exp(-lam + k * math.log(max(lam, 1e-9)) - lf(k))

    pW = pD = pL = pU = 0.0
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            p = pmf(L1, i) * pmf(L2, j)
            if   i > j: pW += p
            elif i == j: pD += p
            else:        pL += p
            if i + j <= 2: pU += p

    tot = pW + pD + pL
    pW /= tot; pD /= tot; pL /= tot
    pBTTS = (1 - math.exp(-L1)) * (1 - math.exp(-L2))
    return pW, pD, pL, 1 - pU, pBTTS


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": "model" in state,
        "teams_in_db":  len(state.get("elo", {})),
    }


@app.get("/elo/{team_name}")
def get_elo(team_name: str):
    return {"team": team_name, "elo": round(find_elo(team_name), 1)}


@app.get("/teams")
def top_teams(limit: int = 32):
    db  = state.get("elo", {})
    top = sorted(db.items(), key=lambda x: x[1], reverse=True)[:limit]
    return {"teams": [{"team": t, "elo": round(e, 1)} for t, e in top]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if "model" not in state:
        raise HTTPException(503, "Model not loaded yet")

    model         = state["model"]
    feature_names = state["feature_names"]

    # ── Elo lookup ─────────────────────────────────────────────────────────
    e1 = find_elo(req.team1)
    e2 = find_elo(req.team2)

    # ── XGBoost prediction ─────────────────────────────────────────────────
    feat = build_features(req, e1, e2)
    X    = np.array([[feat[n] for n in feature_names]], dtype=np.float32)
    xgb_p = model.predict_proba(X)[0]   # [p_win, p_draw, p_loss]

    # ── Poisson + Elo prediction ───────────────────────────────────────────
    L1, L2 = compute_lambda(e1, e2, req.form1, req.form2)
    pW, pD, pL, p_over25, p_btts = poisson_probs(L1, L2)

    # ── Blend 60/40 ────────────────────────────────────────────────────────
    w = XGB_WEIGHT
    p1  = w * float(xgb_p[0]) + (1 - w) * pW
    pd_ = w * float(xgb_p[1]) + (1 - w) * pD
    p2  = w * float(xgb_p[2]) + (1 - w) * pL

    tot = p1 + pd_ + p2
    p1 /= tot; pd_ /= tot; p2 /= tot

    lead = max(p1, pd_, p2)
    conf = "Alta" if lead > 0.55 else ("Media" if lead > 0.40 else "Baja")

    return PredictResponse(
        team1=req.team1,        team2=req.team2,
        prob_team1=round(p1,4), prob_draw=round(pd_,4), prob_team2=round(p2,4),
        lambda1=round(L1,2),    lambda2=round(L2,2),
        prob_over25=round(p_over25,4), prob_btts=round(p_btts,4),
        confidence=conf,
        elo1=round(e1,1),       elo2=round(e2,1),
        xgb_probs=[round(float(x),4) for x in xgb_p],
        poisson_probs=[round(pW,4), round(pD,4), round(pL,4)],
        model_used=f"XGBoost({int(w*100)}%) + Poisson+Elo({int((1-w)*100)}%)",
    )