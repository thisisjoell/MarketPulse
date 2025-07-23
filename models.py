#!/usr/bin/env python
# models.py — Alpha‑Lab signal generators  (2025‑07‑21)
# -----------------------------------------------------
"""
Fast, lightweight ML models that map <price + Reddit‑sentiment> ➜ trading
signals.  Public API is 100 % compatible with the original release.

New in this version
-------------------
1. Logistic‑regression now **optionally calibrated** (isotonic) and ships an
   on‑demand SHAP explainer for feature inspection.
2. Added **LightGBM** classifier (`train_lgbm`, `predict_lgbm`) as a faster /
   stronger alternative to vanilla Gradient‑Boosting.
3. ARIMA micro‑grid uses constant‑time **row cap + n_jobs** parallel fit
   and faster statsmodels flags (`trend="n"`, no stationarity enforcement).
4. Prophet keeps the sentiment regressor but trims history to ≤ 1 200 rows
   for sub‑second fits on Streamlit Cloud.

Install
-------
pip install scikit-learn statsmodels prophet lightgbm shap
               # ^ last two are optional but recommended
"""
from __future__ import annotations
import streamlit as st
import warnings, joblib, numpy as np, pandas as pd
from typing import Any, List

# ----------------------------------------------------------------------
# 💡  Feature engineering
# ----------------------------------------------------------------------
def _fe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix:
      • Sentiment dynamics
      • Simple price momentum / vol
      • Basic TA indicators via `ta`
    """
    import ta  # tiny dep already in requirements

    t = df.copy()

    # sentiment history
    t["sent_lag1"] = t["sentiment"].shift(1)
    t["sent_lag2"] = t["sentiment"].shift(2)
    t["d_sent"]    = t["sentiment"] - t["sent_lag1"]
    t["roll3"]     = t["sentiment"].rolling(3).mean()

    # price momentum & vol
    t["mom5"]   = t["price"].pct_change(5)
    t["mom10"]  = t["price"].pct_change(10)
    t["vol10"]  = t["price"].pct_change().rolling(10).std()

    # interactions
    t["sent_x_mom5"]  = t["sentiment"] * t["mom5"]
    t["sent_x_vol10"] = t["sentiment"] * t["vol10"]

    # TA
    t["rsi_14"]    = ta.momentum.rsi(t["price"], window=14)
    t["macd_hist"] = ta.trend.macd_diff(
        t["price"], window_slow=26, window_fast=12, window_sign=9
    )

    return t.fillna(0.0)[
        [
            "sentiment", "sent_lag1", "sent_lag2", "d_sent", "roll3",
            "mom5", "mom10", "vol10",
            "sent_x_mom5", "sent_x_vol10",
            "rsi_14", "macd_hist",
        ]
    ]
# ------------------------------------------------------------------
# Keep legacy name so alpha_lab & old notebooks don't break
# ------------------------------------------------------------------
def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:   # noqa: N802
    """Deprecated alias – use `_fe` going forward."""
    return _fe(df)
# ----------------------------------------------------------------------
# 1.  Logistic‑regression (+ isotonic calibration + SHAP)
# ----------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

def _grid_logistic() -> GridSearchCV:
    return GridSearchCV(
        Pipeline(
            steps=[
                ("imp", SimpleImputer(strategy="constant", fill_value=0)),
                ("sc",  StandardScaler()),
                ("log", LogisticRegression(solver="lbfgs", max_iter=400)),
            ]
        ),
        param_grid={
            "log__C":            [0.1, 0.3, 1, 3, 10],
            "log__class_weight": [None, "balanced"],
        },
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        refit=True,
    )

def train_logistic(
    df: pd.DataFrame,
    calibrate: bool = True,
) -> Pipeline | CalibratedClassifierCV:
    X = _fe(df).iloc[:-3]
    y = (df["ret_pct"].shift(-3) > 0).astype(int).iloc[:-3]

    best = _grid_logistic().fit(X, y).best_estimator_
    if calibrate:
        # isotonic keeps monotonicity – important for prob‑threshold slider
        return CalibratedClassifierCV(best, method="isotonic", cv=3).fit(X, y)
    return best

def predict_logistic(model, row: pd.Series) -> float:
    return float(model.predict_proba(_fe(pd.DataFrame([row])).iloc[:1])[0, 1])

# --- SHAP helper (optional) ---------------------------------------------------
def shap_values(model, df_sample: pd.DataFrame, max_obs: int = 500):
    """
    Return SHAP values for a sample of rows.
    Requires `shap` – will raise if not installed.
    """
    import shap
    X = _fe(df_sample).head(max_obs)
    explainer = shap.Explainer(model, X, feature_names=X.columns, seed=42)
    return explainer(X)

# public aliases ---------------------------------------------------------------
train_bayes   = train_logistic
predict_bayes = predict_logistic

# ----------------------------------------------------------------------
# 1‑bis.  LightGBM classifier  (optional)
# ----------------------------------------------------------------------
try:
    import lightgbm as lgb
    _LGB_OK = True
except ModuleNotFoundError:
    _LGB_OK = False

def train_lgbm(
    df: pd.DataFrame,
    params: dict | None = None,
) -> Any:
    if not _LGB_OK:
        raise ImportError("lightgbm not installed; `pip install lightgbm`")

    X = _fe(df).iloc[:-3]
    y = (df["ret_pct"].shift(-3) > 0).astype(int).iloc[:-3]

    params = params or dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=400,
        seed=42,
    )
    return lgb.LGBMClassifier(**params).fit(X, y)

def predict_lgbm(model, row: pd.Series) -> float:
    return float(model.predict_proba(_fe(pd.DataFrame([row])).iloc[:1])[0, 1])

# ----------------------------------------------------------------------
# 2.  Fast micro‑grid ARIMA  (exogenous sentiment)
# ----------------------------------------------------------------------
from statsmodels.tsa.arima.model import ARIMA

_CANDS: List[tuple[int, int, int]] = [
    (1, 0, 0), (2, 0, 0), (1, 0, 1), (2, 0, 1),
    (0, 0, 1), (0, 1, 1), (1, 1, 0),
]
_MAX_ROWS = 1_000      # constant runtime

def _fit_one(order, y, ex):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdl = ARIMA(
            y, exog=ex, order=order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(method_kwargs={"maxiter": 40})
    return order, mdl.aic, mdl

def train_arima(df: pd.DataFrame):
    y = df["ret_pct"].astype(float)
    ex = df[["sentiment"]].astype(float)
    if len(df) > _MAX_ROWS:
        y, ex = y.iloc[-_MAX_ROWS:], ex.iloc[-_MAX_ROWS:]

    res = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_fit_one)(o, y, ex) for o in _CANDS
    )
    return min(res, key=lambda t: t[1])[2]

def forecast_arima(model, next_sent: float) -> float:
    return float(model.forecast(steps=1, exog=[[next_sent]]).iloc[0])

# ----------------------------------------------------------------------
# 3.  Prophet  (sentiment regressor, row‑capped)
# ----------------------------------------------------------------------
try:
    from prophet import Prophet
    _PROPHET = True
except ModuleNotFoundError:
    _PROPHET = False

_MAX_PR_ROWS = 1_200    # ≈ 5y of trading days

@st.cache_data(show_spinner=False,
               hash_funcs={pd.DataFrame: lambda d: (len(d), d.index[-1])})
def train_prophet(
    df: pd.DataFrame,
    weekly_order: int = 3,
    yearly_order: int = 10,
) -> Prophet:
    if not _PROPHET:
        raise ImportError("prophet not installed; `pip install prophet`")

    d = df.tail(_MAX_PR_ROWS).copy()
    d["ds"] = d.index
    d["y"]  = d["ret_pct"].astype(float)
    d["sentiment"] = d["sentiment"].astype(float)

    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_range=0.90,
    )
    m.add_regressor("sentiment")
    m.add_seasonality("weekly", period=7,     fourier_order=weekly_order)
    m.add_seasonality("yearly", period=365.25, fourier_order=yearly_order)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(d[["ds", "y", "sentiment"]])

    return m

def forecast_prophet(model: Prophet, next_date, next_sent: float) -> float:
    fc = model.predict(pd.DataFrame({"ds": [next_date], "sentiment": [next_sent]}))
    return float(fc["yhat"].iloc[0])
