"""
models.py – ML signal generators for Alpha Lab
==============================================

This module trains three forecasting / classification models that Alpha Lab
uses to turn Reddit–sentiment + price data into trading signals.

Currently implemented
---------------------
1. **Bayesian Ridge “direction” classifier**
   • Features = sentiment_t, Δsentiment, 3-day rolling mean
   • Target  = 1 if next-day return > 0 else 0
   • `predict_bayes()` returns **P(up)**.

2. **ARIMA (1,0,0) with exogenous sentiment** (`statsmodels`)
   • Forecasts next-day **percent return**.
   • In *app.py* the dashboard maps the *Probability / forecast threshold*
     slider ∈ [0.50 … 0.90] to a **return threshold**
     `ret_thresh = (slider - 0.5) / 10  ≈ ±2 % … ±4 %`.
     A forecast above `+ret_thresh` (“long”) or below `–ret_thresh` (“flat”)
     turns into an entry signal.

3. **Prophet with sentiment regressor** (`prophet`)
   • Enabled **weekly** and **yearly** seasonality (`weekly_seasonality=True`,
     `yearly_seasonality=True`).  No extra custom weekly term is added, so
     there is **one** weekly component with Fourier order = 3 (Prophet’s
     default).
   • Dashboard uses the *same* return-threshold mapping as ARIMA.

Install deps
------------
`pip install scikit-learn statsmodels prophet`

Public API
----------
train_bayes(df)                 → fitted_model
predict_bayes(model, row)       → prob_up ∈ [0,1]

train_arima(df)                 → fitted_model
forecast_arima(model, sent)     → ΔP % forecast

train_prophet(df)               → fitted_model
forecast_prophet(model, ds, s)  → ΔP % forecast
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 1. Bayesian-ridge direction classifier
# ------------------------------------------------------------------
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """sent_t, Δsent (lag-1), 3-day rolling mean"""
    tmp = df.copy()
    tmp["d_sent"] = tmp["sentiment"].diff().fillna(0)
    tmp["roll3"] = tmp["sentiment"].rolling(3).mean().fillna(tmp["sentiment"])
    return tmp[["sentiment", "d_sent", "roll3"]]


def train_bayes(df: pd.DataFrame):
    """Target: 1 if next-day return > 0 else 0"""
    feat = _feature_engineering(df)
    y = (df["ret_pct"].shift(-1) > 0).astype(int)
    X = feat.values[:-1]
    y = y.values[:-1]
    pipe = make_pipeline(StandardScaler(), BayesianRidge())
    pipe.fit(X, y)
    return pipe


def predict_bayes(model, sent_row: pd.Series) -> float:
    """Return P(up) using the same engineered features on single row"""
    feat_row = pd.DataFrame([sent_row])  # series → df(1×3)
    prob = model.predict(feat_row)[0]
    # map regression output (~real line) to [0,1] via logistic
    prob_up = 1 / (1 + np.exp(-prob))
    return float(np.clip(prob_up, 0, 1))


# ------------------------------------------------------------------
# 2. ARIMA(1,0,0) + sentiment exogenous
# ------------------------------------------------------------------
from statsmodels.tsa.arima.model import ARIMA


def train_arima(df: pd.DataFrame):
    """
    Fit ARIMA(1,0,0) on daily returns with sentiment as exogenous.
    We forecast next-day % return.
    """
    exog = df["sentiment"]
    endog = df["ret_pct"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(endog, order=(1, 0, 0), exog=exog).fit()
    return model


def forecast_arima(model, next_sent: float) -> float:
    """One-step forecast of % return given exogenous sentiment value."""
    fc = model.forecast(steps=1, exog=[next_sent])
    return float(fc.iloc[0])


# ------------------------------------------------------------------
# 3. Prophet with sentiment regressor
# ------------------------------------------------------------------
try:
    from prophet import Prophet
    _PROPHET_AVAILABLE = True
except ModuleNotFoundError:
    _PROPHET_AVAILABLE = False


def train_prophet(df: pd.DataFrame):
    """
    Fit a **Prophet** model on daily returns (column ``ret_pct`` renamed to
    ``y``) with **sentiment** as an extra regressor.

    Seasonality
    -----------
    * ``weekly_seasonality=True`` – default weekly term (Fourier order = 3)
    * ``yearly_seasonality=True`` – default yearly term  (Fourier order = 10)

    We *do not* add a second custom ``"weekly"`` component to avoid duplicate
    names/warnings – one weekly term is normally sufficient for equities.

    Parameters
    ----------
    df : DataFrame
        Must contain an index that is a DatetimeIndex plus columns
        ``sentiment`` and ``ret_pct``.

    Returns
    -------
    prophet.Prophet
        Fitted model ready for one-step-ahead forecasting.
    """
    if not _PROPHET_AVAILABLE:
        raise ImportError("prophet not installed.  `pip install prophet`")

    # --- build Prophet-friendly dataframe -------------------------
    dff = df.copy()
    dff["ds"] = dff.index               # always create the date column
    dff = dff.reset_index(drop=True)
    dff = dff.rename(columns={"ret_pct": "y"})
    dff["sentiment"] = dff["sentiment"].astype(float)

    # --- train ----------------------------------------------------
    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    m.add_regressor("sentiment")


    # Prophet complains about pandas warnings; silence them
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(dff[["ds", "y", "sentiment"]])

    return m
def forecast_prophet(model, next_date: pd.Timestamp, next_sent: float) -> float:
    """
    One-step forecast of **next-day percent return** using the fitted Prophet
    model.

    Dashboard mapping
    -----------------
    In *app.py* the slider labelled **“Probability / forecast threshold”**
    covers 0.50 → 0.90.
    We convert it to a symmetric return threshold::

        ret_thresh = (slider - 0.50) / 10        # ≈  ±0.02 … ±0.04

    * If ``forecast >  ret_thresh``  → long entry signal
    * If ``forecast < -ret_thresh``  → flat / exit

    Parameters
    ----------
    model : prophet.Prophet
    next_date : Timestamp
        The date we want to forecast (usually *today + 1 business day*).
    next_sent : float
        The accompanying Reddit-sentiment value to feed as exogenous regressor.

    Returns
    -------
    float
        Forecasted **percent price change** for that day.
    """
    df_future = pd.DataFrame({"ds": [next_date], "sentiment": [next_sent]})
    fc = model.predict(df_future)
    return float(fc["yhat"].iloc[0])
