"""
alpha_lab.py  –  strategy-simulation utilities for MarketPulse
----------------------------------------------------------------
Key API
-------
merge_price_sentiment(price_daily, trend_df) -> pd.DataFrame
run_manual_strategy(df, sent_threshold, price_drop_pct, price_rise_pct,
                    stop_loss_pct=0.0, initial_cap=1000) -> (equity_df, trades_df)
show_results(equity_df, trades_df) -> None   # renders in Streamlit
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import models          # ML models file


# ------------------------------------------------------------------
# 1. Merge price & sentiment (daily)
# ------------------------------------------------------------------
def merge_price_sentiment(
    price_daily: pd.Series,
    trend_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return DataFrame with columns:
        price, sentiment, ret_pct
    Missing sentiment forward/back-filled.
    """
    df = (
        pd.DataFrame({"price": price_daily})
        .join(trend_df.set_index("day")["sentiment"], how="left")
        .fillna(method="ffill")
        .fillna(method="bfill")
    )
    df["ret_pct"] = df["price"].pct_change().fillna(0)
    return df


# ------------------------------------------------------------------
# 2. Back-test engine (manual rule)
# ------------------------------------------------------------------
def run_manual_strategy(
    df: pd.DataFrame,
    sent_threshold: float,
    price_drop_pct: float,
    price_rise_pct: float,
    stop_loss_pct: float = 0.0,
    initial_cap: float = 1_000,
):
    """
    Very simple long-only strategy:
      • Enter long on next open when:
          sentiment_prev_day > sent_threshold  AND  today's return ≤ −price_drop_pct
      • Exit when:
          sentiment_prev_day < 0             OR  today's return ≥  price_rise_pct
          OR trailing stop-loss hit
    """
    cash, pos = initial_cap, 0.0
    entry_px = None
    equity_curve = []
    trades = []  # dicts: entry_dt, exit_dt, entry_px, exit_px, pl_pct

    for i in range(1, len(df)):
        row_prev, row = df.iloc[i - 1], df.iloc[i]
        date_idx = row.name

        entry_sig = (
            (row_prev["sentiment"] > sent_threshold)
            and (row["ret_pct"] <= -price_drop_pct)
            and cash > 0
        )

        exit_sig = (
            pos > 0
            and (
                (row_prev["sentiment"] < 0)
                or (row["ret_pct"] >= price_rise_pct)
                or (stop_loss_pct and (row["price"] < entry_px * (1 - stop_loss_pct)))
            )
        )

        # --- execute -------------
        if entry_sig:
            pos = cash / row["price"]
            entry_px = row["price"]
            cash = 0
            trades.append(
                {"entry_dt": date_idx, "entry_px": entry_px,
                 "exit_dt": None, "exit_px": None, "pl_pct": None}
            )

        elif exit_sig:
            cash = pos * row["price"]
            trades[-1]["exit_dt"] = date_idx
            trades[-1]["exit_px"] = row["price"]
            trades[-1]["pl_pct"] = cash / entry_px / pos - 1
            pos = 0
            entry_px = None

        equity_curve.append(cash + pos * row["price"])

    df_bt = df.iloc[1:].copy()
    df_bt["equity"] = equity_curve

    # Benchmarks
    df_bt["bh_equity"] = initial_cap * (df_bt["price"] / df_bt["price"].iloc[0])
    rng = np.random.default_rng(42)
    random_long = rng.integers(0, 2, size=len(df_bt))  # 0/1 each day
    df_bt["random_equity"] = initial_cap * (
        (1 + df_bt["ret_pct"] * random_long).cumprod()
    )

    trades_df = pd.DataFrame(trades)
    return df_bt, trades_df


# ------------------------------------------------------------------
# 3. Risk/return metrics
# ------------------------------------------------------------------
def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    draw = equity / roll_max - 1
    return draw.min()

def _sharpe(ret_series: pd.Series, risk_free: float = 0.0) -> float:
    if ret_series.std() == 0:
        return np.nan
    return np.sqrt(252) * (ret_series.mean() - risk_free) / ret_series.std()

# ------------------------------------------------------------------
# 4. models
# ------------------------------------------------------------------
# ---------------------------------------------------------------
# helper – execute a pre-computed Boolean entry signal
# ---------------------------------------------------------------
def _exec_signal(df: pd.DataFrame,
                 signal: pd.Series,
                 initial_cap: float = 1_000):
    """
    Parameters
    ----------
    df      : DataFrame that contains 'price' (daily) columns.
    signal  : Boolean Series index-aligned with df; True == go long next open.
    Returns
    -------
    equity_df, trades_df  (same format as run_manual_strategy)
    """
    cash, pos = initial_cap, 0.0
    entry_px = None
    equity = []
    trades = []

    for i in range(1, len(df)):
        row_prev, row = df.iloc[i - 1], df.iloc[i]
        date_idx = row.name

        entry_sig = signal.iloc[i - 1] and cash > 0      # look at t-1
        exit_sig  = (not signal.iloc[i - 1]) and pos > 0 # flatten when signal off

        if entry_sig:
            pos = cash / row["price"]
            entry_px = row["price"]
            cash = 0
            trades.append({"entry_dt": date_idx, "entry_px": entry_px,
                           "exit_dt": None, "exit_px": None, "pl_pct": None})

        elif exit_sig:
            cash = pos * row["price"]
            trades[-1].update(exit_dt=date_idx,
                              exit_px=row["price"],
                              pl_pct=cash / (entry_px * pos) - 1)
            pos, entry_px = 0, None

        equity.append(cash + pos * row["price"])

    eq_df = df.iloc[1:].copy()
    eq_df["equity"] = equity
    # buy-and-hold & random baselines for comparability
    eq_df["bh_equity"] = initial_cap * (eq_df["price"] / eq_df["price"].iloc[0])
    rng = np.random.default_rng(42)
    rand = rng.integers(0, 2, len(eq_df))
    eq_df["random_equity"] = initial_cap * ((1 + eq_df["ret_pct"]*rand).cumprod())

    return eq_df, pd.DataFrame(trades)
def run_model_strategy_bayes(df: pd.DataFrame,
                             model,
                             prob_thresh: float = 0.70,
                             initial_cap: float = 1_000):
    # compute engineered features row-wise
    feats = models._feature_engineering(df)
    probs = model.predict(feats)
    probs = 1 / (1 + np.exp(-probs))        # logistic
    sig = pd.Series(probs, index=df.index) > prob_thresh
    return _exec_signal(df, sig, initial_cap)
def run_model_strategy_arima(df: pd.DataFrame,
                             model,
                             ret_thresh: float = 0.02,
                             initial_cap: float = 1_000):
    fc = model.forecast(steps=len(df), exog=df["sentiment"])
    entry_signal = fc > ret_thresh                 # long bias if forecast > +2 %
    return _exec_signal(df, entry_signal, initial_cap)
def run_model_strategy_prophet(df, model, next_date, ret_thresh=0.02, initial_cap=1_000):
    # generate one-step-ahead rolling forecast
    fc = []
    for i in range(len(df)):
        fc.append(models.forecast_prophet(model, df.index[i], df["sentiment"].iloc[i]))
    signal = pd.Series(fc, index=df.index) > ret_thresh
    return _exec_signal(df, signal, initial_cap)


# ------------------------------------------------------------------
# 5. Streamlit display
# ------------------------------------------------------------------
def show_results(equity_df: pd.DataFrame, trades_df: pd.DataFrame):
    st.subheader("📈 Strategy Performance")

    # --- headline metrics ---
    tot_ret = equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0] - 1
    cagr = (1 + tot_ret) ** (252 / len(equity_df)) - 1
    mdd = _max_drawdown(equity_df["equity"])
    sharpe = _sharpe(equity_df["equity"].pct_change().dropna())

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Total return", f"{tot_ret*100:.1f}%")
    colB.metric("CAGR", f"{cagr*100:.1f}%")
    colC.metric("Max draw-down", f"{mdd*100:.1f}%")
    colD.metric("Sharpe", f"{sharpe:.2f}")

    # --- equity curves ---
    fig = px.line(
        equity_df[["equity", "bh_equity", "random_equity"]],
        title="Cumulative equity vs benchmarks",
        labels={"value": "Equity ($)", "index": "Date", "variable": "Series"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- trade log ---
    st.subheader("📜 Trade log")
    if trades_df.empty:
        st.info("No trades executed for the selected rules.")
    else:
        trades_df_disp = trades_df.dropna(subset=["pl_pct"]).copy()
        trades_df_disp["pl_pct"] = (trades_df_disp["pl_pct"] * 100).round(2)
        st.dataframe(trades_df_disp, use_container_width=True, hide_index=True)

        wins = trades_df_disp["pl_pct"] > 0
        win_rate = wins.mean() if len(wins) else np.nan
        avg_win = trades_df_disp.loc[wins, "pl_pct"].mean()
        avg_loss = trades_df_disp.loc[~wins, "pl_pct"].mean()

        st.write(
            f"**Win rate:** {win_rate*100:.1f}% &nbsp;&nbsp; | "
            f"**Avg win:** {avg_win:.2f}% &nbsp;&nbsp; | "
            f"**Avg loss:** {avg_loss:.2f}%"
        )
