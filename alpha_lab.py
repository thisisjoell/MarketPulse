"""
alpha_lab.py  –  strategy-simulation utilities for MarketPulse
----------------------------------------------------------------
Key API
-------
merge_price_sentiment(price_daily, trend_df) -> pd.DataFrame
run_manual_strategy(df, sent_threshold, price_drop_pct, price_rise_pct,
                    vol_thresh=0.02, mom_thresh=0.01,
                    stop_loss_pct=0.0, initial_cap=1000)
show_results(equity_df, trades_df) -> None   # renders in Streamlit
"""

from __future__ import annotations
import numpy as np
import talib as ta
from itertools import product
import pandas as pd
import plotly.express as px
import streamlit as st
import models          # ML models file
from sklearn.model_selection import TimeSeriesSplit
_TS_SPLIT = TimeSeriesSplit(n_splits=5)          # 5 expanding folds


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
# ------------------------------------------------------------------
# 2. Back-test engine (manual rule, multi-factor)
# ------------------------------------------------------------------
def run_manual_strategy(
    df: pd.DataFrame,
    sent_threshold: float,
    price_drop_pct: float,
    price_rise_pct: float,
    stop_loss_pct: float = 0.0,
    tx_cost_bps:  float = 0.0,          # ← NEW: one-way cost, e.g. 5 = 0.05 %
    initial_cap:  float = 1_000,
    vol_thresh:   float = 0.02,
    mom_thresh:   float = 0.01,
    rsi_max: float = 30,  # NEW – oversold
    macd_req: bool = False,  # NEW – bool flag
):
    """
    Long-only toy strategy.

    Entry (next open) when **all** are true
    --------------------------------------
    • sentiment_{t-1} > sent_threshold
    • return_t       ≤ −price_drop_pct
    • vol_10         >  vol_thresh          (volatility spike)
    • mom_5          < −mom_thresh          (bear momentum)

    Exit when
    ---------
    • sentiment_{t-1} < 0   OR
    • return_t       ≥  price_rise_pct  OR
    • trailing stop-loss triggered.

    Parameters
    ----------
    tx_cost_bps : float
        One-way transaction cost in basis-points (1 bp = 0.01 %).
    """
    df  = add_factors(df)                       # enrich with vol_10, mom_5 …
    bps = tx_cost_bps / 10_000                  # convert → proportion

    cash, pos = initial_cap, 0.0
    entry_px   = None
    equity_curve, trades = [], []

    for i in range(1, len(df)):
        row_prev, row = df.iloc[i - 1], df.iloc[i]
        date_idx = row.name

        entry_sig = (
                (row_prev["sentiment"] > sent_threshold) and
                (row["ret_pct"] <= -price_drop_pct) and
                (row["vol_10"] > vol_thresh) and
                (row["mom_5"] < -mom_thresh) and
                (row["rsi_14"] < rsi_max) and  # ▶ RSI filter
                (not macd_req or row["macd_hist"] > 0) and  # ▶ MACD bullish hist
                (row["bb_perc"] < 0.15)  # ▶ price near lower band
        )

        exit_sig = (
            pos > 0 and (
                (row_prev["sentiment"] < 0)            or
                (row["ret_pct"]        >= price_rise_pct) or
                (stop_loss_pct and row["price"] < entry_px * (1 - stop_loss_pct))
            )
        )

        # -------- execute --------
        if entry_sig and cash > 0:
            # pay cost on entry
            pos  = (cash * (1 - bps)) / row["price"]
            entry_px = row["price"]
            cash = 0
            trades.append(
                dict(entry_dt=date_idx, entry_px=entry_px,
                     exit_dt=np.nan,     exit_px=np.nan,  pl_pct=np.nan)
            )

        elif exit_sig:
            # pay cost on exit
            cash = pos * row["price"] * (1 - bps)
            trades[-1]["exit_dt"] = date_idx
            trades[-1]["exit_px"] = row["price"]
            trades[-1]["pl_pct"]  = cash / (entry_px * pos) - 1
            pos, entry_px = 0, None

        equity_curve.append(cash + pos * row["price"])

    # ---------- wrap-up ----------
    df_bt = df.iloc[1:].copy()
    df_bt["equity"] = equity_curve

    df_bt["bh_equity"] = initial_cap * (df_bt["price"] / df_bt["price"].iloc[0])

    rng = np.random.default_rng(42)
    rand_mask = rng.integers(0, 2, len(df_bt))
    df_bt["random_equity"] = initial_cap * (
        (1 + df_bt["ret_pct"] * rand_mask).cumprod()
    )

    trades_df = pd.DataFrame(
        trades,
        columns=["entry_dt", "exit_dt", "entry_px", "exit_px", "pl_pct"]
    )
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
# ---------------------------------------------------------------
# helper – execute a pre-computed Boolean entry signal
# ---------------------------------------------------------------
def _exec_signal(df: pd.DataFrame,
                 signal: pd.Series,
                 initial_cap: float = 1_000):
    """
    Convert a True/False entry signal into trades + equity curve.
    Returns
    -------
    eq_df   : DataFrame with equity, bh_equity, random_equity
    trades  : DataFrame (may be empty) with the 5 standard columns
    """
    cash, pos = initial_cap, 0.0
    entry_px = None
    equity, trades = [], []

    for i in range(1, len(df)):
        row_prev, row = df.iloc[i - 1], df.iloc[i]
        date_idx = row.name

        enter = signal.iloc[i - 1] and cash > 0
        exit_ = (not signal.iloc[i - 1]) and pos > 0

        if enter:
            pos = cash / row["price"]
            entry_px = row["price"]
            cash = 0
            trades.append(
                {"entry_dt": date_idx, "entry_px": entry_px,
                 "exit_dt": np.nan,     "exit_px": np.nan, "pl_pct":np.nan}
            )

        elif exit_:
            cash = pos * row["price"]
            trades[-1].update(
                exit_dt=date_idx,
                exit_px=row["price"],
                pl_pct=cash / (entry_px * pos) - 1
            )
            pos, entry_px = 0, None

        equity.append(cash + pos * row["price"])

    eq_df = df.iloc[1:].copy()
    eq_df["equity"]        = equity
    eq_df["bh_equity"]     = initial_cap * (eq_df["price"] / eq_df["price"].iloc[0])
    rng = np.random.default_rng(42)
    rand = rng.integers(0, 2, len(eq_df))
    eq_df["random_equity"] = initial_cap * ((1 + eq_df["ret_pct"] * rand).cumprod())

    # —— guarantee the columns even when *no* trade fired ——
    trades_df = pd.DataFrame(
        trades,
        columns=["entry_dt", "exit_dt", "entry_px", "exit_px", "pl_pct"]
    )

    return eq_df, trades_df
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
# 5. Walk-forward cross-validation
# ------------------------------------------------------------------
def walk_forward_backtest(
    df: pd.DataFrame,
    train_fn,
    strategy_fn,
    model_kwargs: dict | None = None,
    strat_kwargs: dict | None = None,
):
    """
    Expanding-window walk-forward back-test (TimeSeriesSplit, 5 folds).
    Returns stitched equity curve and trade log.  Safe if some folds
    generate zero trades.
    """
    model_kwargs = model_kwargs or {}
    strat_kwargs = strat_kwargs or {}

    eq_all, trades_all = [], []

    for fold, (train_idx, test_idx) in enumerate(_TS_SPLIT.split(df), start=1):
        train_df = df.iloc[train_idx]
        test_df  = df.iloc[test_idx]

        model = train_fn(train_df, **model_kwargs)
        eq, tr = strategy_fn(test_df, model, **strat_kwargs)

        # drop 1st row of subsequent folds to avoid duplicate timestamps
        if fold > 1:
            eq = eq.iloc[1:]
            if not tr.empty:
                tr = tr.iloc[1:]

        eq_all.append(eq)
        trades_all.append(tr.assign(fold=fold))

    # —— equity is always present ——
    equity = pd.concat(eq_all).sort_index()

    # —— trades may all be empty ——
    non_empty = [t for t in trades_all if not t.empty]
    if non_empty:
        trades = pd.concat(non_empty, ignore_index=True).sort_values("entry_dt")
    else:
        trades = pd.DataFrame(
            columns=["entry_dt", "exit_dt", "entry_px", "exit_px", "pl_pct", "fold"]
        )

    return equity, trades


# ------------------------------------------------------------------
# 6. Extension
# ------------------------------------------------------------------
def add_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich <price, sentiment> frame with common technical factors.
    Adds: vol_10, mom_5, z_20, rsi_14, macd_hist, bb_perc, bb_width.
    """
    out = df.copy()
    px  = out["price"].values.astype(float)      # numpy view for TA-Lib

    # ── basic rolling stats ───────────────────────────────────────
    out["vol_10"] = out["price"].pct_change().rolling(10).std()
    out["mom_5"]  = out["price"].pct_change(5)

    sma20 = out["price"].rolling(20).mean()
    std20 = out["price"].rolling(20).std()
    out["z_20"]   = (out["price"] - sma20) / std20

    # ── TA-Lib indicators ────────────────────────────────────────
    # 1) RSI-14
    out["rsi_14"] = ta.RSI(px, timeperiod=14)

    # 2) MACD histogram (12,26,9)
    macd, macd_sig, macd_hist = ta.MACD(px, fastperiod=12,
                                            slowperiod=26,
                                            signalperiod=9)
    out["macd_hist"] = macd_hist

    # 3) Bollinger Bands (20, 2 σ)
    upper, middle, lower = ta.BBANDS(px, timeperiod=20,
                                          nbdevup=2, nbdevdn=2)
    out["bb_low"]   = lower
    out["bb_high"]  = upper

    width           = upper - lower
    out["bb_perc"]  = (px - lower) / width
    out["bb_width"] = width / sma20

    # ── housekeeping ─────────────────────────────────────────────
    out["bb_perc"] = out["bb_perc"].clip(-0.1, 1.0)   # flat-market safeguard
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.bfill(inplace=True)                           # fill first 19 NaNs

    return out

def grid_search(
    df: pd.DataFrame,
    param_grid: dict,
    backtest_callable,
    base_kwargs: dict | None = None,
):
    """
    Run an exhaustive grid search.

    Parameters
    ----------
    df : DataFrame                    – price/sentiment frame
    param_grid : dict[str, list]      – each key = kwarg; values = grid
    backtest_callable : callable      – must return (equity_df, trades_df)
    base_kwargs : dict | None         – kwargs held constant across the grid
    """

    base_kwargs = base_kwargs or {}

    # --- safety: ensure mandatory arg present --------------------
    if "price_rise_pct" not in param_grid:
        param_grid = {**param_grid, "price_rise_pct": [0.05]}   # sensible default

    keys, grids = zip(*param_grid.items())
    rows, n_total = [], np.prod([len(g) for g in grids])
    pbar = st.progress(0, text="Grid search…")

    for i, vals in enumerate(product(*grids), start=1):
        kwargs = {**base_kwargs, **dict(zip(keys, vals))}
        equity, _ = backtest_callable(df, **kwargs)

        daily_ret = equity["equity"].pct_change().dropna()
        sharpe    = _sharpe(daily_ret)
        mdd_pct   = _max_drawdown(equity["equity"]) * 100
        totret_pct = (equity["equity"].iloc[-1] / equity["equity"].iloc[0] - 1) * 100


        rows.append({**kwargs,
                     "Sharpe":   round(sharpe, 3),
                     "MaxDD %":  round(mdd_pct, 1),
                     "TotalRet %": round(totret_pct, 1)})

        pbar.progress(i / n_total, text=f"Grid {i}/{n_total}")

    pbar.empty()
    return pd.DataFrame(rows).sort_values("Sharpe", ascending=False)


# ------------------------------------------------------------------
# Run the same strategy across many tickers
# ------------------------------------------------------------------
def batch_benchmark(
    tickers: list[str],
    fetch_fn,            # should return (price_series, trend_df)
    merge_fn,
    run_fn,
    **run_kwargs,
):
    """
    Benchmarks a single strategy across multiple tickers.

    NOTE
    ----
    Accuracy depends on the `fetch_fn` you pass in.
    The default demo fetch (`fetch_price_and_trend`) grabs
    per-ticker Reddit sentiment live, so each symbol is independent.
    """
    out, n = [], len(tickers)
    bar = st.progress(0, text="Benchmarking tickers…")

    for i, tic in enumerate(tickers, start=1):
        price, trend = fetch_fn(tic)
        df = merge_fn(price, trend)

        equity, _ = run_fn(df, **run_kwargs)
        daily = equity["equity"].pct_change().dropna()

        out.append({
            "Ticker":      tic,
            "Sharpe":      round(_sharpe(daily), 3),
            "MaxDD %":     round(_max_drawdown(equity["equity"]) * 100, 1),
            "TotalRet %":  round(
                (equity["equity"].iloc[-1]/equity["equity"].iloc[0]-1) * 100, 1
            )
        })
        bar.progress(i / n, text=f"Ticker {i}/{n}")

    bar.empty()
    return pd.DataFrame(out).sort_values("Sharpe", ascending=False)

def corr_matrix(price_map: dict[str, pd.Series]) -> pd.DataFrame:
    """
    price_map : {ticker -> price Series aligned on date}
    Returns    : DataFrame correlation matrix (levels on CLOSE returns)
    """
    df = pd.concat(price_map, axis=1)           # wide frame, columns = tickers
    ret = df.pct_change().dropna()
    return ret.corr()                           # Pearson by default


def rolling_corr_pair(a: pd.Series,
                      b: pd.Series,
                      win: int = 30) -> pd.Series:
    """
    30-day rolling correlation of two return series.
    Index = original DatetimeIndex aligned on **both** series.
    """
    df = pd.concat({"a": a, "b": b}, axis=1).pct_change().dropna()
    return df["a"].rolling(win).corr(df["b"])


def plot_corr_heat(corr_df: pd.DataFrame):
    """Nice heat-map with Plotly."""
    fig = px.imshow(
        corr_df.round(2),
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        title="Return correlation matrix"
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)


def plot_rolling_corr(rho: pd.Series, symA: str, symB: str):
    fig = px.line(rho, title=f"30-day rolling ρ: {symA} vs {symB}",
                  labels={"value": "Correlation", "index": "Date"})
    fig.add_hline(0, line_dash="dash", line_color="grey")
    st.plotly_chart(fig, use_container_width=True)
# ------------------------------------------------------------------
# 7. Streamlit display
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
    # Coerce pl_pct to numeric once, then drop / format
    trades_df["pl_pct"] = pd.to_numeric(trades_df["pl_pct"], errors="coerce")

    trades_df_disp = trades_df.dropna(subset=["pl_pct"]).copy()
    if trades_df_disp.empty:
        st.info("No trades executed for the selected rules.")
        return

    trades_df_disp["pl_pct"] = (trades_df_disp["pl_pct"] * 100).round(2)
    st.dataframe(trades_df_disp, use_container_width=True, hide_index=True)

    wins = trades_df_disp["pl_pct"] > 0
    win_rate = wins.mean()
    avg_win  = trades_df_disp.loc[wins,  "pl_pct"].mean()
    avg_loss = trades_df_disp.loc[~wins, "pl_pct"].mean()

    st.write(
        f"**Win rate:** {win_rate*100:.1f}% &nbsp;&nbsp; | "
        f"**Avg win:** {avg_win:.2f}% &nbsp;&nbsp; | "
        f"**Avg loss:** {avg_loss:.2f}%"
    )
