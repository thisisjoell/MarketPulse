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
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import numpy as np
import ta
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
    price_daily: pd.Series | pd.DataFrame,
    trend_df: pd.DataFrame,
) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    # Build price DataFrame
    if isinstance(price_daily, pd.DataFrame):
        if "Close" in price_daily.columns and price_daily.shape[1] == 1:
            df = price_daily.rename(columns={"Close": "price"})
        elif price_daily.shape[1] == 1:
            df = price_daily.rename(columns={price_daily.columns[0]: "price"})
        else:
            raise ValueError("price_daily DataFrame must have a single column.")
    else:
        df = price_daily.to_frame(name="price")

    # Try to ensure 'day' is present in trend_df
    if not trend_df.empty:
        if "day" not in trend_df.columns:
            # Try to use index as 'day'
            trend_df = trend_df.copy()
            trend_df["day"] = trend_df.index
        if "sentiment" not in trend_df.columns:
            raise ValueError("trend_df must have a 'sentiment' column.")
        sentiment = trend_df.set_index("day")["sentiment"]
        df = df.join(sentiment, how="left")
        df = df.ffill().bfill()
        df["sentiment"] = df["sentiment"].fillna(0.0)  # fill any remaining gaps
    else:
        # Trend is empty: default to neutral sentiment (0.0)
        df["sentiment"] = 0.0

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
    tx_cost_bps:  float = 0.0,
    initial_cap:  float = 1_000,
    vol_thresh:   float = 0.02,
    mom_thresh:   float = 0.01,
    rsi_max: float = 30,
    macd_req: bool = False,
):
    """
    Runs a long-only trading strategy with multi-factor entry and exit logic.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price, sentiment, and indicator columns (output of `add_factors`).
    sent_threshold : float
        Sentiment threshold for entry.
    price_drop_pct : float
        Minimum negative return (in percent) to trigger entry.
    price_rise_pct : float
        Positive return threshold for exit.
    stop_loss_pct : float, optional
        Trailing stop-loss (as a fraction, e.g. 0.02 for 2%).
    tx_cost_bps : float, optional
        One-way transaction cost in basis points (1 bp = 0.01%).
    initial_cap : float, optional
        Starting capital for the backtest.
    vol_thresh, mom_thresh, rsi_max, macd_req: floats/bool
        Additional indicator thresholds for entry.

    Returns
    -------
    df_bt : pd.DataFrame
        Copy of input DataFrame with columns:
            - 'equity': Strategy account value after each step.
            - 'bh_equity': Buy-and-hold equity curve.
            - 'random_equity': Random strategy baseline.

        **The equity curve always matches the DataFrame's index and length**.
        The first value is initial_cap; subsequent values reflect cash+position after each step.

    trades_df : pd.DataFrame
        DataFrame of executed trades with entry/exit details.
    """
    # --- Defensive: handle short DataFrames (0 or 1 row) ---
    if len(df) < 2:
        df_bt = df.copy()
        df_bt["equity"] = [initial_cap] * len(df_bt)
        df_bt["bh_equity"] = initial_cap * (df_bt["price"] / df_bt["price"].iloc[0])
        df_bt["random_equity"] = initial_cap
        trades_df = pd.DataFrame(columns=["entry_dt", "exit_dt", "entry_px", "exit_px", "pl_pct"])
        return df_bt, trades_df

    # Normal logic (now always runs on at least 2 rows)
    df  = add_factors(df)
    bps = tx_cost_bps / 10_000

    cash, pos = initial_cap, 0.0
    entry_px = None
    equity_curve, trades = [initial_cap], []

    for i in range(1, len(df)):
        row_prev, row = df.iloc[i - 1], df.iloc[i]
        date_idx = row.name

        entry_sig = (
            (row_prev["sentiment"] > sent_threshold) and
            (row["ret_pct"] <= -price_drop_pct) and
            (row["vol_10"] > vol_thresh) and
            (row["mom_5"] < -mom_thresh) and
            (row["rsi_14"] < rsi_max) and
            (not macd_req or row["macd_hist"] > 0) and
            (row["bb_perc"] < 0.15)
        )

        exit_sig = (
            pos > 0 and (
                (row_prev["sentiment"] < 0) or
                (row["ret_pct"] >= price_rise_pct) or
                (stop_loss_pct and row["price"] < entry_px * (1 - stop_loss_pct))
            )
        )

        if entry_sig and cash > 0:
            pos = (cash * (1 - bps)) / row["price"]
            entry_px = row["price"]
            cash = 0
            trades.append(
                dict(entry_dt=date_idx, entry_px=entry_px,
                     exit_dt=np.nan, exit_px=np.nan, pl_pct=np.nan)
            )

        elif exit_sig:
            cash = pos * row["price"] * (1 - bps)
            trades[-1]["exit_dt"] = date_idx
            trades[-1]["exit_px"] = row["price"]
            trades[-1]["pl_pct"]  = cash / (entry_px * pos) - 1
            pos, entry_px = 0, None

        equity_curve.append(cash + pos * row["price"])

    df_bt = df.copy()
    assert len(equity_curve) == len(df_bt), "Equity curve and DataFrame must be same length"

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
def _exec_signal(
    df: pd.DataFrame,
    signal: pd.Series,
    initial_cap: float = 1_000,
    tx_cost_bps: float = 0.0,          # ← NEW: cost per side (1 bp = 0.01 %)
):
    """
    Convert a Boolean entry signal into equity & trade log, **including
    transaction costs** on every entry and exit.

    Parameters
    ----------
    df            : DataFrame with at least a 'price' column.
    signal        : Series[bool] – True means “hold long” for that day.
    initial_cap   : Starting capital in dollars.
    tx_cost_bps   : One‑way cost in basis‑points (e.g. 5 = 0.05 %).

    Returns
    -------
    eq_df   : DataFrame with equity, bh_equity, random_equity.
    trades  : DataFrame (may be empty) with the 5 standard columns.
    """
    bps  = tx_cost_bps / 10_000        # convert → proportion
    cash, pos = initial_cap, 0.0
    entry_px = None
    equity, trades = [], []

    for i in range(1, len(df)):
        row_prev, row = df.iloc[i - 1], df.iloc[i]
        date_idx = row.name

        enter = signal.iloc[i - 1] and cash > 0
        exit_ = (not signal.iloc[i - 1]) and pos > 0

        if enter:
            # pay cost on entry
            pos = (cash * (1 - bps)) / row["price"]
            entry_px = row["price"]
            cash = 0
            trades.append(
                {"entry_dt": date_idx, "entry_px": entry_px,
                 "exit_dt": np.nan, "exit_px": np.nan, "pl_pct": np.nan}
            )

        elif exit_:
            # pay cost on exit
            cash = pos * row["price"] * (1 - bps)
            trades[-1].update(
                exit_dt=date_idx,
                exit_px=row["price"],
                pl_pct=cash / (entry_px * pos) - 1
            )
            pos, entry_px = 0, None

        equity.append(cash + pos * row["price"])

    # -------- wrap‑up --------
    eq_df = df.iloc[1:].copy()
    eq_df["equity"] = equity
    eq_df["bh_equity"] = initial_cap * (eq_df["price"] / eq_df["price"].iloc[0])

    rng = np.random.default_rng(42)
    rand = rng.integers(0, 2, len(eq_df))
    eq_df["random_equity"] = initial_cap * ((1 + eq_df["ret_pct"] * rand).cumprod())

    # guarantee columns even if no trades
    trades_df = pd.DataFrame(
        trades, columns=["entry_dt", "exit_dt", "entry_px", "exit_px", "pl_pct"]
    )

    return eq_df, trades_df
def run_model_strategy_bayes(
    df: pd.DataFrame,
    model,
    prob_thresh: float = 0.70,
    initial_cap: float = 1_000,
    tx_cost_bps=0.0
):
    """
    Logistic‑regression (Bayes tab) strategy.
    Uses calibrated P(up) from model.predict_proba().
    """
    # 1‑step‑ahead engineered features
    feats = models._feature_engineering(df)
    # column  = probability that next‑day return > 0
    probs = model.predict_proba(feats)[:, 1]
    entry_signal = pd.Series(probs, index=df.index) > prob_thresh
    return _exec_signal(df, entry_signal, initial_cap, tx_cost_bps)
def run_model_strategy_arima(
    df: pd.DataFrame,
    model,
    ret_thresh: float = 0.02,
    initial_cap: float = 1_000,
    tx_cost_bps=0.0
):
    """
    ARIMA(1,0,0)+sentiment strategy (no look‑ahead).

    * Uses in‑sample one‑step forecasts that share df.index
    * Works on statsmodels 0.12 -> 0.15+
    """
    # --------------------------------------------------------------
    # Older statsmodels (<0.14) needs `typ="levels"` or the result
    # is in log‑terms. Passing it on >=0.14 is harmless
    # --------------------------------------------------------------
    fc = model.predict(
        start=0,
        end=len(df) - 1,
        exog=df["sentiment"],
        typ="levels",          # <‑‑ compatibility flag
    )

    fc = pd.Series(fc.values, index=df.index)

    # trade on *yesterday’s* forecast
    entry_signal = fc.shift(1) > ret_thresh
    entry_signal.iloc[0] = False      # first day cannot trade

    return _exec_signal(df, entry_signal, initial_cap, tx_cost_bps)

def run_model_strategy_prophet(
    df: pd.DataFrame,
    ret_thresh: float = 0.02,
    initial_cap: float = 1_000,
    tx_cost_bps: float = 0.0,
    warmup: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expanding‑window walk‑forward Prophet strategy.

    Re‑fits Prophet each day on data ≤ t‑1 and trades on the forecast
    for day t‑1 (shifted so there’s no look‑ahead).
    """
    if len(df) <= warmup:
        raise ValueError("Not enough data for walk‑forward Prophet.")

    fc_vals = [np.nan] * len(df)
    for i in range(warmup, len(df)):
        train = df.iloc[:i]
        mdl   = models.train_prophet(train)
        fc_vals[i] = models.forecast_prophet(
            mdl,
            next_date=df.index[i],
            next_sent=float(df["sentiment"].iloc[i]),
        )

    # build yesterday’s signal
    entry_signal = (
        pd.Series(fc_vals, index=df.index)
          .shift(1)
          .gt(ret_thresh)
    )
    entry_signal.iloc[: warmup + 1] = False

    return _exec_signal(df, entry_signal, initial_cap, tx_cost_bps)


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
    (Pure‑Python implementation with the `ta` library.)
    """
    out   = df.copy()
    close = out["price"]

    # ── basic rolling stats ───────────────────────────────────────
    out["vol_10"] = close.pct_change().rolling(10).std()
    out["mom_5"]  = close.pct_change(5)

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["z_20"]   = (close - sma20) / std20

    # ── technical indicators (ta) ─────────────────────────────────
    out["rsi_14"]    = ta.momentum.rsi(close, window=14)

    # MACD histogram (fast 12, slow 26, signal 9)
    out["macd_hist"] = ta.trend.macd_diff(
        close,
        window_slow=26, window_fast=12, window_sign=9
    )

    # Bollinger Bands 20‑period ±2 σ
    bb_high = ta.volatility.bollinger_hband(close, window=20, window_dev=2)
    bb_low  = ta.volatility.bollinger_lband(close, window=20, window_dev=2)
    width   = bb_high - bb_low

    out["bb_low"]   = bb_low
    out["bb_high"]  = bb_high
    out["bb_perc"]  = (close - bb_low) / width
    out["bb_width"] = width / sma20

    # ── housekeeping ─────────────────────────────────────────────
    out["bb_perc"] = out["bb_perc"].clip(-0.1, 1.0)   # flat‑market safeguard
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.bfill(inplace=True)                           # fill initial NaNs

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


def rolling_corr_pair(a: pd.Series, b: pd.Series, win: int = 30) -> pd.Series:
    """
    Compute rolling window correlation of the daily returns of two price series.
    Output index matches input index after .pct_change().dropna() alignment.
    Returns empty Series if not enough data.
    """
    import pandas as pd

    # Flatten input to 1D Series if passed as 1-column DataFrame
    if isinstance(a, pd.DataFrame) and a.shape[1] == 1:
        a = a.iloc[:, 0]
    if isinstance(b, pd.DataFrame) and b.shape[1] == 1:
        b = b.iloc[:, 0]

    # Now always Series
    a = pd.Series(a)
    b = pd.Series(b)
    # Align series on shared index
    a, b = a.align(b, join='inner')

    if len(a) < 2 or len(b) < 2 or a.isnull().all() or b.isnull().all():
        return pd.Series(dtype=float, index=a.index)

    a_ret = a.pct_change()
    b_ret = b.pct_change()
    if a_ret.isnull().all() or b_ret.isnull().all():
        return pd.Series(dtype=float, index=a.index)

    returns = pd.DataFrame({"a": a_ret, "b": b_ret}).dropna()
    if returns.empty or len(returns) < win:
        return pd.Series(dtype=float, index=returns.index)

    rho = returns["a"].rolling(win, min_periods=win).corr(returns["b"])
    return rho

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
    # ── new: give the Series a proper label for Plotly ───────────────────
    rho = rho.copy()                       # make it writable
    rho.name = f"{symA}–{symB}"            # legend / hover label
    # --------------------------------------------------------------------

    fig = px.line(
        rho,
        title=f"30‑day rolling ρ: {symA} vs {symB}",
        labels={"value": "Correlation", "index": "Date"},
    )
    fig.add_hline(0, line_dash="dash", line_color="grey")
    st.plotly_chart(fig, use_container_width=True)
# ------------------------------------------------------------------
# 7. Streamlit display
# ------------------------------------------------------------------
def show_results(equity_df: pd.DataFrame, trades_df: pd.DataFrame):
    """Render strategy performance metrics, equity curves, and interactive trade log."""
    st.subheader("📈 Strategy Performance")

    # --- headline metrics ---------------------------------------------------
    tot_ret = equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0] - 1
    cagr    = (1 + tot_ret) ** (252 / len(equity_df)) - 1
    mdd     = _max_drawdown(equity_df["equity"])
    sharpe  = _sharpe(equity_df["equity"].pct_change().dropna())

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Total return",   f"{tot_ret*100:.1f}%")
    colB.metric("CAGR",           f"{cagr*100:.1f}%")
    colC.metric("Max draw-down",  f"{mdd*100:.1f}%")
    colD.metric("Sharpe",         f"{sharpe:.2f}")

    # --- equity curves ------------------------------------------------------
    fig = px.line(
        equity_df[["equity", "bh_equity", "random_equity"]],
        title="Cumulative equity vs benchmarks",
        labels={"value": "Equity ($)", "index": "Date", "variable": "Series"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- trade log ----------------------------------------------------------
    st.subheader("📜 Trade log")

    # Make a working copy; ensure numeric P/L and drop still-open trades
    trades_df = trades_df.copy()
    trades_df["pl_pct"] = pd.to_numeric(trades_df["pl_pct"], errors="coerce")
    trades_df_disp = trades_df.dropna(subset=["pl_pct"]).copy()

    if trades_df_disp.empty:
        st.info("No trades executed for the selected rules.")
        return

    # Display P/L in %
    trades_df_disp["pl_pct"] = (trades_df_disp["pl_pct"] * 100).round(2)

    # Interactive Ag-Grid table
    gb = GridOptionsBuilder.from_dataframe(trades_df_disp)
    gb.configure_default_column(filter=True, sortable=True, resizable=True)
    gb.configure_column("pl_pct", header_name="P/L %", type=["numericColumn"], width=90)
    gb.configure_side_bar()  # optional: columns & filters side panel
    grid_opts = gb.build()

    AgGrid(
        trades_df_disp,
        gridOptions=grid_opts,
        update_mode=GridUpdateMode.NO_UPDATE,
        enable_enterprise_modules=False,
        height=220,
        fit_columns_on_grid_load=True,
        theme="streamlit",
    )

    # --- trade summary stats ------------------------------------------------
    wins = trades_df_disp["pl_pct"] > 0
    win_rate = wins.mean()
    avg_win  = trades_df_disp.loc[wins,  "pl_pct"].mean()
    avg_loss = trades_df_disp.loc[~wins, "pl_pct"].mean()

    st.write(
        f"**Win rate:** {win_rate*100:.1f}% &nbsp;&nbsp; | "
        f"**Avg win:** {avg_win:.2f}% &nbsp;&nbsp; | "
        f"**Avg loss:** {avg_loss:.2f}%"
    )
