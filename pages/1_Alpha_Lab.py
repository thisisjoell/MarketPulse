#
# pages/1_Alpha_Lab.py
import streamlit as st
import pandas as pd
import numpy as np
import shared, alpha_lab, models
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# ------------------------------------------------------------------
# Page config, header, Pro gating
# ------------------------------------------------------------------
st.set_page_config(page_title="Alpha Lab", page_icon="", layout="wide")
pro_mode = shared.sidebar_mode_toggle()
shared.global_header()
if not pro_mode:
    st.warning("Enable **Pro mode** in the sidebar to access Alpha Lab.")
    st.stop()

# ------------------------------------------------------------------
# Global context (ticker + date‑range from the header widgets)
# ------------------------------------------------------------------
ticker               = st.session_state["ticker"]
start_date, end_date = st.session_state["date_range"]

# ------------------------------------------------------------------
# Ensure data in session…
# ------------------------------------------------------------------
key = f"{ticker}_{start_date.isoformat()}_{end_date.isoformat()}"
price_daily = st.session_state.get(f"price_daily_{key}")
trend       = st.session_state.get(f"trend_{key}")

if price_daily is None or trend is None:
    with st.spinner(f"Fetching price & sentiment for {ticker}…"):
        price_daily, trend, _ = shared.fetch_price_and_trend(ticker, start_date, end_date)
        st.session_state[f"price_daily_{key}"] = price_daily
        st.session_state[f"trend_{key}"]       = trend
        st.session_state["price_daily"]        = price_daily
        st.session_state["trend"]              = trend
    shared.mark_data_fetched()
    st.toast("Data refreshed for Alpha Lab.", icon="🟢")
else:
    st.session_state["price_daily"] = price_daily
    st.session_state["trend"]       = trend

st.title(f"Alpha Lab — {ticker}")

# ==================================================================
# TABS: Manual Rules | Model Signal
# ==================================================================
tab_manual, tab_model = st.tabs(["Manual rules", "Model signal"])

# ==================================================================
#  Manual rules strategy
# ==================================================================
with tab_manual:
    # ----------------- parameter inputs --------------------------
    s_thr = st.slider(
        "Sentiment >", 0.00, 1.00, 0.05, 0.01,
        help="Prev‑day aggregated Reddit sentiment must exceed this."
    )

    p_drop_ui = st.slider(
        "Price drop % ≥", 0.0, 10.0, 2.0, 0.10,
        help="Require today's drop (close/prev close) to be at least this %."
    )
    p_drop = p_drop_ui / 100

    p_rise_ui = st.slider(
        "Take‑profit % ≥", 0.0, 10.0, 5.0, 0.10,
        help="Exit when gain meets/exceeds this %."
    )
    p_rise = p_rise_ui / 100

    vol_th_ui = st.slider(
        "Volatility spike >", 0.0, 5.0, 1.5, 0.10,
        help="10‑day rolling σ(returns) must exceed this %."
    )
    vol_th = vol_th_ui / 100

    mom_th_ui = st.slider(
        "Bear momentum % <", 0.0, 5.0, 1.0, 0.10,
        help="5‑day % change must be worse (more negative) than −this %."
    )
    mom_th = mom_th_ui / 100

    tx_bps = st.slider(
        "Tx‑cost (bps), per side", 0, 50, 5, 1,
        help="One‑way commission/slippage in basis points (1bp = 0.01%)."
    )
    initial_cap = st.number_input(
        "Initial capital ($)", min_value=100, value=1_000, step=100,
        help="Starting capital for your back‑tests."
    )
    st.session_state["tx_bps"]      = tx_bps
    st.session_state["initial_cap"] = initial_cap

    stop_ls_ui = st.slider(
        "Stop‑loss %", 0.0, 20.0, 0.0, 0.50,
        help="If >0, exit when price falls this % below entry."
    )
    stop_ls = stop_ls_ui / 100

    st.markdown("**Technical‑Indicator Guards**")
    rsi_max = st.slider(
        "RSI ≤", 10, 70, 30, 1,
        help="Only enter if 14‑period RSI is ≤ this level (oversold guard)."
    )
    use_macd = st.checkbox(
        "Require MACD histogram > 0", value=False,
        help="Only enter if MACD(12,26,9) histogram is positive (bullish)."
    )

    if st.button("Run manual back‑test"):
        price_daily = st.session_state.get("price_daily")
        trend       = st.session_state.get("trend")
        if price_daily is None:
            st.warning("Run **Home → Fetch Data & Sentiment** first.")
            st.stop()

        merged = alpha_lab.merge_price_sentiment(price_daily, trend)
        eq, trades = alpha_lab.run_manual_strategy(
            merged,
            sent_threshold   = s_thr,
            price_drop_pct   = p_drop,
            price_rise_pct   = p_rise,
            vol_thresh       = vol_th,
            mom_thresh       = mom_th,
            rsi_max          = rsi_max,
            macd_req         = use_macd,
            stop_loss_pct    = stop_ls,
            tx_cost_bps      = tx_bps,
            initial_cap      = initial_cap,
        )
        alpha_lab.show_results(eq, trades)
        st.toast("Manual back‑test complete.", icon="🧮")

    st.markdown("---")
    st.subheader("🔧 Hyper‑parameter grid search")
    st.caption("Exhaustively sweeps key rule thresholds to maximise Sharpe on the current data window.")

    if st.button("Run grid on current ticker"):
        price_daily = st.session_state.get("price_daily")
        trend       = st.session_state.get("trend")
        if price_daily is None:
            st.info("Go to **Home** and Fetch data first.")
            st.stop()

        merged = alpha_lab.merge_price_sentiment(price_daily, trend)
        grid = {
            "sent_threshold": [0.02, 0.05, 0.10],
            "price_drop_pct": [0.01, 0.02],
            "price_rise_pct": [0.03, 0.05],   # mandatory
            "vol_thresh":     [0.01, 0.015],
        }
        res = alpha_lab.grid_search(
            merged,
            grid,
            backtest_callable=lambda d, **k: alpha_lab.run_manual_strategy(d, **k),
            base_kwargs=dict(
                mom_thresh    = mom_th,
                stop_loss_pct = stop_ls,
                tx_cost_bps   = tx_bps,
                initial_cap   = initial_cap,
            ),
        )
        shared.mp_table(res, key="grid_res")
        st.toast("Grid search complete.", icon="🧮")

    st.markdown("---")
    st.subheader("📊 Multi‑ticker benchmark")
    st.session_state["bench_params"] = dict(
        sent_threshold   = s_thr,
        price_drop_pct   = p_drop,
        price_rise_pct   = p_rise,
        vol_thresh       = vol_th,
        mom_thresh       = mom_th,
        rsi_max          = rsi_max,
        macd_req         = use_macd,
        stop_loss_pct    = stop_ls,
        tx_cost_bps      = tx_bps,
        initial_cap      = initial_cap,
    )
    st.info(
        "To compare these rules across many tickers, head to the **Benchmarks** page. "
        "Your current parameters are pre‑loaded there."
    )
    try:
        st.page_link("pages/3_Benchmarks.py", label="➡ Go to Benchmarks", icon="📊")
    except Exception:
        st.markdown("**[Go to Benchmarks ▶](./3_Benchmarks)**")

# ==================================================================
#  Model signal strategies
# ==================================================================
with tab_model:
    model_type = st.selectbox(
        "Model",
        ["Logistic", "ARIMA", "Prophet"],
        help=("Choose the engine that generates entry signals:\n"
              "• Logistic = fast linear classifier on engineered features.\n"
              "• ARIMA = time‑series model with sentiment as exogenous regressor.\n"
              "• Prophet = additive model with weekly/yearly seasonality + sentiment.")  # NEW help
    )

    thresh = st.slider(
        "Probability / forecast threshold",
        0.50, 0.90, 0.70, 0.01,
        help=("Minimum predicted probability (Logistic/LGBM) **or** expected 1‑day "
              "return (ARIMA/Prophet) required to open a long position. "
              "Higher = fewer trades, higher precision.")  # NEW help
    )

    use_cv = st.checkbox(
        "Walk‑forward CV (5 folds)", value=False,
        help=("Splits data into 5 expanding windows, retraining each time. "
              "Mimics live model updates and prevents look‑ahead bias.")  # NEW help
    )

    tx_bps      = st.session_state.get("tx_bps", 0)
    initial_cap = st.session_state.get("initial_cap", 1_000)

    if model_type == "Prophet":
        warmup = st.slider(
            "Prophet warm‑up (min history for first forecast)",
            min_value=30, max_value=500, value=120, step=10,
            help="Days of data to train before your first 1‑day forecast."
        )
    else:
        warmup = None

    if st.button("Train model & back‑test"):
        merged = alpha_lab.merge_price_sentiment(price_daily, trend)

        if model_type == "Logistic":
            train_fn = models.train_bayes
            strat_fn = lambda df_slice, mdl: alpha_lab.run_model_strategy_bayes(
                df_slice, mdl,
                prob_thresh=thresh,
                initial_cap=initial_cap,
                tx_cost_bps=tx_bps,
            )
        elif model_type == "ARIMA":
            train_fn = models.train_arima
            strat_fn = lambda df_slice, mdl: alpha_lab.run_model_strategy_arima(
                df_slice, mdl,
                ret_thresh=(thresh - 0.5) / 10,
                initial_cap=initial_cap,
                tx_cost_bps=tx_bps,
            )
        else:  # Prophet
            train_fn = lambda df_slice, **kw: None
            strat_fn = lambda df_slice, _: alpha_lab.run_model_strategy_prophet(
                df_slice,
                ret_thresh=(thresh - 0.5) / 10,
                initial_cap=initial_cap,
                tx_cost_bps=tx_bps,
                warmup=warmup,
            )

        try:
            if use_cv:
                with st.spinner("Running walk‑forward CV…"):
                    eq, trades = alpha_lab.walk_forward_backtest(merged, train_fn, strat_fn)
            else:
                mdl      = train_fn(merged)
                eq, trades = strat_fn(merged, mdl)
        except ValueError as e:
            st.warning(str(e))
        else:
            alpha_lab.show_results(eq, trades)
            st.toast("Model back‑test complete.", icon="🤖")

