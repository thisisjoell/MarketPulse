# pages/2_Correlations.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import yfinance as yf

import shared
import alpha_lab


# ──────────────────────────────────────────────
#  Page setup & Pro‑mode gate
# ──────────────────────────────────────────────
st.set_page_config(page_title="Correlation Analysis", layout="wide")

pro_mode = shared.sidebar_mode_toggle()
shared.inject_global_css()

if not pro_mode:
    st.warning("Enable **Pro mode** in the sidebar to access Correlations.")
    st.stop()

_DARK_UI      = st.get_option("theme.base") != "light"
HEATMAP_SCALE = "RdBu" if _DARK_UI else "RdBu_r"

st.title("Cross‑Asset Correlation Analysis")

# ══════════════════════════════════════════════
#  CONTROL PANEL
# ══════════════════════════════════════════════
with st.expander("Controls", expanded=True):
    c1, c2 = st.columns(2)

    # ── left column ────────────────────────────────────────────────
    with c1:
        tick_sel = st.text_input(
            "Tickers (comma separated)",
            st.session_state.get("corr_tickers", "AAPL, TSLA, NVDA, BTC-USD"),
            key="corr_tickers",
            help="Enter two or more Yahoo Finance symbols.",
        )

        mode = st.selectbox(
            "Correlation of …",
            ("Price returns", "Manual‑strategy equity"),
            index=0 if st.session_state.get("corr_mode") == "Price returns" else 1,
        )
        st.session_state["corr_mode"] = mode

    # ── right column ───────────────────────────────────────────────
    with c2:
        win_days = st.slider(
            "Rolling window (days)",
            10, 120,
            st.session_state.get("corr_win", 30),
            5,
            key="corr_win",
            help="Window length for rolling correlations.",
        )

        # ░▒▓ safe date‑range picker ▓▒░
        default_range = st.session_state.get(
            "corr_date_range",
            (
                pd.to_datetime("today").date() - pd.Timedelta(days=365),
                pd.to_datetime("today").date(),
            ),
        )
        date_sel = st.date_input(
            "Date range",
            value=default_range,
            max_value=pd.to_datetime("today").date(),
            help="Historical window fetched for each ticker.",
        )

        # guard – user may click only one side of the range
        if isinstance(date_sel, tuple) and len(date_sel) == 2:
            start_date, end_date = date_sel
            st.session_state["corr_date_range"] = (start_date, end_date)
        else:
            start_date, end_date = st.session_state.get("corr_date_range", default_range)

    # ── strategy filters (equity‑mode only) ───────────────────────
    strat_kwargs: dict = {}
    if mode == "Manual‑strategy equity":
        st.markdown("#### Equity strategy filters")
        l, r = st.columns(2)
        with l:
            strat_kwargs["sent_threshold"]  = st.slider("Sentiment >",        0.00, 1.00, 0.05, 0.01)
            strat_kwargs["price_drop_pct"]  = st.slider("Price drop % ≥",     0.0, 10.0, 2.0, 0.10) / 100
            strat_kwargs["price_rise_pct"]  = st.slider("Take‑profit % ≥",    0.0, 10.0, 5.0, 0.10) / 100
            strat_kwargs["vol_thresh"]      = st.slider("Volatility spike >", 0.0,  5.0, 1.5, 0.10) / 100
        with r:
            strat_kwargs["mom_thresh"]      = st.slider("Bear momentum % <",  0.0,  5.0, 1.0, 0.10) / 100
            strat_kwargs["rsi_max"]         = st.slider("RSI ≤", 10, 70, 30, 1)
            strat_kwargs["macd_req"]        = st.checkbox("Require MACD histogram > 0", value=False)
            strat_kwargs["tx_cost_bps"]     = st.slider("Tx‑cost (bps)", 0, 50, 5, 1)
            strat_kwargs["stop_loss_pct"]   = st.slider("Stop‑loss %", 0.0, 20.0, 0.0, 0.50) / 100


# ══════════════════════════════════════════════
#  Hot‑key ⌘/Ctrl + Enter & Run button
# ══════════════════════════════════════════════
st.markdown(
    """
    <script>
    document.addEventListener('keydown', e=>{
        if((e.metaKey||e.ctrlKey)&&e.key==='Enter'){
            const btn=window.parent.document.querySelector('button[kind="primary"]');
            if(btn){btn.click();}
        }
    });
    </script>
    """,
    unsafe_allow_html=True,
)
run_btn = st.button("Compute correlations", type="primary")

# ================================================================
#  1— Build a signature of the current controls
# ================================================================
def _signature() -> str:
    parts = [tick_sel.lower(), mode, str(win_days)]
    if strat_kwargs:
        parts += [f"{k}:{v}" for k, v in strat_kwargs.items()]
    parts += [start_date.isoformat(), end_date.isoformat()]
    return "|".join(parts)

sig_now        = _signature()
cache          = st.session_state.get("corr_cache")
has_valid_cache= cache is not None and cache["signature"] == sig_now

# -----------------------------------------------------------------
#  USER‑ACTION LOGIC
# -----------------------------------------------------------------
if run_btn:
    needs_refresh = True
elif has_valid_cache:
    needs_refresh = False
else:
    st.info("Adjust the settings, then press **Compute correlations**.")
    st.stop()

# ══════════════════════════════════════════════
#  2— computation (only when needed)
# ══════════════════════════════════════════════
if needs_refresh:
    # 1) validate tickers -------------------------------------------------------
    symbols = [s.strip().upper() for s in tick_sel.split(",") if s.strip()]
    if len(symbols) < 2:
        st.warning("Enter at least two tickers."); st.stop()

    valid_syms, bad_syms = [], []
    for sym in symbols:
        try:
            if not yf.Ticker(sym).fast_info:
                bad_syms.append(sym)
            else:
                valid_syms.append(sym)
        except Exception:
            bad_syms.append(sym)

    if bad_syms:
        st.warning(f"Unknown symbols skipped: {', '.join(bad_syms)}")
    if len(valid_syms) < 2:
        st.warning("Not enough valid tickers to continue."); st.stop()

    # 2) fetch data -------------------------------------------------------------
    price_map, equity_map, notes = {}, {}, []
    with st.status("Downloading data…", expanded=True) as status:
        for i, sym in enumerate(valid_syms, start=1):
            status.update(label=f"Downloading {sym} ({i}/{len(valid_syms)})")
            try:
                px_daily, trend_df, _ = shared.fetch_price_and_trend(sym, start_date, end_date)
            except Exception as err:
                notes.append(f"{sym}: {err}")
                status.write(f"⚠️ {sym} – {err}")
                continue

            if px_daily.empty:
                notes.append(f"{sym}: no price data returned – skipped.")
                status.write(f"ℹ️ {sym}: no price data returned – skipped.")
                continue

            price_map[sym] = px_daily.asfreq("D").ffill()

            if mode == "Manual‑strategy equity":
                merged      = alpha_lab.merge_price_sentiment(px_daily, trend_df)
                eq_curve, _ = alpha_lab.run_manual_strategy(merged, **strat_kwargs)
                if eq_curve["equity"].nunique() > 1:
                    equity_map[sym] = eq_curve["equity"]
                else:
                    notes.append(f"{sym}: no trades met – skipped.")
                    status.write(f"ℹ️ {sym}: no trades met – skipped.")
        status.update(state="complete", expanded=False)

    for m in notes:
        st.info(m)

    data_dict = price_map if mode == "Price returns" else equity_map
    if len(data_dict) < 2:
        st.warning("No valid series available to build a correlation matrix."); st.stop()

    # 3) build correlation matrix ----------------------------------------------
    try:
        corr_df = alpha_lab.corr_matrix(data_dict)
    except ValueError:
        st.warning("Correlation matrix could not be generated."); st.stop()

    if corr_df.empty:
        st.warning("Correlation matrix is empty."); st.stop()

    # 4) cache results ----------------------------------------------------------
    cache = dict(
        signature = sig_now,
        corr_df   = corr_df,
        data_dict = data_dict,
        valid_syms= valid_syms,
        win_days  = win_days,
    )
    st.session_state["corr_cache"] = cache

# -------------------------------------------------------------------
#  DISPLAY – always uses the cached object
# -------------------------------------------------------------------
corr_df    = cache["corr_df"]
data_dict  = cache["data_dict"]
valid_syms = cache["valid_syms"]
win_days   = cache["win_days"]

# ══════════════════════════════════════════════
#  KPI strip
# ══════════════════════════════════════════════
mask      = ~np.eye(len(corr_df), dtype=bool)
abs_vals  = corr_df.where(mask).abs().stack().dropna()
avg_abs   = abs_vals.mean()
pair      = abs_vals.idxmax()
pair_val  = corr_df.loc[pair]

cA, cB = st.columns(2, gap="small")
cA.metric("Average |ρ|", f"{avg_abs:.2f}")
cB.metric(f"Highest |ρ| ({pair[0]}–{pair[1]})", f"{pair_val:+.2f}")

# ══════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════
tab_matrix, tab_roll = st.tabs(["Correlation Matrix", "Rolling Correlation"])

with tab_matrix:
    thr = st.slider("|ρ| highlight threshold", 0.0, 1.0, 0.5, 0.05)

    fig = px.imshow(
        corr_df.round(2),
        text_auto=True,
        color_continuous_scale=HEATMAP_SCALE,
        zmin=-1, zmax=1,
    )
    fig.update_traces(hovertemplate="%{y} ↔ %{x}<br>ρ = %{z:+.2f}<extra></extra>")
    fig.update_layout(
        transition=dict(duration=400, easing="cubic-in-out"),
        font=dict(size=14),
    )
    st.plotly_chart(shared.style_plotly(fig, dark=_DARK_UI), use_container_width=True)

    hl = corr_df.abs() >= thr
    styled = corr_df.style.apply(
        lambda s: [
            "background-color:#123d2b" if hl.loc[s.name, c] else "" for c in s.index
        ],
        axis=1,
    )
    shared.mp_table(styled, key="corr_tbl")

    lcol, rcol = st.columns(2)
    lcol.download_button(
        "Download CSV",
        corr_df.to_csv().encode(),
        "correlation_matrix.csv",
        "text/csv",
    )
    rcol.download_button(
        "Download PNG",
        fig.to_image(format="png"),
        "correlation_matrix.png",
        mime="image/png",
    )

with tab_roll:
    pairs = [
        f"{a} ↔ {b}"
        for i, a in enumerate(valid_syms)
        for b in valid_syms[i + 1 :]
        if a in data_dict and b in data_dict
    ]
    if not pairs:
        st.info("No valid pairs to plot."); st.stop()

    sel_pair = st.selectbox("Choose pair", pairs)
    a_sym, b_sym = [s.strip() for s in sel_pair.split("↔")]

    rho = alpha_lab.rolling_corr_pair(data_dict[a_sym], data_dict[b_sym], win=win_days)
    if rho.empty:
        st.info("Series too short for the selected window.")
    else:
        alpha_lab.plot_rolling_corr(rho, a_sym, b_sym)

shared.mark_data_fetched()
st.toast("Correlation analysis complete.", icon="🔗")












