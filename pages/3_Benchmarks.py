# pages/3_Benchmarks.py
"""
📊 Benchmarks — multi‑ticker back‑tests (UIrev‑10,Jul2025)
────────────────────────────────────────────────────────────
✓ Local date‑range picker (main pane)   ✓ Initial‑capital input
✓ Signature cache                       ✓ ⌘/Ctrl+ Enter
✓ Busy‑button lock                      ✓ Step‑status + ETA
✓ KPI cards (4)                         ✓ Sparkline grid
✓ Theme‑aware bars                      ✓ CSV / PNG export
✓ Validation expander
"""

from __future__ import annotations
import time
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import plotly.express as px
import yfinance as yf

import shared, alpha_lab, viz   # viz.sparkline_row / render_sparkline_grid

# ──────────────────────────────────────────────────────────
#  Basic chrome (skip shared.global_header here)
# ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Benchmarks", page_icon="📊", layout="wide")
shared.inject_global_css()               # fonts + KPI‑card skin

pro_mode = shared.sidebar_mode_toggle()
if not pro_mode:
    st.warning("Enable **Pro mode** in the sidebar to access Benchmarks.")
    st.stop()

# Hot‑key ⌘ /Ctrl + Enter ─────────────────────────────────
st.markdown(
    """
    <script>
    document.addEventListener('keydown', e=>{
      if((e.metaKey||e.ctrlKey)&&e.key==='Enter'){
         const btn=window.parent.document.querySelector('button[kind="primary"]');
         if(btn){btn.click();}
    }});
    </script>""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────
#  Saved strategy defaults (pulled from Alpha Lab if present)
# ──────────────────────────────────────────────────────────
_saved   = st.session_state.get("bench_params", {})
last_df  = st.session_state.get("_bench_last_df")

def _num(key: str, fallback: float | int) -> float:
    val = _saved.get(key, fallback)
    while isinstance(val, (list, tuple)):
        val = val[0] if val else fallback
    try:
        return float(val)
    except Exception:
        return float(fallback)

# ══════════════════════════════════════════════════════════
#  1. Strategy controls
# ══════════════════════════════════════════════════════════
st.title("Multi‑Ticker Strategy Benchmarks")
st.caption("Back‑tests the **manual‑rule strategy** (same logic as Alpha Lab) on every ticker you provide.")

st.markdown("### Strategy parameters")

s_thr      = st.slider("Sentiment >",         0.00, 1.00,  _num("sent_threshold", .05),  .01)
p_drop_ui  = st.slider("Price drop % ≥",      0.0, 10.0,   _num("price_drop_pct", .02)*100, .10)
p_rise_ui  = st.slider("Take‑profit % ≥",     0.0, 10.0,   _num("price_rise_pct", .05)*100, .10)
vol_th_ui  = st.slider("Volatility spike >",  0.0,  5.0,   _num("vol_thresh", .015)*100,  .10)
mom_th_ui  = st.slider("Bear momentum % <",   0.0,  5.0,   _num("mom_thresh",  .01)*100,  .10)
tx_bps     = st.slider("Tx‑cost (bps)",         0,    50,  int(_num("tx_cost_bps", 5)), 1)
stop_ls_ui = st.slider("Stop‑loss %",          0.0, 20.0,  _num("stop_loss_pct", 0)*100,  .50)

# NEW — initial capital (mirrors Alpha Lab)
initial_cap = st.number_input(
    "Initial capital ($)", min_value=100, value=int(_num("initial_cap", 1_000)), step=100
)

p_drop, p_rise = p_drop_ui/100, p_rise_ui/100
vol_th , mom_th= vol_th_ui/100, mom_th_ui/100
stop_ls        = stop_ls_ui/100

st.markdown("**Advanced guards**")
rsi_max  = st.slider("RSI ≤", 10, 70, int(_num("rsi_max", 30)), 1)
use_macd = st.checkbox("Require MACD histogram > 0", bool(_num("macd_req", False)))

# ══════════════════════════════════════════════════════════
#  2. Local benchmark date‑range (in‑page)
# ══════════════════════════════════════════════════════════
today          = date.today()
default_range  = st.session_state.get("bench_date_range",
                                      (today - timedelta(days=180), today))


bench_range = st.date_input("Select range", value=default_range, max_value=today)
if isinstance(bench_range, tuple) and len(bench_range) == 2:
    start_date, end_date = bench_range
else:
    start_date, end_date = default_range
st.session_state["bench_date_range"] = (start_date, end_date)

# ══════════════════════════════════════════════════════════
#  3. Tickers input
# ══════════════════════════════════════════════════════════
tickers_str = st.text_input("Tickers (comma separated)",
                            value="AAPL, TSLA, GOOG",
                            help="Enter one or more Yahoo Finance symbols.")

# ══════════════════════════════════════════════════════════
#  4. Signature‑based cache
# ══════════════════════════════════════════════════════════
def _signature() -> str:
    parts = [tickers_str.lower(), str(start_date), str(end_date),
             f"{s_thr:.3f}", p_drop_ui, p_rise_ui, vol_th_ui, mom_th_ui,
             tx_bps, stop_ls_ui, rsi_max, use_macd, initial_cap]
    return "|".join(map(str, parts))

sig_now   = _signature()
_cache    = st.session_state.get("_bench_cache")
has_cache = _cache and _cache["sig"] == sig_now

# 5. Run button (busy‑lock) ───────────────────────────────
busy    = st.session_state.get("_bench_busy", False)
run_btn = st.button("Run benchmark", type="primary", disabled=busy)
if run_btn:
    st.session_state["_bench_busy"] = True

# If no new run but cache valid → load immediately
if not run_btn and has_cache:
    df_sum      = _cache["df"]
    spark_rows  = _cache["spark_rows"]
    skipped     = _cache["skipped"]
    errors      = _cache["errors"]
    show_cached = True
else:
    show_cached = False

# ══════════════════════════════════════════════════════════
#  5. Compute benchmarks
# ══════════════════════════════════════════════════════════
if run_btn:
    try:
        tick_list = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
        if not tick_list:
            st.warning("Enter at least one ticker."); st.stop()

        invalid = [t for t in tick_list if not yf.Ticker(t).fast_info]
        if invalid:
            st.warning(f"Unknown symbols skipped: {', '.join(invalid)}")
            tick_list = [t for t in tick_list if t not in invalid]
        if not tick_list:
            st.stop()

        rows, spark_rows, skipped, errors = [], [], [], []
        t0 = time.time()

        with st.status("Running benchmarks…", expanded=True) as status:
            for i, sym in enumerate(tick_list, start=1):
                loop_start = time.time()
                status.update(label=f"{i}/{len(tick_list)} • {sym}")

                try:
                    price, trend, _ = shared.fetch_price_and_trend(sym, start_date, end_date)
                except Exception as err:
                    errors.append(f"{sym}: {err}")
                    status.write(f"❌ {sym}: {err}")
                    continue

                price_daily = price.asfreq("D").ffill()
                merged      = alpha_lab.merge_price_sentiment(price_daily, trend)
                equity, _   = alpha_lab.run_manual_strategy(
                    merged,
                    sent_threshold = s_thr,
                    price_drop_pct = p_drop,
                    price_rise_pct = p_rise,
                    vol_thresh     = vol_th,
                    mom_thresh     = mom_th,
                    rsi_max        = rsi_max,
                    macd_req       = use_macd,
                    stop_loss_pct  = stop_ls,
                    tx_cost_bps    = tx_bps,
                    initial_cap    = initial_cap,      # ← NEW
                )

                if equity["equity"].nunique() == 1:
                    skipped.append(sym)
                    status.write(f"⚪ {sym}: no trades met criteria.")
                    continue

                daily = equity["equity"].pct_change().dropna()
                rows.append(dict(
                    Ticker=sym,
                    Sharpe=round(alpha_lab._sharpe(daily), 3),
                    MaxDD_pct=round(alpha_lab._max_drawdown(equity["equity"]) * 100, 1),
                    TotalRet_pct=round((equity["equity"].iloc[-1] /
                                        equity["equity"].iloc[0] - 1) * 100, 1),
                    EndEquity_usd=round(equity["equity"].iloc[-1], 2),  # ← NEW
                ))

                spark_rows.append(viz.sparkline_row(sym, price_daily[-90:], trend))

                dt  = time.time() - loop_start
                eta = (time.time()-t0)/i * (len(tick_list)-i)
                status.update(label=f"{i}/{len(tick_list)} • ~{eta:,.1f}s left")
                status.write(f"✅ {sym} done in {dt:.1f}s")

            status.update(state="complete", expanded=False)

        df_sum = pd.DataFrame(rows).rename(columns={
            "MaxDD_pct": "MaxDD %",
            "TotalRet_pct": "TotalRet %",
            "EndEquity_usd": "End Equity $"  # ← NEW
        })
        if not df_sum.empty and "Sharpe" in df_sum.columns:
            df_sum = df_sum.sort_values("Sharpe", ascending=False)

        st.session_state["_bench_cache"] = dict(
            sig=sig_now, df=df_sum,
            spark_rows=spark_rows, skipped=skipped, errors=errors
        )
        st.session_state["_bench_last_df"] = df_sum
    finally:
        st.session_state["_bench_busy"] = False

# ══════════════════════════════════════════════════════════
#  6. Display section
# ══════════════════════════════════════════════════════════
if show_cached:
    st.info("Showing last result (controls unchanged).")

if 'df_sum' in locals() and not df_sum.empty:
    # KPI strip --------------------------------------------------------
    med_sharpe = df_sum["Sharpe"].median()
    pct_pos    = (df_sum["Sharpe"] > 0).mean()
    best_mdd   = df_sum["MaxDD %"].max()
    worst_mdd  = df_sum["MaxDD %"].min()

    k1, k2, k3, k4 = st.columns(4, gap="small")
    k1.metric("Median Sharpe",   f"{med_sharpe:.2f}")
    k2.metric("Sharpe > 0",      f"{pct_pos:.0%}")
    k3.metric("Best MaxDD %",    f"{best_mdd:.1f}%")
    k4.metric("Worst MaxDD %",   f"{worst_mdd:.1f}%")

    # Results table ----------------------------------------------------
    shared.mp_table(df_sum, key="bench_sum", small=True)

    # Sparkline grid ---------------------------------------------------
    with st.expander("Price trend sparklines (90 \d)"):
        viz.render_sparkline_grid(spark_rows, cols_per_row=4)

    # Bar chart --------------------------------------------------------
    horiz = len(df_sum) > 8
    _DARK = st.get_option("theme.base") != "light"

    if horiz:
        fig = px.bar(df_sum, x="TotalRet %", y="Ticker",
                     orientation='h',
                     title="Total Return % by Ticker",
                     color_discrete_sequence=None if _DARK else px.colors.qualitative.Set2)
    else:
        fig = px.bar(df_sum, x="Ticker", y="TotalRet %",
                     title="Total Return % by Ticker",
                     color_discrete_sequence=None if _DARK else px.colors.qualitative.Set2)

    fig = shared.style_plotly(fig, dark=_DARK)
    st.plotly_chart(fig, use_container_width=True)

    # Export buttons ---------------------------------------------------
    c1, c2 = st.columns(2)
    c1.download_button("Download CSV", df_sum.to_csv().encode(),
                       "benchmarks.csv", "text/csv")
    c2.download_button("Download PNG", fig.to_image(format="png"),
                       "benchmarks.png", mime="image/png")

    # Validation / error log ------------------------------------------
    if skipped or errors:
        with st.expander("⚠️Skipped / error tickers"):
            if skipped:
                st.markdown("**No trades executed:** " + ", ".join(skipped))
            if errors:
                st.markdown("**Errors:**")
                for e in errors:
                    st.text(f"• {e}")

elif run_btn and 'df_sum' in locals() and df_sum.empty:
    st.info("No trades met the criteria for any ticker.")
elif not run_btn and last_df is None:
    st.info("Set parameters, pick a date‑range, then press **Run benchmark**.")

shared.mark_data_fetched()
st.toast("Benchmark complete.", icon="🏁")


