# MarketPulse — Day 5 (adds anomaly detector + mini-forecast)
# -----------------------------------------------------------
# • Multi-subreddit sentiment (Day 3)                      ✔
# • Daily sentiment trend & GPT summary (Day 4)            ✔
# • 🚨 Anomaly alert (sent ↑30 %, price ↓2 %)              ✔
# • 🔮 Mini linear-regression forecast                     ✔
# -----------------------------------------------------------

import os
from datetime import date, datetime, timedelta

import numpy as np              # 🆕
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
import praw
import openai
from sklearn.linear_model import LinearRegression   # 🆕

# ── sentiment engines ───────────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import alpha_lab
import models
# ────────────────────────────────────────────────────────────────

# ------------------------------------------------------------------
# Init
# ------------------------------------------------------------------
load_dotenv()
st.set_page_config(page_title="MarketPulse", page_icon="📈", layout="wide")

# ------------------------------------------------------------------
# Sidebar (unchanged)
# ------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 API Keys")
    openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    reddit_client_id = st.text_input("Reddit Client ID", type="password", value=os.getenv("REDDIT_CLIENT_ID", ""))
    reddit_client_secret = st.text_input("Reddit Client Secret", type="password", value=os.getenv("REDDIT_CLIENT_SECRET", ""))

    for k, v in {
        "OPENAI_API_KEY": openai_key,
        "REDDIT_CLIENT_ID": reddit_client_id,
        "REDDIT_CLIENT_SECRET": reddit_client_secret,
    }.items():
        if v:
            os.environ[k] = v
openai.api_key = os.getenv("OPENAI_API_KEY", "")

# ------------------------------------------------------------------
# Controls
# ------------------------------------------------------------------
st.title("📊 MarketPulse — Beta")

col1, col2, col3 = st.columns([2, 2, 1.2])
with col1:
    ticker = st.text_input("Ticker (Stock or Crypto)", value="AAPL").upper().strip()
with col2:
    default_start = date.today() - timedelta(days=60)  # 60 d to feed the model
    date_range = st.date_input("Date range", value=(default_start, date.today()), max_value=date.today())
    start_date, end_date = (date_range if isinstance(date_range, tuple) else (date_range, date_range))
with col3:
    fetch_btn = st.button("📥 Fetch Data & Sentiment")
st.markdown("---")

# ------------------------------------------------------------------
# Price helpers (unchanged)
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_price_df(sym: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError("No price data returned.")
    if isinstance(df.columns, pd.MultiIndex):
        tickers = df.columns.get_level_values(1).unique()
        df = df.xs(tickers[0], axis=1, level=1) if len(tickers) == 1 else df
    df.index = pd.to_datetime(df.index)
    return df

def plot_price(df: pd.DataFrame, sym: str):
    fig = px.line(df, x=df.index, y="Close",
                  title=f"{sym} — closing price",
                  labels={"Close": "Price (USD)", "index": "Date"})
    fig.update_traces(line_width=2)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Sentiment engines (unchanged)
# ------------------------------------------------------------------
HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
@st.cache_resource(show_spinner=False)
def _load_roberta():
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
    return tok, mdl
tokenizer, model = _load_roberta()
vader = SentimentIntensityAnalyzer()

def _prep(text: str) -> str:
    return " ".join("@user" if t.startswith("@") else "http" if t.startswith("http") else t for t in text.split())

def _rob_score(text: str) -> float:
    encoded = tokenizer(_prep(text), return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**encoded).logits[0].numpy()
    p = softmax(logits)
    return float(p[2] - p[0])   # pos − neg

def sentiment(text: str) -> float:
    try:
        return _rob_score(text)
    except Exception:
        return vader.polarity_scores(text)["compound"]

# ------------------------------------------------------------------
# Reddit fetch (unchanged)
# ------------------------------------------------------------------
SUBREDDITS = [
    "stocks", "wallstreetbets", "investing", "options",
    "cryptocurrency", "pennystocks", "news", "technology",
]

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_reddit_df(sym: str, start_date: date, end_date: date) -> pd.DataFrame:
    if not (os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET")):
        raise RuntimeError("Reddit API keys not set.")

    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent="MarketPulse/0.5",
    )

    def grab(sub):
        return reddit.subreddit(sub).search(
            f'"{sym}"', sort="top", limit=100, time_filter="year"
        )

    seen, recs = set(), []
    for sub in SUBREDDITS:
        for p in grab(sub):
            post_dt = datetime.fromtimestamp(p.created_utc)
            if not (start_date <= post_dt.date() <= end_date):
                continue
            if p.id in seen:
                continue
            seen.add(p.id)
            text = f"{p.title}\n{p.selftext or ''}"
            recs.append({
                "id": p.id,
                "created": post_dt,
                "sub": p.subreddit.display_name,
                "upvotes": p.score,
                "title": p.title,
                "body": p.selftext or "",
                "permalink": f"https://reddit.com{p.permalink}",
                "sentiment": sentiment(text),
            })
    return pd.DataFrame(recs).sort_values("created", ascending=False)

# ────────────────────────────────────────────────────────────
# Helper: price + sentiment fetch  (cached for 1 hour)
# ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=60 * 60)  # 60 min = 3 600 s
def fetch_price_and_trend(
    sym: str,
    start: date,
    end: date,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Cached fetch so multi-ticker benchmarks don’t hammer the APIs.

    Returns
    -------
    price_daily : Series           adjusted close, DatetimeIndex
    trend_df    : DataFrame        ['day', 'sentiment']
    """
    # --- price --------------------------------------------------
    price_daily = get_price_df(sym, start, end + timedelta(days=1))["Close"]

    # --- sentiment ---------------------------------------------
    posts = fetch_reddit_df(sym, start, end)
    posts["day"] = posts["created"].dt.date
    trend_df = posts.groupby("day")["sentiment"].mean().reset_index()

    return price_daily, trend_df

# ------------------------------------------------------------------
# GPT summary (unchanged)
# ------------------------------------------------------------------
def top_posts_hybrid(df: pd.DataFrame, k: int = 10, tau_hrs: float = 72) -> pd.DataFrame:
    if df.empty:
        return df
    up_norm = df["upvotes"].clip(lower=0)
    up_norm = up_norm / up_norm.max() if up_norm.max() else up_norm
    now_naive = pd.Timestamp.utcnow().tz_localize(None)
    created_naive = pd.to_datetime(df["created"]).dt.tz_localize(None)
    age_hrs = (now_naive - created_naive).dt.total_seconds() / 3600
    rec_w   = np.exp(-age_hrs / tau_hrs)
    df = df.copy()
    df["score_hybrid"] = 0.6 * up_norm + 0.4 * rec_w
    return df.sort_values("score_hybrid", ascending=False).head(k)

@st.cache_data(show_spinner=False, ttl=60*30)
def gpt_summary(sym: str, top_posts: pd.DataFrame) -> str:
    if not openai.api_key:
        return "⚠️ No OpenAI key provided."
    posts_text = "\n\n".join(
        [f"{i+1}. {r['title']} (▲{r['upvotes']}): {r['body'][:300]}" for i, r in top_posts.iterrows()]
    )
    prompt = (
        f"You are a financial analyst. Summarize the overall market sentiment for {sym} "
        f"based on the following Reddit posts. Highlight bullish vs bearish tone, recurring themes, "
        f"and any noteworthy catalysts or concerns in 3–4 sentences.\n\nPosts:\n{posts_text}"
    )
    try:
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=160,
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ GPT error: {e}"

# ------------------------------------------------------------------
# Display helpers (unchanged)
# ------------------------------------------------------------------
def sentiment_color(v):
    if v > 0.05:  return "background-color:#d4f4dd"
    if v < -0.05: return "background-color:#f8d7da"
    return "background-color:#f0f0f0"

def show_table(df: pd.DataFrame):
    styled = (
        df[["created", "sub", "upvotes", "sentiment", "title"]]
        .head(20)
        .style.applymap(sentiment_color, subset=["sentiment"])
        .format({"sentiment": "{:.2f}", "created": lambda d: d.strftime('%Y-%m-%d %H:%M')})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

def show_dist(df: pd.DataFrame):
    st.plotly_chart(
        px.histogram(df, x="sentiment", nbins=20,
                     title="Sentiment distribution (compound score)")
        .update_layout(margin=dict(l=0, r=0, t=40, b=0)),
        use_container_width=True,
    )

def show_trend(tr):
    st.plotly_chart(
        px.line(tr, x="day", y="sentiment", markers=True,
                title="Daily average Reddit sentiment")
        .update_traces(line_width=2)
        .update_layout(margin=dict(l=0, r=0, t=40, b=0)),
        use_container_width=True,
    )

# ------------------------------------------------------------------
# Main orchestration
# ------------------------------------------------------------------
if fetch_btn:
    try:
        # ========== PRICE ==========
        df_price = get_price_df(ticker, start_date, end_date + timedelta(days=1))
        st.success(f"Price rows: {len(df_price):,}")
        plot_price(df_price, ticker)

        # ========== SENTIMENT POSTS ==========
        df_posts = fetch_reddit_df(ticker, start_date, end_date)
        if df_posts.empty:
            st.warning("No Reddit posts found for that ticker.")
            st.stop()

        # ========== GPT SUMMARY ==========
        with st.sidebar:
            st.subheader("📢 Reddit Summary")
            top10 = top_posts_hybrid(df_posts, k=10, tau_hrs=72)
            with st.spinner("Generating summary..."):
                st.write(gpt_summary(ticker, top10))

        # ---------- Daily sentiment ----------
        df_posts["day"] = df_posts["created"].dt.date
        trend = df_posts.groupby("day")["sentiment"].mean().reset_index()

        # ---------- Daily price % change ----------
        price_daily = df_price["Close"].resample("1D").last().dropna()
        price_pct = price_daily.pct_change()
        trend["pct_next"] = price_pct.shift(-1).reindex(trend["day"]).values

        # persist for Alpha Lab
        st.session_state["price_daily"] = price_daily
        st.session_state["trend"] = trend

        # ========== MINI FORECAST ==========
        valid = trend.dropna(subset=["pct_next"])
        if len(valid) >= 10:
            X = valid[["sentiment"]].values
            y = valid["pct_next"].values
            lr = LinearRegression().fit(X, y)
            pred_pct = float(lr.predict([[trend.iloc[-1]["sentiment"]]])[0])
            st.metric(
                "🔮 Forecast",
                f"{pred_pct * 100:+.2f}%",
                help="Predicted using linear regression on sentiment data",
            )
        else:
            st.metric("🔮 Forecast", "n/a", help="Not enough history yet")

        # ========== SMART ANOMALY DETECTORS =========================================
        def rolling_corr(series_a, series_b, win: int = 7):
            a_tail = series_a[-win:]
            b_tail = series_b[-win:]
            mask = ~np.isnan(a_tail) & ~np.isnan(b_tail)
            if mask.sum() < win:
                return np.nan
            return np.corrcoef(a_tail[mask], b_tail[mask])[0, 1]

        if len(trend) >= 2:
            sent_today = trend.iloc[-1]["sentiment"]
            sent_prev = trend.iloc[-2]["sentiment"]
            price_today = price_daily.iloc[-1]
            price_prev = price_daily.iloc[-2]
            pct_today = (price_today - price_prev) / price_prev if price_prev else 0.0
            roll_std = price_pct.rolling(30).std().iloc[-1]
            z_score = pct_today / roll_std if roll_std else 0.0
            corr7 = rolling_corr(trend["sentiment"].values, price_pct.values, 7)

            neg_flags = 0
            if z_score <= -1.5:
                neg_flags += 1
            if (sent_today - sent_prev) / (abs(sent_prev) or 1) >= 0.30:
                neg_flags += 1
            if sent_today > 0.05 and pct_today < -0.02:
                neg_flags += 1
            if not np.isnan(corr7) and corr7 < -0.3:
                neg_flags += 1

            pos_flags = 0
            if z_score >= 1.5:
                pos_flags += 1
            if sent_today < 0.05:
                pos_flags += 1
            if sent_today < sent_prev and pct_today > 0.02:
                pos_flags += 1
            if not np.isnan(corr7) and corr7 < 0.1:
                pos_flags += 1

            if neg_flags >= 2:
                st.warning("🚨 **Divergence Detected:** Price dropped but sentiment remains optimistic "
                           "(≥ 2 confirming signals).")
            if pos_flags >= 2:
                st.info("📈 **Early Breakout Signal:** Price surged while sentiment is muted "
                        "(≥ 2 confirming signals).")

        # ========== VISUALS ==========
        st.subheader("🗣️ Latest Reddit chatter")
        show_table(df_posts)
        show_dist(df_posts)
        show_trend(trend)

    except Exception as err:
        st.error(f"⚠️ {err}")

# ------------------------------------------------------------------
# 🧠 Alpha Lab – strategy simulator (always visible)
# ------------------------------------------------------------------
with st.expander("🧠 Alpha Lab — Strategy Simulator"):
    tab_manual, tab_model = st.tabs(["Manual rules", "Model signal"])
    # ---------- Manual rules tab ----------
    with tab_manual:
        # ── manual-rules sliders ─────────────────────────────────────────
        s_thr = st.slider("Sentiment >", 0.00, 1.00, 0.05, 0.01)  # 0.01 step
        p_drop = st.slider("Price drop % ≥", 0.0, 10.0, 2.0, 0.10) / 100  # 0.10 % step
        p_rise = st.slider("Take-profit % ≥", 0.0, 10.0, 5.0, 0.10) / 100  # 0.10 % step
        vol_th = st.slider("Volatility spike >", 0.0, 5.0, 1.5, 0.10) / 100  # 0.10 % step
        mom_th = st.slider("Bear momentum % <", 0.0, 5.0, 1.0, 0.10) / 100  # 0.10 % step
        tx_bps = st.slider("Tx-cost (bps), per side", 0, 50, 5, 1)  # 1 bp step
        stop_ls = st.slider("Stop-loss %", 0.0, 20.0, 0.0, 0.50) / 100  # 0.50 % step
        # ── NEW technical-indicator guards ────────────────────────────
        rsi_max = st.slider("RSI ≤", 10, 70, 30, 1)  # oversold filter
        use_macd = st.checkbox("Require MACD histogram > 0", value=False)

        if st.button("Run manual back-test"):
            price_daily = st.session_state.get("price_daily")
            trend       = st.session_state.get("trend")
            if price_daily is None:
                st.warning("Run the main fetch first.")
            else:
                merged = alpha_lab.merge_price_sentiment(price_daily, trend)
                eq, trades = alpha_lab.run_manual_strategy(
                    merged,
                    s_thr, p_drop, p_rise,
                    vol_thresh=vol_th,
                    mom_thresh=mom_th,
                    rsi_max=rsi_max,  # ▶ forward
                    macd_req=use_macd,  # ▶ forward
                    stop_loss_pct=stop_ls,
                    tx_cost_bps=tx_bps,
                )
                alpha_lab.show_results(eq, trades)
    with st.expander("🔧 Hyper-parameter grid search"):
        if st.button("Run grid on current ticker"):
            price_daily = st.session_state.get("price_daily")
            trend = st.session_state.get("trend")
            if price_daily is None:
                st.info("Run **Fetch Data & Sentiment** first.")
                st.stop()

            merged = alpha_lab.merge_price_sentiment(price_daily, trend)

            grid = {
                "sent_threshold": [0.02, 0.05, 0.10],
                "price_drop_pct": [0.01, 0.02],
                "price_rise_pct": [0.03, 0.05],  # required
                "vol_thresh": [0.01, 0.015],
                #"tx_cost_bps": [0, 5],
            }

            # SINGLE call — no extra loop, no extra progress bar
            res = alpha_lab.grid_search(
                merged,
                grid,
                lambda d, **k: alpha_lab.run_manual_strategy(d, **k),
                base_kwargs=dict(  # keep these fixed
                    mom_thresh=mom_th,
                    stop_loss_pct=stop_ls,
                    tx_cost_bps=tx_bps,
                )
            )
            st.dataframe(res, use_container_width=True)

    with st.expander("📊 Multi-ticker benchmark"):
        tickers_str = st.text_input("Tickers (comma)", "AAPL, TSLA, NVDA")
        if st.button("Run across tickers"):
            tick_list = [x.strip().upper() for x in tickers_str.split(",") if x.strip()]

            df_sum = alpha_lab.batch_benchmark(
                tick_list,
                lambda sym: fetch_price_and_trend(sym, start_date, end_date),
                alpha_lab.merge_price_sentiment,
                alpha_lab.run_manual_strategy,
                sent_threshold=s_thr,
                price_drop_pct=p_drop,
                price_rise_pct=p_rise,
                vol_thresh=vol_th,
                mom_thresh=mom_th,

                # NEW – pass the two extra guards
                rsi_max=rsi_max,
                macd_req=use_macd,

                stop_loss_pct=stop_ls,
                tx_cost_bps=tx_bps,
            )
            st.dataframe(df_sum, use_container_width=True)
    # ------------------------------------------------------------------
    # 🧩 Correlation analysis
    # ------------------------------------------------------------------
    with st.expander("🔗 Cross-asset correlation analysis"):
        tick_sel = st.text_input("Tickers to analyse (comma)",
                                 "AAPL, TSLA, NVDA, BTC-USD")
        mode = st.selectbox("Correlation of …",
                            ["Price returns", "Manual-strategy equity"])
        win_days = st.slider("Rolling window (days)", 10, 120, 30, 5)

        if st.button("Compute correlations"):
            syms = [x.strip().upper() for x in tick_sel.split(",") if x.strip()]
            if len(syms) < 2:
                st.warning("Need at least two tickers.")
                st.stop()

            price_map, eq_map = {}, {}

            # ▶▶ NEW wrapper — one spinner for the whole job
            with st.spinner("Running back-tests and correlations …"):
                bar = st.progress(0.0, text="Fetching …")

                for i, s in enumerate(syms, start=1):
                    # —— (1) pull prices ——————————————
                    try:
                        price_map[s] = get_price_df(
                            s, start_date, end_date + timedelta(1)
                        )["Close"]
                    except Exception as e:
                        st.error(f"{s}: {e}")
                        st.stop()

                    # —— (2) optional strategy equity ————
                    if mode == "Manual-strategy equity":
                        price, trend = fetch_price_and_trend(s, start_date, end_date)
                        merged = alpha_lab.merge_price_sentiment(price, trend)
                        eq, _ = alpha_lab.run_manual_strategy(
                            merged,
                            sent_threshold=s_thr,
                            price_drop_pct=p_drop,
                            price_rise_pct=p_rise,
                            vol_thresh=vol_th,
                            mom_thresh=mom_th,
                            rsi_max=rsi_max,
                            macd_req=use_macd,
                            stop_loss_pct=stop_ls,
                            tx_cost_bps=tx_bps,
                        )
                        eq_map[s] = eq["equity"]

                    bar.progress(i / len(syms))

                bar.empty()  # 🧹 tidy up the progress bar

            # —— build matrix + heat-map ——————————————
            if mode == "Price returns":
                corr_df = alpha_lab.corr_matrix(price_map)
            else:
                corr_df = alpha_lab.corr_matrix(eq_map)

            alpha_lab.plot_corr_heat(corr_df)

            # —— rolling ρ plots (always based on returns) ———
            st.subheader("Rolling correlations")
            for i in range(len(syms) - 1):
                for j in range(i + 1, len(syms)):
                    if mode == "Price returns":
                        a, b = price_map[syms[i]], price_map[syms[j]]
                    else:
                        a, b = eq_map[syms[i]], eq_map[syms[j]]

                    rho = alpha_lab.rolling_corr_pair(a, b, win=win_days)
                    alpha_lab.plot_rolling_corr(rho, syms[i], syms[j])

    # ---------- Model signal tab ----------
    with tab_model:
        model_type = st.selectbox("Model", ["Bayesian", "ARIMA", "Prophet"])
        thresh = st.slider("Probability / forecast threshold", 0.50, 0.90, 0.70, 0.01)
        # NEW ▶ choose single-fit or 5-fold walk-forward
        use_cv = st.checkbox("Walk-forward CV (5 folds)", value=False)
        if st.button("Train model & back-test"):
            price_daily = st.session_state.get("price_daily")
            trend = st.session_state.get("trend")
            if price_daily is None:
                st.warning("Run the main fetch first.")
                st.stop()

            merged = alpha_lab.merge_price_sentiment(price_daily, trend)

            # ----- map selection → callables ----------------------------------
            if model_type == "Bayesian":
                train_fn = models.train_bayes
                strat_fn = lambda df_slice, mdl: alpha_lab.run_model_strategy_bayes(
                    df_slice, mdl, prob_thresh=thresh
                )

            elif model_type == "ARIMA":
                train_fn = models.train_arima
                strat_fn = lambda df_slice, mdl: alpha_lab.run_model_strategy_arima(
                    df_slice, mdl, ret_thresh=(thresh - 0.5) / 10
                )

            else:  # Prophet
                train_fn = models.train_prophet
                strat_fn = lambda df_slice, mdl: alpha_lab.run_model_strategy_prophet(
                    df_slice,
                    mdl,
                    df_slice.index[-1] + pd.Timedelta(days=1),
                    ret_thresh=(thresh - 0.5) / 10,
                )

            # ----- single fit vs walk-forward ---------------------------------
            if use_cv:
                with st.spinner("Running walk-forward CV…"):
                    eq, trades = alpha_lab.walk_forward_backtest(
                        merged, train_fn, strat_fn
                    )
            else:
                mdl = train_fn(merged)  # fit on the full window
                eq, trades = strat_fn(merged, mdl)

            alpha_lab.show_results(eq, trades)

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.caption(
    "Data: Yahoo Finance & Reddit • Sentiment: RoBERTa+VADER • "
    "GPT summary • Forecast: LinearRegression • MarketPulse 2025"
)
