# MarketPulse — Day 3 (RoBERTa upgrade + multi-subreddit)
# -----------------------------------------------------------
# • Price chart (Day 2)                                   ✔
# • Reddit sentiment (RoBERTa primary, VADER fallback)    ✔
# • Pulls posts from 8 subreddits                         ✔
#   r/stocks, r/wallstreetbets, r/investing, r/options,
#   r/cryptocurrency, r/pennystocks, r/news, r/technology
# -----------------------------------------------------------

import os
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
import praw

# ── sentiment engines ───────────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# ────────────────────────────────────────────────────────────────

# ------------------------------------------------------------------
# Initial setup
# ------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="MarketPulse — Price & Sentiment",
    page_icon="📈",
    layout="wide",
)

# ------------------------------------------------------------------
# Sidebar — API key vault
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

# ------------------------------------------------------------------
# Main panel — controls
# ------------------------------------------------------------------
st.title("📊 MarketPulse — Alpha Preview")

col1, col2, col3 = st.columns([2, 2, 1.2])

with col1:
    ticker = st.text_input("Ticker (Stock or Crypto)", value="AAPL").upper().strip()

with col2:
    default_start = date.today() - timedelta(days=30)
    date_range = st.date_input("Date range", value=(default_start, date.today()), max_value=date.today())
    start_date, end_date = (date_range if isinstance(date_range, tuple) else (date_range, date_range))

with col3:
    fetch_btn = st.button("📥 Fetch Data & Sentiment")

st.markdown("---")

# ------------------------------------------------------------------
# Price helpers
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_price_df(ticker_symbol: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(ticker_symbol, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError("No price data returned.")
    if isinstance(df.columns, pd.MultiIndex):
        tickers = df.columns.get_level_values(1).unique()
        df = df.xs(tickers[0], axis=1, level=1) if len(tickers) == 1 else df
    df.index = pd.to_datetime(df.index)
    return df


def plot_price(df: pd.DataFrame, symbol: str):
    fig = px.line(df, x=df.index, y="Close",
                  title=f"{symbol} — closing price",
                  labels={"Close": "Price (USD)", "index": "Date"})
    fig.update_traces(line_width=2)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Sentiment engines
# ------------------------------------------------------------------
HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


@st.cache_resource(show_spinner=False)
def _load_roberta():
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
    return tok, mdl


tokenizer_roberta, model_roberta = _load_roberta()
analyzer_vader = SentimentIntensityAnalyzer()


def _preprocess_social(text: str) -> str:
    clean = []
    for t in text.split():
        if t.startswith("@") and len(t) > 1:
            clean.append("@user")
        elif t.startswith("http"):
            clean.append("http")
        else:
            clean.append(t)
    return " ".join(clean)


def _roberta_compound(text: str) -> float:
    text = _preprocess_social(text)
    encoded = tokenizer_roberta(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model_roberta(**encoded).logits[0].numpy()
    probs = softmax(logits)  # [neg, neu, pos]
    return float(probs[2] - probs[0])  # pos − neg


def robust_sentiment(text: str) -> float:
    try:
        return _roberta_compound(text)
    except Exception:
        return analyzer_vader.polarity_scores(text)["compound"]

# ------------------------------------------------------------------
# Reddit fetch & sentiment
# ------------------------------------------------------------------
SUBREDDITS = [
    "stocks",
    "wallstreetbets",
    "investing",
    "options",
    "cryptocurrency",
    "pennystocks",
    "news",
    "technology",
]


@st.cache_data(show_spinner=False, ttl=60 * 30)  # 30-min cache
def fetch_reddit_df(symbol: str) -> pd.DataFrame:
    if not (os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET")):
        raise RuntimeError("Reddit API keys not set (sidebar).")

    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent="MarketPulseSentiment/0.2 (by u/your_username)",
    )

    def grab(sub):
        query = f'"{symbol}"'
        return list(reddit.subreddit(sub).search(query, sort="new", limit=50, time_filter="week"))

    # collect & de-dup posts across all subs
    seen = set()
    records = []

    for sub in SUBREDDITS:
        for p in grab(sub):
            if p.id in seen:  # skip duplicates
                continue
            seen.add(p.id)
            text = f"{p.title}\n{p.selftext or ''}"
            score = robust_sentiment(text)
            records.append(
                {
                    "created": datetime.fromtimestamp(p.created_utc),
                    "sub": p.subreddit.display_name,
                    "upvotes": p.score,
                    "title": p.title,
                    "sentiment": score,
                    "permalink": f"https://reddit.com{p.permalink}",
                }
            )

    df = pd.DataFrame(records).sort_values("created", ascending=False)
    return df


def sentiment_color(val):
    if val > 0.05:
        return "background-color:#d4f4dd"  # bullish
    if val < -0.05:
        return "background-color:#f8d7da"  # bearish
    return "background-color:#f0f0f0"      # neutral


def display_sentiment(df: pd.DataFrame):
    if df.empty:
        st.info("No recent Reddit posts mentioning this symbol.")
        return

    st.subheader("🗣️ Latest Reddit chatter")

    styled = (
        df[["created", "sub", "upvotes", "sentiment", "title"]]
        .head(20)
        .style.applymap(sentiment_color, subset=["sentiment"])
        .format({"sentiment": "{:.2f}", "created": lambda d: d.strftime("%Y-%m-%d %H:%M")})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    hist = px.histogram(df, x="sentiment", nbins=20,
                        title="Sentiment distribution (compound score)")
    hist.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(hist, use_container_width=True)

# ------------------------------------------------------------------
# Orchestrate
# ------------------------------------------------------------------
if fetch_btn:
    try:
        df_prices = get_price_df(ticker, start_date, end_date + timedelta(days=1))
        st.success(f"Loaded {len(df_prices):,} price rows for **{ticker}**")
        plot_price(df_prices, ticker)

        df_reddit = fetch_reddit_df(ticker)
        display_sentiment(df_reddit)

    except Exception as exc:
        st.error(f"⚠️ {exc}")

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.caption(
    "Data: Yahoo Finance & Reddit • Sentiment: RoBERTa (TweetEval) with VADER fallback • "
    "Built with Streamlit & Plotly — MarketPulse 2025"
)
