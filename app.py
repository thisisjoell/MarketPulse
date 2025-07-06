# MarketPulse — Day 4 (Sentiment trend + GPT summary)
# -----------------------------------------------------------
# • Multi-subreddit sentiment (Day 3)                     ✔
# • Daily sentiment trend plot                            ✔
# • GPT-4o/3.5 summary of top Reddit posts                ✔
# -----------------------------------------------------------

import os
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
import praw
import openai

# ── sentiment engines ───────────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# ────────────────────────────────────────────────────────────────

# ------------------------------------------------------------------
# Init
# ------------------------------------------------------------------
load_dotenv()
st.set_page_config(page_title="MarketPulse", page_icon="📈", layout="wide")

# ------------------------------------------------------------------
# Sidebar
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
def get_price_df(sym: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(sym, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError("No price data returned.")
    if isinstance(df.columns, pd.MultiIndex):
        tickers = df.columns.get_level_values(1).unique()
        df = df.xs(tickers[0], axis=1, level=1) if len(tickers) == 1 else df
    df.index = pd.to_datetime(df.index)
    return df

def plot_price(df: pd.DataFrame, sym: str):
    fig = px.line(df, x=df.index, y="Close", title=f"{sym} — closing price",
                  labels={"Close": "Price (USD)", "index": "Date"})
    fig.update_traces(line_width=2)
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
tokenizer, model = _load_roberta()
vader = SentimentIntensityAnalyzer()

def _prep(text: str) -> str:
    out = []
    for t in text.split():
        if t.startswith("@") and len(t) > 1:
            out.append("@user")
        elif t.startswith("http"):
            out.append("http")
        else:
            out.append(t)
    return " ".join(out)

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
# Reddit fetch
# ------------------------------------------------------------------
SUBREDDITS = [
    "stocks", "wallstreetbets", "investing", "options",
    "cryptocurrency", "pennystocks", "news", "technology",
]

@st.cache_data(show_spinner=False, ttl=60*30)
def fetch_reddit_df(sym: str) -> pd.DataFrame:
    if not (os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET")):
        raise RuntimeError("Reddit API keys not set.")
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent="MarketPulse/0.3",
    )
    def grab(sub):
        q = f'"{sym}"'
        return reddit.subreddit(sub).search(q, sort="new", limit=50, time_filter="week")

    seen, recs = set(), []
    for sub in SUBREDDITS:
        for p in grab(sub):
            if p.id in seen:
                continue
            seen.add(p.id)
            text = f"{p.title}\n{p.selftext or ''}"
            recs.append({
                "id": p.id,
                "created": datetime.fromtimestamp(p.created_utc),
                "sub": p.subreddit.display_name,
                "upvotes": p.score,
                "title": p.title,
                "sentiment": sentiment(text),
                "body": p.selftext or "",
                "permalink": f"https://reddit.com{p.permalink}",
            })
    return pd.DataFrame(recs).sort_values("created", ascending=False)

# ------------------------------------------------------------------
# GPT summary
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=60*30)
def gpt_summary(sym: str, top_posts: pd.DataFrame) -> str:
    if not openai.api_key:
        return "⚠️ No OpenAI key provided."
    posts_text = "\n\n".join(
        [f"{i+1}. {r['title']} (▲{r['upvotes']}): {r['body'][:300]}"  # 300 chars cap
         for i, r in top_posts.iterrows()]
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
# Display helpers
# ------------------------------------------------------------------
def sentiment_color(v):
    if v > 0.05:
        return "background-color:#d4f4dd"
    if v < -0.05:
        return "background-color:#f8d7da"
    return "background-color:#f0f0f0"

def show_table(df: pd.DataFrame):
    styled = (
        df[["created", "sub", "upvotes", "sentiment", "title"]]
        .head(20)
        .style.map(sentiment_color, subset=["sentiment"])
        .format({"sentiment": "{:.2f}", "created": lambda d: d.strftime('%Y-%m-%d %H:%M')})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

def show_dist(df: pd.DataFrame):
    hist = px.histogram(df, x="sentiment", nbins=20,
                        title="Sentiment distribution (compound score)")
    hist.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(hist, use_container_width=True)

def show_trend(df: pd.DataFrame):
    daily = df.copy()
    daily["day"] = daily["created"].dt.date
    trend = daily.groupby("day")["sentiment"].mean().reset_index()
    fig = px.line(trend, x="day", y="sentiment", markers=True,
                  title="Daily average Reddit sentiment")
    fig.update_traces(line_width=2)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Main orchestration
# ------------------------------------------------------------------
if fetch_btn:
    try:
        # price
        df_price = get_price_df(ticker, start_date, end_date + timedelta(days=1))
        st.success(f"Price rows: {len(df_price):,}")
        plot_price(df_price, ticker)

        # sentiment
        df_posts = fetch_reddit_df(ticker)
        if df_posts.empty:
            st.warning("No Reddit posts found for that ticker.")
            st.stop()

        # GPT summary – sidebar
        with st.sidebar:
            st.subheader("📢 Reddit Summary")
            top10 = df_posts.sort_values("upvotes", ascending=False).head(10)
            with st.spinner("Generating summary..."):
                summary = gpt_summary(ticker, top10)
            st.write(summary)

        st.subheader("🗣️ Latest Reddit chatter")
        show_table(df_posts)
        show_dist(df_posts)
        show_trend(df_posts)

    except Exception as err:
        st.error(f"⚠️ {err}")

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.caption(
    "Data: Yahoo Finance & Reddit • Sentiment: RoBERTa+VADER • "
    "GPT summary powered by OpenAI • Built with Streamlit — MarketPulse 2025"
)
