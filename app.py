# MarketPulse — Day 2
# -----------------------------------------------------------
# Core dashboard skeleton: ticker/date inputs → price fetch → plot
# Plotting uses Plotly for interactive visuals
# Later days (3‑5) will graft sentiment, GPT summaries, anomaly alerts, etc.
# -----------------------------------------------------------
import os
from datetime import date, timedelta
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Initial setup
# ------------------------------------------------------------------
load_dotenv()  # pull any keys already saved in a local .env file

st.set_page_config(
    page_title="MarketPulse — Price Intelligence",
    page_icon="📈",
    layout="wide",
)

# ------------------------------------------------------------------
# Sidebar — API key vault (we'll use these on Day 3/4)
# ------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 API Keys")

    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        placeholder="sk-...",
    )
    reddit_client_id = st.text_input(
        "Reddit Client ID",
        type="password",
        value=os.getenv("REDDIT_CLIENT_ID", ""),
    )
    reddit_client_secret = st.text_input(
        "Reddit Client Secret",
        type="password",
        value=os.getenv("REDDIT_CLIENT_SECRET", ""),
    )

    # For backend
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if reddit_client_id:
        os.environ["REDDIT_CLIENT_ID"] = reddit_client_id
    if reddit_client_secret:
        os.environ["REDDIT_CLIENT_SECRET"] = reddit_client_secret

# ------------------------------------------------------------------
# Main panel — Price fetch controls / UI skeleton
# ------------------------------------------------------------------
st.title("📊 MarketPulse — Alpha Preview")

st.subheader("Select asset & date range")

col1, col2, col3 = st.columns([2, 2, 1.2])

with col1:
    ticker = (
        st.text_input(
            "Ticker (Stock or Crypto)",
            value="AAPL",
            help="Examples: AAPL, TSLA, BTC-USD",
            key="ticker_input",
        )
        .upper()
        .strip()
    )

with col2:
    default_start = date.today() - timedelta(days=30)
    date_range = st.date_input(
        "Date range",
        value=(default_start, date.today()),
        max_value=date.today(),
        help="Select start and end dates (max today)",
        key="date_range",
    )

    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        # If single date is somehow returned, treat as start == end
        start_date = date_range
        end_date = date_range

with col3:
    fetch_btn = st.button("📥 Fetch Data")

st.markdown("---")

# ------------------------------------------------------------------
# Data layer — Yahoo Finance
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def get_price_df(ticker_symbol: str, start: date, end: date) -> pd.DataFrame:
    """
    Pull OHLCV data via yfinance and return a tidy DataFrame
    with **single-level** columns.
    """
    df = yf.download(ticker_symbol, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError("No data returned — check ticker symbol or date range.")

    # ── Flatten Multi-Index (yfinance ≥0.2.x often uses it) ────────────────────
    if isinstance(df.columns, pd.MultiIndex):
        # If there is only one ticker level, select it → single-level columns
        tickers_in_cols = df.columns.get_level_values(1).unique()
        if len(tickers_in_cols) == 1:
            df = df.xs(tickers_in_cols[0], axis=1, level=1)
        else:
            # fallback: concatenate levels (“Close_BTC-USD” …) so we never break
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


# ------------------------------------------------------------------
# Plotting helper
# ------------------------------------------------------------------
def plot_price(df: pd.DataFrame, symbol: str):
    if "Close" not in df.columns:
        st.error("Could not find a 'Close' column after flattening; "
                 "please report this ticker.")
        return

    fig = px.line(
        df,
        x=df.index,
        y="Close",
        title=f"{symbol} — closing price",
        labels={"Close": "Price (USD)", "index": "Date"},
    )
    fig.update_traces(line_width=2)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------
# Orchestrate
# ------------------------------------------------------------------
if fetch_btn:
    try:
        # yfinance end date is exclusive — pad by +1 day to include user-selected end
        df_prices = get_price_df(ticker, start_date, end_date + timedelta(days=1))
        st.success(f"Loaded {len(df_prices):,} rows for **{ticker}**")
        plot_price(df_prices, ticker)
    except Exception as ex:
        st.error(f"⚠️ {ex}")

# ------------------------------------------------------------------
# Footer (unchanged)
# ------------------------------------------------------------------
st.caption("Data: Yahoo Finance • Built with Streamlit, Plotly & 💙 MarketPulse 2025")