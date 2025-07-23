# shared.py
import os
from datetime import date, timedelta, datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import praw
import openai
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import streamlit as st
from datetime import date, timedelta
from datetime import datetime, timezone
from datetime import date as _date, datetime as _dt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_FRESH_MAX_MIN = 30  # minutes considered "fresh"

def _coerce_date(obj):
    """
    Convert anything the date‑picker can emit into a `datetime.date`.

    • date ‑‑> date
    • datetime ‑‑> date part
    • 1‑element tuple ‑‑> unwrap and recurse (happens while user is still picking)
    • ISO string / "today" -> parsed
    """
    if isinstance(obj, date):
        return obj

    if isinstance(obj, datetime):
        return obj.date()

    # Streamlit range picker, midway through selection, returns (date,) tuples
    if isinstance(obj, tuple) and len(obj) == 1:
        return _coerce_date(obj[0])

    if isinstance(obj, str):
        if obj.lower() == "today":
            return date.today()
        return date.fromisoformat(obj)

    # Anything else – let caller decide what to do
    raise ValueError(f"Unsupported date type: {type(obj)}")
def add_link(label: str, url: str, icon: str | None = None):
    """Inline link chip; safe fallback that doesn't depend on extras."""
    icon_html = f"{icon}&nbsp;" if icon else ""
    st.markdown(
        f"{icon_html}<a href='{url}' target='_blank' style='text-decoration:none;'>{label}</a>",
        unsafe_allow_html=True,
    )
# ------------------------------------------------------------------
# UI coloring
# ------------------------------------------------------------------

COLORS = {
    "bull": "#2ecc71",   # or px.colors.sequential.Greens[4]
    "bear": "#e74c3c",   # or px.colors.sequential.Reds[4]
    "neutral": "#888888",
    "accent": "#00b7ff",
}

ICONS = {
    "divergence_warn": "🟥",   # alt: "🚨"
    "divergence_info": "🟧",
    "good": "🟢",
    "bad": "🔴",
    "run": "🏃",
    "calc": "🧮",
    "model": "🤖",
    "sentiment": "🗣️",
}
def icon(name: str) -> str:
    """Safe icon lookup; returns '' if name missing."""
    return ICONS.get(name, "")
def inject_global_css(
    *,
    use_poppins: bool = True,
    kpi_compact: bool = False,
    dataframe_font_scale: float = 0.85,
):
    """
    Inject app‑wide CSS overrides (dark‑UI friendly, Streamlit‑safe).

    Parameters
    ----------
    use_poppins : bool, default True
        Load & apply Google Poppins font.
    kpi_compact : bool, default False
        Tighten vertical padding around st.metric widgets.
    dataframe_font_scale : float, default 0.85
        % scale for tables rendered inside <div class="mp-table">...</div>.
    """
    google_font_import = (
        "@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');"
        if use_poppins
        else ""
    )
    df_font_pct = f"{dataframe_font_scale * 100:.0f}%"

    kpi_rule = """
        /* compact KPI row (metrics) */
        div[data-testid="stMetric"] {
            padding-top: 0.25rem;
            padding-bottom: 0.25rem;
        }
    """ if kpi_compact else ""

    st.markdown(
        f"""
        <style>
        {google_font_import}

        :root {{
            --mp-accent: #00b7ff;
        }}

        html, body, [class*="css"] {{
            font-family: {'Poppins' if use_poppins else 'inherit'}, sans-serif;
        }}

        /* tighter section spacing */
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 4rem;
        }}

        /* heavier metric labels for readability */
        div[data-testid="stMetric"] label {{
            font-weight: 600 !important;
        }}

        {kpi_rule}

        /* ==============================================================
           KPI CARD UPGRADE
           Add classes=['kpi'] to the st.columns() call and every st.metric
           inside gets a sleek dark card background + subtle shadow.
        ============================================================== */
        .kpi [data-testid="stMetric"] {{
            background: #1b1e26;
            border: 1px solid #2d2d2d;
            border-radius: 6px;
            padding: 0.75rem 1.25rem;
            box-shadow: 0 0 8px rgba(0,0,0,.45);
            /* ensure the card doesn’t inherit unwanted margins */
            margin: 0 !important;
        }}
        .kpi [data-testid="stMetric"] label {{
            color: #c8c8c8 !important;
        }}
        .kpi [data-testid="stMetric"] div[data-testid="stMetricValue"] {{
            font-size: 1.35rem !important;
        }}

        /* ------------------------------------------------------------------
           Robust DataFrame font scaling
        ------------------------------------------------------------------ */
        .mp-table table {{ font-size: {df_font_pct}; }}
        .mp-table-sm table {{ font-size: 70%; }}

        /* make expander headers look more like cards */
        .streamlit-expanderHeader {{
            font-weight: 600;
            background-color: #1b1e26;
            border-radius: 4px;
            padding: 0.35rem 0.5rem;
        }}

        /* dim code blocks for dark theme */
        code, pre {{
            background-color: #1b1e26 !important;
            color: #f5f5f5 !important;
            border-radius: 3px;
            padding: 0.1rem 0.25rem;
        }}

        /* semantic colour helpers */
        .bullish {{ color: #16c172; font-weight: 600; }}
        .bearish {{ color: #ff4d4d; font-weight: 600; }}
        .neutral {{ color: #c0c0c0; font-weight: 400; }}

        /* badge chips */
        .chip {{
            display: inline-block;
            padding: 0.1rem 0.4rem;
            border-radius: 999px;
            font-size: 0.75rem;
            line-height: 1.2;
            margin-left: 0.25rem;
        }}
        .chip-good {{ background:#123d2b; color:#16c172; }}
        .chip-bad  {{ background:#3d1212; color:#ff4d4d; }}
        .chip-warn {{ background:#5a4b15; color:#ffda6a; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def mp_table(df_or_styler, *, key=None, width=True, hide_index=True, small=False):
    cls = "mp-table-sm" if small else "mp-table"
    with st.container():
        st.markdown(f"<div class='{cls}'>", unsafe_allow_html=True)
        st.dataframe(df_or_styler, use_container_width=width, hide_index=hide_index, key=key)
        st.markdown("</div>", unsafe_allow_html=True)
def style_plotly(fig, *, dark=True):
    if dark:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111217",
            plot_bgcolor="#1b1e26",
            font=dict(color="#f5f5f5"),
            hovermode="x unified",
        )
    # minor grid softening
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")
    return fig
# ------------------------------------------------------------------
# API key sync
# ------------------------------------------------------------------
def sync_api_keys(from_session: bool = True) -> bool:
    """
    Ensure OpenAI key (and optionally Reddit creds) are available.

    If `from_session` is True, look first in st.session_state for:
        'OPENAI_API_KEY', 'REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET'
    Fall back to os.environ.

    Side-effects:
    - Writes any found values into os.environ (so downstream libs see them).
    - Sets openai.api_key.
    Returns True if an OpenAI key is available, else False.
    """
    key_names = ["OPENAI_API_KEY", "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"]
    if from_session:
        for k in key_names:
            v = st.session_state.get(k)
            if v:  # user typed it earlier
                os.environ[k] = v

    # now read from env and push into OpenAI
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    return bool(openai.api_key)
# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SUBREDDITS = [
    "stocks","wallstreetbets","investing","options",
    "cryptocurrency","pennystocks","news","technology"
]

# ------------------------------------------------------------------
# Model loaders (cached once per session)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
    return tok, mdl, SentimentIntensityAnalyzer()

def prep_text(text: str) -> str:
    return " ".join(
        "@user" if t.startswith("@")
        else "http" if t.startswith("http")
        else t
        for t in text.split()
    )

def rob_score(text: str, tokenizer, model) -> float:
    encoded = tokenizer(prep_text(text), return_tensors="pt",
                        truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**encoded).logits[0].numpy()
    p = softmax(logits)
    return float(p[2] - p[0])   # pos − neg

def sentiment(text: str, tokenizer, model, vader) -> float:
    try:
        return rob_score(text, tokenizer, model)
    except Exception:
        return vader.polarity_scores(text)["compound"]

# ------------------------------------------------------------------
# Data fetchers
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_price_df(sym: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(sym, start=start, end=end,
                     progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError("No price data returned.")
    if isinstance(df.columns, pd.MultiIndex):
        tics = df.columns.get_level_values(1).unique()
        df = df.xs(tics[0], axis=1, level=1) if len(tics) == 1 else df
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_reddit_df(sym: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Fetch Reddit posts that mention *sym* between start_date and end_date
    (inclusive) and attach a sentiment score.

    • Uses the cached HF‑RoBERTa + VADER combo from `load_models()`.
    • Always returns a DataFrame that already contains **all expected
      columns**, so downstream code never KeyErrors even when no posts
      are found.
    """
    tokenizer, model, vader = load_models()

    if not (os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET")):
        raise RuntimeError("Reddit API keys not set.")

    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent="MarketPulse/0.5",
    )

    def _grab(sub):
        # search top posts from the past year containing the exact symbol
        return reddit.subreddit(sub).search(
            f'"{sym}"', sort="top", limit=100, time_filter="year"
        )

    seen, recs = set(), []
    for sub in SUBREDDITS:
        for p in _grab(sub):
            ts = datetime.fromtimestamp(p.created_utc)

            # time‑window filter first
            if not (start_date <= ts.date() <= end_date):
                continue

            # skip duplicates resurfacing across subs
            if p.id in seen:
                continue
            seen.add(p.id)

            text = f"{p.title}\n{p.selftext or ''}"
            recs.append(
                {
                    "id": p.id,
                    "created": ts,
                    "sub": p.subreddit.display_name,
                    "upvotes": p.score,
                    "title": p.title,
                    "body": p.selftext or "",
                    "permalink": f"https://reddit.com{p.permalink}",
                    "sentiment": sentiment(text, tokenizer, model, vader),
                }
            )

    # ------------------------------------------------------------------
    #  Build the DataFrame – guarantee all columns even if recs is empty
    # ------------------------------------------------------------------
    cols = [
        "id",
        "created",
        "sub",
        "upvotes",
        "title",
        "body",
        "permalink",
        "sentiment",
    ]

    if not recs:
        return pd.DataFrame(columns=cols)

    posts = pd.DataFrame(recs)[cols].sort_values("created", ascending=False)

    # strong dtypes help later grouping / plotting
    posts["created"] = pd.to_datetime(posts["created"])
    posts["sentiment"] = posts["sentiment"].astype(float)

    return posts

# convenience combo fetch
@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_price_and_trend(sym, start, end):
    # 1) price – pad end by +1 day so the final date is included
    price_daily = get_price_df(sym, start, end + timedelta(days=1))["Close"]

    # 2) Reddit posts (may be empty)
    posts = fetch_reddit_df(sym, start, end)

    if posts.empty:
        # guarantee downstream columns so .groupby / UI never explode
        posts = pd.DataFrame(columns=["created", "sentiment", "day"])
        trend_df = pd.DataFrame(columns=["day", "sentiment"])
        return price_daily, trend_df, posts

    # 3) normal path – derive “day” and trend
    posts["day"] = posts["created"].dt.date
    trend_df = (
        posts.groupby("day")["sentiment"]
        .mean()
        .reset_index()
        .sort_values("day")
    )

    return price_daily, trend_df, posts
def calc_mini_forecast(trend_df: pd.DataFrame, price_daily: pd.Series, min_obs: int = 10) -> float | None:
    """
    Quick 1-step LinearRegression forecast of next-day % return from sentiment.

    Expects:
      trend_df['day'], trend_df['sentiment']
    price_daily: Series of prices (DatetimeIndex)

    Returns forecast as proportion (e.g., 0.012 = +1.2%) or None if insufficient data.
    """
    if trend_df.empty or len(price_daily) < min_obs:
        return None

    price_pct = price_daily.pct_change()
    tmp = trend_df.copy()
    tmp["pct_next"] = price_pct.shift(-1).reindex(tmp["day"]).values
    valid = tmp.dropna(subset=["pct_next"])
    if len(valid) < min_obs:
        return None

    from sklearn.linear_model import LinearRegression
    X = valid[["sentiment"]].values
    y = valid["pct_next"].values
    lr = LinearRegression().fit(X, y)
    # forecast uses *last available* sentiment in trend_df
    return float(lr.predict([[trend_df.iloc[-1]["sentiment"]]])[0])
def detect_divergences(
    price_daily: pd.Series,
    trend_df: pd.DataFrame,
    *,
    sent_jump=0.30,      # +30 % jump vs prev
    price_drop=-0.02,    # –2 %
    price_rise=0.02,     # +2 %
    z_cut=1.5,           # |z| threshold
    corr_win=7,
    corr_cut_neg=-0.3,
    corr_cut_pos=0.1,
) -> dict:
    """
    Run several quick anomaly / divergence heuristics on the **latest day**.

    Returns a dict with flag counts and diagnostics; safe on very small
    data windows (no broadcasting errors).
    """
    # ------------------------------------------------------------------
    # Guard‑rails – need at least 2 obs to compare today vs yesterday
    # ------------------------------------------------------------------
    res = {"neg_flags": 0, "pos_flags": 0, "rules": {}}
    if len(trend_df) < 2 or len(price_daily) < 2:
        return res

    # ------------------------------------------------------------------
    # Basic one‑day numbers & z‑score
    # ------------------------------------------------------------------
    sent_today = trend_df.iloc[-1]["sentiment"]
    sent_prev  = trend_df.iloc[-2]["sentiment"]

    price_today = price_daily.iloc[-1]
    price_prev  = price_daily.iloc[-2]
    pct_today   = (price_today - price_prev) / price_prev if price_prev else 0.0

    price_pct = price_daily.pct_change()
    roll_std  = price_pct.rolling(30).std().iloc[-1]
    z_score   = pct_today / roll_std if roll_std else 0.0

    # ------------------------------------------------------------------
    # 7‑day rolling correlation (sentiment ↔ return)
    #   • Trim both tails to equal length before masking NaNs
    # ------------------------------------------------------------------
    a_tail = trend_df["sentiment"].values[-corr_win:]
    b_tail = price_pct.values[-corr_win:]
    min_len = min(len(a_tail), len(b_tail))

    if min_len < corr_win:
        corr7 = np.nan          # not enough data yet
    else:
        a_tail = a_tail[-min_len:]
        b_tail = b_tail[-min_len:]
        mask   = ~np.isnan(a_tail) & ~np.isnan(b_tail)
        corr7  = (
            np.corrcoef(a_tail[mask], b_tail[mask])[0, 1]
            if mask.sum() >= corr_win
            else np.nan
        )

    # ------------------------------------------------------------------
    # Heuristic rule flags
    # ------------------------------------------------------------------
    # negative divergence flavours
    r_z_neg   = (z_score <= -abs(z_cut))
    r_sent_up = ((sent_today - sent_prev) / (abs(sent_prev) or 1) >= sent_jump)
    r_posvneg = (sent_today > 0.05 and pct_today < price_drop)
    r_corrneg = (not np.isnan(corr7) and corr7 < corr_cut_neg)

    # positive / early‑breakout flavours
    r_z_pos    = (z_score >= abs(z_cut))
    r_sent_mut = (sent_today < 0.05)
    r_sent_dn  = (sent_today < sent_prev and pct_today > price_rise)
    r_corrflat = (not np.isnan(corr7) and corr7 < corr_cut_pos)

    neg_flags = sum([r_z_neg, r_sent_up, r_posvneg, r_corrneg])
    pos_flags = sum([r_z_pos, r_sent_mut, r_sent_dn, r_corrflat])

    # ------------------------------------------------------------------
    # Package & return
    # ------------------------------------------------------------------
    res.update(
        sent_today=sent_today,
        sent_prev=sent_prev,
        pct_today=pct_today,
        z_score=z_score,
        corr7=corr7,
        neg_flags=neg_flags,
        pos_flags=pos_flags,
        rules=dict(
            z_score_neg=r_z_neg,
            sent_jump=r_sent_up,
            sent_vs_price=r_posvneg,
            corr_neg=r_corrneg,
            z_score_pos=r_z_pos,
            sent_muted=r_sent_mut,
            sent_dn_price_up=r_sent_dn,
            corr_flat=r_corrflat,
        ),
    )
    return res
def calc_kpis(price_daily: pd.Series, trend_df: pd.DataFrame, forecast: float | None = None):
    """
    Return dict with last price, day change %, last sentiment, vol_10, forecast (optional).
    `forecast` pass-through lets you feed calc_mini_forecast output.
    """
    price_today = price_daily.iloc[-1]
    price_prev  = price_daily.iloc[-2] if len(price_daily) > 1 else price_today
    delta_pct   = (price_today / price_prev - 1) if price_prev else 0.0
    sent_daily  = trend_df.iloc[-1]["sentiment"] if not trend_df.empty else np.nan
    vol_10      = price_daily.pct_change().rolling(10).std().iloc[-1]
    return dict(
        price_today=price_today,
        delta_pct=delta_pct,
        sent_daily=sent_daily,
        vol_10=vol_10,
        forecast=forecast,
    )
# ------------------------------------------------------------------
# GPT summary
# ------------------------------------------------------------------
def top_posts_hybrid(df: pd.DataFrame, k: int = 10, tau_hrs: float = 72) -> pd.DataFrame:
    """
    Rank Reddit posts by a blend of upvotes (scaled 0-1) and a recency decay.

    score = 0.6 * upvote_score + 0.4 * exp(-age_hours / tau_hrs)

    Parameters
    ----------
    df : DataFrame (needs columns: 'upvotes', 'created')
    k : int, top-k rows to return
    tau_hrs : float, decay scale in hours (bigger = slower decay)

    Returns
    -------
    DataFrame sorted by 'score_hybrid' desc, top-k.
    """
    if df.empty:
        return df

    # scale upvotes to 0..1 and clip negatives
    up_norm = df["upvotes"].clip(lower=0)
    max_up = up_norm.max()
    if max_up:
        up_norm = up_norm / max_up

    # recency weight
    now_naive = pd.Timestamp.utcnow().tz_localize(None)
    created_naive = pd.to_datetime(df["created"]).dt.tz_localize(None)
    age_hrs = (now_naive - created_naive).dt.total_seconds() / 3600.0
    rec_w = np.exp(-age_hrs / tau_hrs)

    out = df.copy()
    out["score_hybrid"] = 0.6 * up_norm + 0.4 * rec_w
    return out.sort_values("score_hybrid", ascending=False).head(k)

@st.cache_data(show_spinner=False, ttl=60*30)
def gpt_summary(sym: str, top_posts: pd.DataFrame) -> str:
    if not openai.api_key:
        return "⚠️ No OpenAI key provided."
    posts_text = "\n\n".join(
        [f"{i+1}. {r['title']} (▲{r['upvotes']}): {r['body'][:300]}"
         for i, r in top_posts.iterrows()]
    )
    prompt = (
        f"You are a financial analyst. Summarize market sentiment for {sym} "
        f"in 3–4 sentences (bull vs bear, themes, catalysts) based on Reddit posts:\n\n{posts_text}"
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
# Display helpers (moved from app)
# ------------------------------------------------------------------
def sentiment_color(v):
    if v > 0.05:  return "background-color:#d4f4dd"
    if v < -0.05: return "background-color:#f8d7da"
    return "background-color:#f0f0f0"

def show_table(df: pd.DataFrame):
    """
    Render the latest 20 Reddit posts in a small table.
    • Title is now a hyperlink (↗) to the original post, opening in a new tab.
    • Keeps existing sentiment‑colour shading.
    """
    if df.empty:
        st.info("No Reddit posts in the chosen window.")
        return

    # -------- 1) prep subset & cosmetic tweaks -----------------------------
    disp = (
        df[["created", "sub", "upvotes", "sentiment", "title", "permalink"]]
        .head(20)           # DataFrame is already sorted newest→oldest
        .copy()
    )

    # date as nice string
    disp["created"] = disp["created"].dt.strftime("%Y‑%m‑%d %H:%M")

    # clickable title with arrow icon
    disp["title"] = disp.apply(
        lambda r: (
            f"<a href='{r.permalink}' target='_blank' "
            f"style='text-decoration:none;'>{r.title} ↗</a>"
        ),
        axis=1,
    )

    # user‑friendly column names
    disp.rename(columns={
        "created":  "Date UTC",
        "sub":      "Sub",
        "upvotes":  "▲",
        "sentiment":"Sent",
        "title":    "Post (title → reddit)",
    }, inplace=True)

    # -------- 2) Styler with your sentiment shading ------------------------
    styled = (
        disp.drop(columns="permalink")            # hide raw URL
             .style
             .applymap(sentiment_color, subset=["Sent"])
             .format({"Sent": "{:.2f}"})
    )

    # -------- 3) render – allow HTML so the <a> links survive -------------
    html = styled.to_html(escape=False, index=False)   # <-- crucial
    st.markdown(
        f"<div class='mp-table'>{html}</div>",
        unsafe_allow_html=True,
    )

def show_dist(df: pd.DataFrame):
    fig = px.histogram(
        df, x="sentiment", nbins=20,
        title="Sentiment distribution (compound score)"
    ).update_layout(margin=dict(l=0,r=0,t=40,b=0))
    fig = style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

def show_trend(tr: pd.DataFrame):
    fig = px.line(
        tr, x="day", y="sentiment", markers=True,
        title="Daily average Reddit sentiment"
    ).update_traces(line_width=2).update_layout(margin=dict(l=0,r=0,t=40,b=0))
    fig = style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

def plot_price(df: pd.DataFrame, sym: str):
    fig = px.line(
        df, y="Close",
        title=f"{sym} — closing price",
        labels={"Close":"Price (USD)", "index":"Date"}
    )
    fig.update_traces(line_width=2)
    fig = style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)
# ------------------------------------------------------------------
# UI chrome
# ------------------------------------------------------------------
def sidebar_mode_toggle():
    """
    Render the “🧪 Pro mode” switch (once per session) **and**
    return True when Pro is enabled.
    """
    # first call: bootstrap session defaults
    st.session_state.setdefault("mode", "Basic")
    st.session_state.setdefault("pro_mode", False)

    st.sidebar.markdown("---")
    pro = st.sidebar.toggle(
        "🧪 Pro mode",
        key="pro_mode_toggle",                     # ← one key for every page
        value=st.session_state["pro_mode"],        # sync with session
        help="Show Alpha Lab, ML models, correlations, etc.",
    )

    # keep both aliases in‑sync for older code
    st.session_state["pro_mode"] = pro
    st.session_state["mode"]     = "Pro" if pro else "Basic"

    return pro

def mark_data_fetched():
    """Store UTC timestamp in session_state after a successful fetch."""
    st.session_state["_last_fetch_utc"] = datetime.now(timezone.utc)

def _render_data_freshness_chip(ph=None):
    """Render or update the data freshness indicator."""
    if ph is None:
        ph = st.empty()

    ts = st.session_state.get("_last_fetch_utc")

    ph.empty()  # clear whatever was there

    if ts is None:
        ph.caption("⚪ No data yet")
        return ph

    age_min = (datetime.now(timezone.utc) - ts).total_seconds() / 60
    if age_min < _FRESH_MAX_MIN:
        ph.success(f"🟢 Data fresh ({age_min:.0f}m)")
    else:
        ph.warning(f"🟡 Stale {age_min:.0f}m ago")
    return ph


def global_header(show_date_presets: bool = True):
    """
    Shared branded header row (all pages).

    Row 1  – Logo + data‑freshness chip
    Row 2  – Ticker, date‑range picker (+ optional preset buttons)
    """
    inject_global_css()  # font & spacing tweaks

    # ─────────────────────────────────── Row 1 ───────────────────────────────────
    col_l, col_r = st.columns([1, 5])
    with col_l:
        st.markdown("### 📈 **MarketPulse**")
    with col_r:
        ph = st.session_state.get("_freshness_ph") or st.empty()
        st.session_state["_freshness_ph"] = ph
        _render_data_freshness_chip(ph)

    st.markdown("")  # spacer

    # ─────────────────────────────────── Row 2 ───────────────────────────────────
    st.markdown(
        """
        <style>
          .mp-header{padding:0.25rem 0;border-bottom:1px solid #333;margin-bottom:0.75rem;}
        </style>""",
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="mp-header"></div>', unsafe_allow_html=True)
        col_tic, col_dates, col_presets = st.columns([1.2, 2.4, 2.4])

        # ── 1) Ticker ---------------------------------------------------------
        with col_tic:
            st.session_state.setdefault("ticker", "AAPL")
            tval = st.text_input(
                "Ticker",
                st.session_state["ticker"],
                help="Enter a Yahoo Finance symbol (e.g. AAPL, TSLA, NVDA, BTC‑USD).",
            )
            st.session_state["ticker"] = tval.upper().strip() or "AAPL"

        # ── 2) Date‑range picker --------------------------------------------
        with col_dates:
            default_start = st.session_state.get("default_start") or (
                date.today() - timedelta(days=60)
            )

            prev_start, prev_end = st.session_state.get(
                "date_range", (default_start, date.today())
            )
            current_range = tuple(map(_coerce_date, (prev_start, prev_end)))

            dr = st.date_input(
                "Date range",
                value=current_range,
                max_value=date.today(),
                help="Historical window for price, sentiment & back‑tests.",
            )

            # Streamlit emits *either* a single date *or* a (start,end) tuple
            if isinstance(dr, tuple) and len(dr) == 2:
                # user completed the range – store it
                start, end = map(_coerce_date, sorted(dr))
                st.session_state["date_range"] = (start, end)
                st.session_state.pop("_partial_start", None)
            else:
                # only the first click so far → remember it but DON’T overwrite the
                # existing full range (avoids errors + “snap back” UX quirk)
                st.session_state["_partial_start"] = _coerce_date(dr)

        # ── 3) Preset buttons (optional) -------------------------------------
        with col_presets:
            if show_date_presets:
                st.markdown("**Quick range**")
                b1, b2, b3, b4 = st.columns(4)
                today = date.today()
                if b1.button("3M"):
                    st.session_state["date_range"] = (today - timedelta(days=90), today)
                if b2.button("6M"):
                    st.session_state["date_range"] = (today - timedelta(days=180), today)
                if b3.button("1Y"):
                    st.session_state["date_range"] = (today - timedelta(days=365), today)
                if b4.button("Max"):
                    st.session_state["date_range"] = (today - timedelta(days=5 * 365), today)

# ==================================================================
# Cross-page convenience helpers
# ==================================================================

def current_selection():
    """
    Return (ticker, start_date, end_date) from session_state
    with sane defaults if user hasn't changed anything yet.
    """
    import datetime as _dt
    tic = st.session_state.get("ticker", "AAPL")
    start, end = st.session_state.get(
        "date_range",
        (_dt.date.today() - _dt.timedelta(days=60), _dt.date.today())
    )
    return tic, start, end


def ensure_data(sym: str, start, end):
    """
    Guarantee that `price_daily` + `trend_df` for `sym` exist in session_state.

    If missing, fetch using `fetch_price_and_trend()` and store:

        price_daily -> f"price_daily_{sym}"
        trend_df    -> f"trend_{sym}"
        posts_df    -> f"posts_{sym}"

    Returns (price_daily, trend_df, posts_df).
    """
    key_p = f"price_daily_{sym}"
    key_t = f"trend_{sym}"
    key_posts = f"posts_{sym}"

    if key_p not in st.session_state or key_t not in st.session_state:
        price, trend, posts = fetch_price_and_trend(sym, start, end)
        st.session_state[key_p] = price
        st.session_state[key_t] = trend
        st.session_state[key_posts] = posts
        mark_data_fetched()
    else:
        price  = st.session_state[key_p]
        trend  = st.session_state[key_t]
        posts  = st.session_state.get(key_posts)

    return price, trend, posts
