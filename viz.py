"""
viz.py – reusable Plotly visual components for MarketPulse
----------------------------------------------------------

Contents
--------
1. anim_price_sentiment   : animated 60‑day "replay" of price + sentiment
2. sparkline helpers       : tiny charts for watch‑lists / tables
   • sparkline_fig
   • sparkline_row
   • render_sparkline_grid
3. price_line_by_sentiment : line coloured by bullish/bearish sentiment

All functions return a fully‑styled Plotly Figure using shared.style_plotly().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

import shared  # for style_plotly


# ----------------------------------------------------------------------
# 1. Animated price‑vs‑sentiment play‑through (last N trading days)
# ----------------------------------------------------------------------

def anim_price_sentiment(price_daily, trend_df, lookback=60):
    """
    Animated replay of the last *lookback* trading days.
    • Blue price line is revealed one day at a time
    • Green bars show cumulative sentiment on a secondary axis
    """
    # -------- 1) slice the last N sessions --------
    pxs = price_daily[-lookback:]                 # Series (DatetimeIndex)
    dates = pxs.index.to_list()

    # -------- 2) align sentiment by DATE only -----
    sent_series = (
        trend_df.set_index("day")["sentiment"]    # index = python date
        .reindex(pxs.index.date)                  # match by date
        .ffill()                                  # carry last known value
        .fillna(0)                                # leading gaps → neutral
    )
    # put the Timestamp index back so sent & px share the same index
    sent_series.index = pxs.index

    # -------- 3) base figure with two starter traces --------
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[dates[0]],
            y=[pxs.iloc[0]],
            mode="lines",
            name="Price",
            line=dict(color="#00b7ff", width=2),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[dates[0]],
            y=[sent_series.iloc[0]],
            name="Sentiment",
            marker=dict(color="#16c172"),
            yaxis="y2",
            opacity=0.6,
        )
    )

    # -------- 4) animation frames --------
    frames = [
        go.Frame(
            name=str(i),
            data=[
                go.Scatter(x=dates[: i + 1], y=pxs.iloc[: i + 1]),
                go.Bar(x=dates[: i + 1], y=sent_series.iloc[: i + 1]),
            ],
            traces=[0, 1],
        )
        for i in range(len(dates))
    ]
    fig.frames = frames

    # -------- 5) slider + play/pause controls -----
    steps = [
        {
            "args": [[str(k)], {"frame": {"duration": 0, "redraw": True},
                                "mode": "immediate"}],
            "label": dates[k].strftime("%Y‑%m‑%d"),
            "method": "animate",
        }
        for k in range(len(frames))
    ]

    fig.update_layout(
        title="Animated Price + Sentiment (last 60 d)",
        xaxis=dict(domain=[0, 0.9]),
        yaxis=dict(title="Price", side="left"),
        yaxis2=dict(
            title="Sentiment",
            side="right",
            overlaying="y",
            range=[-1, 1],
        ),
        sliders=[
            {
                "steps": steps,
                "active": len(steps) - 1,
                "pad": {"b": 0},
                "len": 0.9,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 1.02,
                "y": 1,
                "buttons": [
                    {
                        "label": "▶",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 120, "redraw": False},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "⏸",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        margin=dict(l=40, r=40, t=60, b=40),
    )
    # reuse your shared dark‑theme helper
    return shared.style_plotly(fig)

# ----------------------------------------------------------------------
# 2. Sparkline helpers (watch‑list mini‑charts)
# ----------------------------------------------------------------------
def sparkline_fig(series: pd.Series, color: str = "#00b7ff", height: int = 60):
    """
    Tiny %‑change line chart with no axes.
    """
    s = series.dropna()
    if s.empty:
        return go.Figure()
    y_rel = (s / s.iloc[0]) - 1  # pct‑from‑start
    fig = go.Figure(
        go.Scatter(
            x=y_rel.index,
            y=y_rel.values,
            mode="lines",
            line=dict(color=color, width=1.5),
            showlegend=False,
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def sparkline_row(sym: str, price_daily: pd.Series, trend_df: pd.DataFrame):
    """
    Builds a dictionary of quick metrics + sparkline figure.
    Designed to feed render_sparkline_grid().
    """
    last_px = price_daily.iloc[-1]
    day_ret = price_daily.pct_change().iloc[-1]
    sent = (
        trend_df.set_index("day")["sentiment"].iloc[-1]
        if not trend_df.empty
        else np.nan
    )
    fig = sparkline_fig(price_daily)
    return dict(
        Symbol=sym,
        Last=f"${last_px:,.2f}",
        Chg=f"{day_ret:+.1%}",
        Sent=f"{sent:+.2f}",
        Fig=fig,
    )


def render_sparkline_grid(rows: list[dict], cols_per_row: int = 4):
    """
    Renders a responsive grid of sparkline cards using st.columns().
    """
    import streamlit as st

    for chunk_start in range(0, len(rows), cols_per_row):
        cols = st.columns(cols_per_row)
        for c, row in zip(cols, rows[chunk_start : chunk_start + cols_per_row]):
            with c:
                st.markdown(
                    f"**{row['Symbol']}** &nbsp; {row['Chg']} &nbsp; ({row['Sent']})"
                )
                st.plotly_chart(row["Fig"], use_container_width=True)


# ----------------------------------------------------------------------
# 3. Price line coloured by sentiment gradient
# ----------------------------------------------------------------------
def price_line_by_sentiment(
    price_daily: pd.Series,
    trend_df: pd.DataFrame,
    cmap: str = "RdYlGn",
    sent_clip: float = 1.0,
):
    """
    Draws a segmented price line whose colour encodes daily sentiment.
    Green → bullish, Red → bearish (default 'RdYlGn' scale).

    Parameters
    ----------
    price_daily : Series (date index)
    trend_df    : DataFrame with ['day', 'sentiment']
    cmap        : Plotly colour‑scale name
    sent_clip   : clip sentiment to ±sent_clip before colour mapping
    """
    df = (
        pd.DataFrame({"price": price_daily})
        .join(trend_df.set_index("day")["sentiment"], how="left")
        .fillna(method="ffill")
    )
    df["sent_clipped"] = df["sentiment"].clip(-sent_clip, sent_clip)

    # ----- fixed: ensure no NaN passes to colour scale -----
    norm = (df["sent_clipped"] + sent_clip) / (2 * sent_clip)
    norm = norm.fillna(0)           # treat remaining gaps as neutral
    # -------------------------------------------------------

    fig = go.Figure()
    xs, ys, cs = df.index.to_list(), df["price"].to_list(), norm.to_list()

    for i in range(len(xs) - 1):
        colour = sample_colorscale(cmap, cs[i])[0]
        fig.add_trace(
            go.Scatter(
                x=[xs[i], xs[i + 1]],
                y=[ys[i], ys[i + 1]],
                mode="lines",
                line=dict(color=colour, width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # invisible 0‑size markers to attach a colour‑bar
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(
                size=0.1,
                color=df["sent_clipped"],
                colorscale=cmap,
                cmin=-sent_clip,
                cmax=sent_clip,
                colorbar=dict(title="Sentiment", ticks="outside"),
            ),
            showlegend=False,
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>Price=%{y:$,.2f}"
                "<br>Sent=%{marker.color:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Price coloured by Sentiment",
        xaxis_title="Date",
        yaxis_title="Price",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return shared.style_plotly(fig)
