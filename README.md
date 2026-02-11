
# MarketPulse
https://marketpulse-ai.streamlit.app/

> **Real‑time market radar that fuses price action with crowd sentiment to surface actionable divergences and lets you back‑test ideas in seconds – all from a single Streamlit interface.**  
> Fetch, visualise, simulate – ship an insight before the bell rings.

---

## Table of Contents
1. [Why MarketPulse?](#why-marketpulse)  
2. [Feature Tour](#feature-tour)  
3. [Model Architectures & Research](#model-architectures--research)  
4. [Metrics & AB Testing](#metrics--ab-testing)  
5. [Road‑map](#road-map)   
6. [License](#license)  

---

## Why MarketPulse?
* **Retail chatter moves markets** – small‑cap squeezes and crypto pumps are routinely catalysed by Reddit & X/Twitter minutes before the tape reacts.  
* **Sentiment ≠ statistics** – a dull score alone misses context, sarcasm, and velocity. Traders need a *narrative lens* not a numeric dump.  
* **Zero‑setup workflow** – spreadsheets & notebooks slow ideation. A single‑page app that fetches, visualises **and** simulates strategies removes all friction between a hunch and a validated trade‑plan.  

---

## Feature Tour

### Core Data‑Pipeline

| Stage             | Tech                                                                 | Notes                                                                                                                 |
| ----------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Prices            | `yfinance` wrapper                                                   | Daily *adj‑close* series for any Yahoo Finance ticker.                                                                |
| Reddit Ingestion  | `praw` search API                                                    | Scans eight finance‑heavy subs (`stocks`, `wallstreetbets`, …) with duplicate‑ID de‑duplication and 1‑year look‑back. |
| Sentiment Scoring | **RoBERTa‑base‑sentiment‑latest** (Cardiff NLP) + **VADER** fallback | `pos – neg` probability spread stored per post; model exceptions roll over to VADER `compound` for robustness.        |
| Persistence       | `@st.cache_data`                                                     | Hash‑keyed on (ticker, date‑range) so warm reloads hit sub‑second even on Streamlit Cloud.                            |

### Home Dashboard (Basic Mode)

* One‑click **Fetch** → downloads price + Reddit, updates freshness chip.  
* KPI strip — last‑price Δ %, aggregated daily sentiment, 24‑h linear‑reg forecast, 10‑day σ.  
* Mini Forecast — `sentiment[t‑1] → return[t]` OLS retrained on every click; displayed as probability.  
* Divergence Detector — seven heuristic rules (z‑score spikes, sentiment jumps, sent‑price ρ reversal…) with per‑rule diagnostics.  
* Interactive Charts — price & sentiment overlay, sentiment‑gradient price line, 60‑day animated replay.  
* Live Reddit Table — hyper‑linked titles, sentiment shading, up‑vote counts.  
* Distribution + LLM Summary — histogram of sentiment and a 3‑sentence GPT‑3.5 synopsis of the last 72 h discussion.  

### Divergence & Anomaly Detection — formal rules

| Rule | Condition (inline math) | Intuition |
|------|-------------------------|-----------|
| **Price *z‑score*** | $z_t = \dfrac{r_t}{\sigma_{30}},\;\; \lvert z_t \rvert \ge 1.5$ | Statistically large daily move. |
| **Sentiment jump** | $\Delta s_t = \dfrac{s_t - s_{t-1}}{\lvert s_{t-1} \rvert + \varepsilon} \;>\; 0.30$ | Crowd mood swings sharply. |
| **Bullish sent / Price ↓** | $s_t \;>\; 0.05 \;\land\; r_t \;<\; -0.02$ | Crowd bullish while tape dumps. |
| **7‑day negative ρ** | $\rho_{7}(s,r) \;<\; -0.30$ | Sentiment moves opposite to price. |
| **Sentiment muted** | $s_t \;<\; 0.05$ | Little chatter despite price action. |
| **Sent ↓ / Price ↑** | $s_t \;<\; s_{t-1} \;\land\; r_t \;>\; 0.02$ | Bearish talk while price pops. |
| **Flat ρ** | $\lvert \rho_{7}(s,r) \rvert \;<\; 0.10$ | Sentiment & price temporarily decoupled. |

*Flat‑ρ marks a “decoupling” regime where narrative no longer tracks tape and breakouts often incubate.*

### Alpha Lab (Pro Mode)

| Block                       | Description                                                                                                       |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Manual Rule Back‑tester | 10+ guards (sent, price drop, vol spike, RSI, MACD, Bollinger %B, momentum, stop/TP, transaction‑costs, initial capital). |
| Hyper‑parameter Grid | Exhaustive sweep across 4–6 parameters with live progress bar; Sharpe‑sorted results table. |
| Model Strategies | Logistic (calibrated), ARIMA(+sent), Prophet(+sent) – each with walk‑forward CV (5 folds) or expanding retrain. |
| Trade Log | Ag‑Grid with sortable columns, filter panel, and P/L summary metrics. |
| Equity Curves | Strategy vs buy‑&‑hold vs random baseline. |

### Correlations (Pro Mode)
* Cross‑asset ρ matrix (price returns or strategy equity curves).  
* Threshold‑highlight heat‑map, CSV/PNG export.  
* Rolling‑window explorer with pair selector and dual‑axis plot.  

### Benchmarks (Pro Mode)
* Batch‑run the manual‑rule strategy across any ticker list.  
* Outputs Sharpe, MaxDD %, TotalRet %, End‑equity USD, sparkline grid.  
* Signature‑based result caching so duplicate runs never stall the worker.  

### Visualisation Helper Library
* 60‑day price‑and‑sentiment animation.  
* Tiny SVG sparklines for watch‑lists.  
* Sentiment‑gradient price‑line component.  

### Re‑usable UI Chrome
* Global header, Poppins font, KPI card skin.  
* Toggleable Pro Mode state persisted in `st.session_state`.  
* Keyboard shortcut **Ctrl/Cmd + Enter** triggers the primary button on every page.  


---

## Model Architectures & Research

### 1. Logistic‑Regression (Classifier)

| Aspect | Detail |
|--------|--------|
| Purpose | Fast probabilistic estimate that next‑day return > 0. Drives KPI & Alpha Lab rules. |
| Features | 13 engineered inputs: raw & lagged sentiment, Δsent, moving averages, momentum, volatility, sentiment×momentum, RSI‑14, MACD‑hist. |
| Pipeline | `Imputer → Scaler → LogisticRegression(lbfgs)` inside `sklearn.Pipeline`. |
| Grid Search | `C ∈ {0.1,…,10}`, `class_weight ∈ {None,'balanced'}`; 3‑fold AUC. |
| Calibration | Optional isotonic via `CalibratedClassifierCV` for monotone P(up). |
| Explainability | SHAP values on demand (≤ 500 rows). |
| Inference | `predict_proba` vectorised; model hash prevents redundant re‑fits. |

### 2. ARIMA (p,d,q) + Sentiment Regressor
| Aspect | Detail |
|--------|--------|
| Objective | 1‑day point forecast of return. |
| Grid | (p,d,q) ∈ {(1,0,0),(2,0,0),(1,0,1),(2,0,1),(0,0,1),(0,1,1),(1,1,0)} |
| Runtime Cap | 1 000 rows; parallel fit with `joblib`. |
| Selection | Minimum AIC retained. |
| Signal | Trade if µ̂ > `ret_thresh` (default 2 %). |

### 3. Prophet (+ Sentiment Regressor)
| Aspect | Detail |
|--------|--------|
| Role | Weekly & yearly seasonality plus sentiment shocks for multi‑day forecasting. |
| History | Trimmed to ≤ 1 200 rows (~5 years). |
| Regressor | `add_regressor("sentiment")`, multiplicative mode. |
| Seasonalities | Weekly (order 3), yearly (order 10); `changepoint_range = 0.9`. |
| Walk‑forward | Expanding retrain; first forecast after warm‑up = 120 days. |

### 4. LSTM & Transformer (Road‑map)
* **Input** — 64‑step look‑back of [return %, sentiment, TA‑features].  
* **Architecture** — two LSTM layers (128 units) → Dense 64 → output 1.  
* **Quantisation** — post‑training INT8 dynamic (< 200 kB).  
* **Cross‑Asset Transformer** — Informer/TimesNet hybrid, shared embedding, 5‑day horizon.  
* **Online inference** — future Streamlit WebSocket (< 2 ms on‑GPU).  

*Why no LightGBM?*  Tree ensembles were evaluated but excluded because the risk engine requires closed‑form variance estimates.

---

## Metrics & A/B Testing

| Metric                    | Description                                     | Collection             |
|---------------------------|-------------------------------------------------|------------------------|
| DAU / WAU                 | Active users daily / weekly.                    | Front‑end ping.        |
| Fetch Success %           | Successful Reddit+price fetches ÷ attempts.     | Back‑end logs.         |
| Time‑to‑Insight           | Fetch click → divergence flag median seconds.   | JavaScript timer.      |
| Back‑test Runs per User   | Engagement proxy for Alpha Lab.                 | Session cache.         |
| Model P&L                 | Out‑of‑sample equity vs buy‑and‑hold.           | Alpha Lab logs.        |

### Current A/B Plan
1. **Hypothesis** — sentiment‑gradient price line surfaces divergences faster than plain overlay.  
2. **Split** — 50 % control (plain), 50 % test (gradient).  
3. **Success** — time‑to‑insight decreases by ≥ 15 % (α = 0.05).  
4. **Run Length** — at least 500 fetch sessions per cohort.  
5. **Analysis** — χ² on conversion; statistical power ≥ 0.8 (instrumentation via PostHog).  

---

## Road‑map

| Status | Item |
|--------|------|
| WIP | LSTM / Seq2Seq forecaster – sentiment+TA, quantised, live alpha stream. |
| WIP | Informer/TimesNet cross‑asset model – 5‑day horizon, multi‑ticker embedding. |
| Planned | Plugin ingestion for X (Twitter) & StockTwits sentiment. |
| Planned | Docker image and GitHub Actions CI (ruff, pytest, mypy, playwright tests). |
| Planned | User authentication & billing (Streamlit Auth + Stripe). |
| Planned | WebSocket live price bars with push divergence alerts. |
| Planned | Mobile‑responsive redesign (CSS Grid, swipeable KPI cards). |


---

## License
Released under the [MIT License](LICENSE).  Data courtesy of Yahoo Finance and the Reddit API under their respective terms of service. *No investment advice expressed or implied.*

---

*Made with caffeine, Plotly, and late‑night back‑tests.*


