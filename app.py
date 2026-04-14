import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from newsapi import NewsApiClient
from sklearn.ensemble import IsolationForest, RandomForestRegressor

# Auto-refresh import 
from streamlit_autorefresh import st_autorefresh

load_dotenv()

# Constants      
TICKERS = ["SPY", "TLT", "GLD", "VNQ", "USO"]
TICKER_QUERIES = {
    "SPY": "S&P 500 stock market",
    "GLD": "Gold ETF commodity",
    "TLT": "US Treasury bonds TLT",
    "VNQ": "Real estate REIT VNQ",
    "USO": "crude oil USO ETF",
}

#Page config 
st.set_page_config(
    page_title="NSGA-Twin Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-refresh every 60 seconds
st_autorefresh(interval=60000, limit=None, key="data_refresh")

# Session state init 
st.session_state.setdefault("pareto_front", None)
st.session_state.setdefault("event_log", [])
st.session_state.setdefault("refresh_count", 0)
st.session_state.setdefault("last_nsga2_run", None)   # timestamp of last NSGA-II run
st.session_state.setdefault("run_nsga2_now", False)   # set True by the manual button
st.session_state.setdefault("sentiment_scores", None)     # cached after first FinBERT run
st.session_state.setdefault("article_sentiments", None)   # cached after first FinBERT run

# Increment refresh counter on every run
st.session_state["refresh_count"] += 1

# Title 
st.title("AI & NSGA-II Driven Portfolio Digital Twin")
st.markdown("### Real-Time Portfolio Management")
st.markdown("---")

@st.cache_data(ttl=60)
def load_market_data():
    """Fetch daily closing prices for the last 6 months from Yahoo Finance."""
    end = datetime.now()
    start = end - timedelta(days=180)
    data = yf.download(
        TICKERS,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
    )
    prices = data["Close"].dropna()
    return prices


@st.cache_data(ttl=60)
def load_latest_prices():
    """Fetch most recent intraday prices (1-min interval) for live feel."""
    try:
        data = yf.download(TICKERS, period="1d", interval="1m", progress=False)
        return data["Close"].dropna()
    except Exception:
        return None


@st.cache_resource
def load_finbert():
    """Load ProsusAI/finbert """
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
    )


@st.cache_data(ttl=600)
def fetch_news(ticker: str):
    """Fetch the 5 most recent news articles for a ticker via NewsAPI org."""
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key or api_key == "your_newsapi_key_here":
        return []
    try:
        newsapi = NewsApiClient(api_key=api_key)
        query = TICKER_QUERIES.get(ticker, ticker)
        response = newsapi.get_everything(
            q=query, language="en", sort_by="publishedAt", page_size=5
        )
        articles = response.get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "description": a.get("description", "") or "",
                "url": a.get("url", ""),
                "published": a.get("publishedAt", "")[:10],
                "source": a.get("source", {}).get("name", ""),
            }
            for a in articles
            if a.get("title")
        ]
    except Exception as e:
        return [{"title": f"Error fetching news: {e}", "description": "", "url": "", "published": "", "source": ""}]


def analyze_sentiment(texts: list, finbert) -> list:
    """Run FinBERT on a list of strings. Returns [{label, score}, ...]."""
    if not texts:
        return []
    truncated = [t[:512] for t in texts]
    return finbert(truncated)



def compute_features(prices_df, sentiment_scores=None):
    """
    Build an 8-feature DataFrame per asset (40 columns total).

    Features per asset:
      log_ret_1d, log_ret_5d, log_ret_20d, log_ret_60d,
      vol_20d, rsi_14, macd, sentiment
    """
    if sentiment_scores is None:
        sentiment_scores = {t: 0.0 for t in TICKERS}

    log_prices = np.log(prices_df)
    feature_frames = []

    for ticker in TICKERS:
        p = log_prices[ticker]

        # Log-returns
        lr1  = p.diff(1)
        lr5  = p.diff(5)
        lr20 = p.diff(20)
        lr60 = p.diff(60)

        # 20-day rolling volatility of 1-day returns
        vol20 = lr1.rolling(20).std()

        # RSI 14 using EWM (Wilder smoothing)
        delta = prices_df[ticker].diff()
        gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
        rs    = gain / (loss + 1e-9)
        rsi14 = 100 - (100 / (1 + rs))

        # MACD = EMA(12) - EMA(26)
        ema12 = prices_df[ticker].ewm(span=12, adjust=False).mean()
        ema26 = prices_df[ticker].ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26

        # Sentiment (constant per ticker, broadcast to all rows)
        sentiment = pd.Series(
            sentiment_scores.get(ticker, 0.0), index=prices_df.index
        )

        df_t = pd.DataFrame({
            f"{ticker}_log_ret_1d":  lr1,
            f"{ticker}_log_ret_5d":  lr5,
            f"{ticker}_log_ret_20d": lr20,
            f"{ticker}_log_ret_60d": lr60,
            f"{ticker}_vol_20d":     vol20,
            f"{ticker}_rsi14":       rsi14,
            f"{ticker}_macd":        macd,
            f"{ticker}_sentiment":   sentiment,
        })
        feature_frames.append(df_t)

    features_df = pd.concat(feature_frames, axis=1)
    features_df = features_df.dropna()
    return features_df


# RANDOM FOREST RETURN PREDICTION

def predict_returns(features_df, prices_df):
    """
    Walk-forward RF prediction of next-day 1-day log-return per asset.
    Train on all rows except the last 20, predict the last row.
    Returns dict {"SPY": 0.001, ...}.
    """
    predictions = {}

    for ticker in TICKERS:
        # Select this ticker's 8 feature columns
        cols = [c for c in features_df.columns if c.startswith(f"{ticker}_")]
        X = features_df[cols]

        # Target: next-day 1-day log-return (shift -1)
        log_ret = np.log(prices_df[ticker]).diff(1)
        y = log_ret.reindex(features_df.index).shift(-1)

        # Align and drop the last row (NaN target, which is what we predict)
        Xy = pd.concat([X, y.rename("target")], axis=1).dropna()

        if len(Xy) < 30:
            predictions[ticker] = 0.0
            continue

        X_train = Xy[cols].iloc[:-20]
        y_train = Xy["target"].iloc[:-20]
        X_pred  = X.iloc[[-1]]  # latest row from full feature set

        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        predictions[ticker] = float(model.predict(X_pred)[0])

    # Clip to realistic daily return bounds (max ±0.5% per day)
    predictions = {t: float(np.clip(v, -0.005, 0.005)) for t, v in predictions.items()}
    return predictions


# ═════════════════════════════════════════════════════════════════════════════
# TASK 4: ISOLATION FOREST ANOMALY DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def detect_anomaly(returns_df):
    """
    Uses Isolation Forest to detect anomalous market regimes.
    Features: 20-day rolling volatility + upper triangle of 60-day rolling corr.
    Returns True if last row is anomalous.
    """
    # 20-day rolling volatility per asset
    vol = returns_df.rolling(20).std().dropna()

    # Build correlation features: rolling 60-day corr upper triangle flattened
    n = len(TICKERS)
    corr_rows = []
    for i in range(len(vol)):
        window_start = max(0, i - 60 + 1)
        window = returns_df.iloc[window_start : i + 1]
        if len(window) < 10:
            corr_rows.append([np.nan] * (n * (n - 1) // 2))
            continue
        c = window.corr().values
        upper = c[np.triu_indices(n, k=1)]
        corr_rows.append(upper.tolist())

    corr_df = pd.DataFrame(
        corr_rows,
        index=vol.index,
    )

    # Align indices
    corr_df = corr_df.loc[vol.index].dropna()
    vol_aligned = vol.loc[corr_df.index]

    combined = pd.concat([vol_aligned.reset_index(drop=True),
                          corr_df.reset_index(drop=True)], axis=1).dropna()
    combined.columns = combined.columns.astype(str)

    if len(combined) < 20:
        return False

    # Train on all but the last row, predict on the last row
    clf = IsolationForest(contamination=0.05, random_state=42)
    clf.fit(combined.iloc[:-1])
    pred = clf.predict(combined.iloc[[-1]])
    return bool(pred[0] == -1)


# ═════════════════════════════════════════════════════════════════════════════
# TASK 5: NSGA-II FROM SCRATCH (numpy only)
# ═════════════════════════════════════════════════════════════════════════════

def _objectives(weights, expected_returns, cov_matrix):
    """Two objectives: (portfolio variance, -portfolio return)."""
    variance = float(weights @ cov_matrix @ weights)
    ret      = float(-expected_returns @ weights)  # negate → minimise
    return np.array([variance, ret])


def _fast_nondominated_sort(obj_vals):
    """
    Returns fronts: list of lists of indices.
    obj_vals shape: (pop_size, n_objectives), all minimised.
    """
    n = len(obj_vals)
    dom_count  = np.zeros(n, dtype=int)   # how many dominate individual i
    dom_set    = [[] for _ in range(n)]   # individuals dominated by i
    fronts     = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # i dominates j if i is no worse on all and strictly better on at least one
            diff = obj_vals[i] - obj_vals[j]
            if np.all(diff <= 0) and np.any(diff < 0):
                dom_set[i].append(j)
            elif np.all(diff >= 0) and np.any(diff > 0):
                dom_count[i] += 1

        if dom_count[i] == 0:
            fronts[0].append(i)

    current = 0
    while fronts[current]:
        next_front = []
        for i in fronts[current]:
            for j in dom_set[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    next_front.append(j)
        current += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def _crowding_distance(front, obj_vals):
    """Compute crowding distance for individuals in a front."""
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)

    dist = np.zeros(n)
    n_obj = obj_vals.shape[1]

    for m in range(n_obj):
        vals  = obj_vals[front, m]
        order = np.argsort(vals)
        dist[order[0]]  = np.inf
        dist[order[-1]] = np.inf
        rng = vals[order[-1]] - vals[order[0]]
        if rng == 0:
            continue
        for k in range(1, n - 1):
            dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / rng

    return dist


def _tournament(ranks, distances):
    """Binary tournament: prefer lower rank, then higher crowding distance."""
    n = len(ranks)
    a, b = np.random.choice(n, 2, replace=False)
    if ranks[a] < ranks[b]:
        return a
    elif ranks[b] < ranks[a]:
        return b
    else:
        return a if distances[a] > distances[b] else b


def _sbx_crossover(p1, p2, eta=20):
    """Simulated Binary Crossover on weight vectors."""
    n = len(p1)
    child1, child2 = p1.copy(), p2.copy()
    for i in range(n):
        u = np.random.rand()
        if u <= 0.5:
            beta = (2 * u) ** (1 / (eta + 1))
        else:
            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        child1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        child2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
    return child1, child2


def _polynomial_mutation(x, eta=20, prob=0.05):
    """Polynomial mutation per gene."""
    x = x.copy()
    for i in range(len(x)):
        if np.random.rand() < prob:
            u  = np.random.rand()
            if u < 0.5:
                delta = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            x[i] += delta
    return x


def _repair(x):
    """Clip weights to [0, 1] and normalise to sum = 1."""
    x = np.clip(x, 0, 1)
    s = x.sum()
    return x / s if s > 0 else np.ones(len(x)) / len(x)


def run_nsga2(expected_returns, cov_matrix, pop_size=100, n_gen=200,
              crossover_prob=0.8, mutation_prob=0.05):
    """
    NSGA-II portfolio optimisation.
    Objectives: minimise portfolio variance, maximise portfolio return.
    Returns a DataFrame of the first Pareto front.
    """
    n_assets = len(expected_returns)

    # Initialise population using Dirichlet (guarantees sum=1 & weights≥0)
    pop = np.random.dirichlet(np.ones(n_assets), size=pop_size)

    for _ in range(n_gen):
        # Evaluate objectives
        obj = np.array([_objectives(ind, expected_returns, cov_matrix) for ind in pop])

        # Non-dominated sort
        fronts = _fast_nondominated_sort(obj)
        ranks  = np.zeros(pop_size, dtype=int)
        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank

        # Crowding distance
        dists = np.zeros(pop_size)
        for front in fronts:
            cd = _crowding_distance(front, obj)
            for k, idx in enumerate(front):
                dists[idx] = cd[k]

        # Generate offspring
        offspring = []
        while len(offspring) < pop_size:
            a = _tournament(ranks, dists)
            b = _tournament(ranks, dists)
            if np.random.rand() < crossover_prob:
                c1, c2 = _sbx_crossover(pop[a], pop[b])
            else:
                c1, c2 = pop[a].copy(), pop[b].copy()
            c1 = _repair(_polynomial_mutation(c1, prob=mutation_prob))
            c2 = _repair(_polynomial_mutation(c2, prob=mutation_prob))
            offspring.extend([c1, c2])

        # Combine parent + offspring, then select best pop_size individuals
        combined     = np.vstack([pop, np.array(offspring[:pop_size])])
        obj_combined = np.array([_objectives(ind, expected_returns, cov_matrix) for ind in combined])

        fronts  = _fast_nondominated_sort(obj_combined)
        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend(front)
            else:
                # Fill remaining spots using crowding distance
                needed = pop_size - len(new_pop)
                cd     = _crowding_distance(front, obj_combined)
                sorted_front = [front[i] for i in np.argsort(-cd)]
                new_pop.extend(sorted_front[:needed])
                break

        pop = combined[new_pop]

    # Final evaluation — extract first Pareto front
    obj_final = np.array([_objectives(ind, expected_returns, cov_matrix) for ind in pop])
    fronts    = _fast_nondominated_sort(obj_final)
    first     = fronts[0]

    rows = []
    for idx in first:
        w    = pop[idx]
        var  = obj_final[idx, 0]
        ret  = -obj_final[idx, 1]        # un-negate
        rows.append({
            "Risk":            np.sqrt(var) * np.sqrt(252),
            "Expected Return": (1 + ret) ** 252 - 1,
            "w_SPY": w[TICKERS.index("SPY")],
            "w_TLT": w[TICKERS.index("TLT")],
            "w_GLD": w[TICKERS.index("GLD")],
            "w_VNQ": w[TICKERS.index("VNQ")],
            "w_USO": w[TICKERS.index("USO")],
        })

    result = pd.DataFrame(rows).sort_values("Risk").reset_index(drop=True)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# TASK 6: EVENT-DRIVEN TRIGGERS
# ═════════════════════════════════════════════════════════════════════════════

def check_triggers(returns_df, current_weights, optimal_weights, is_anomaly):
    """
    Returns (should_rerun: bool, reason: str).
    Triggers NSGA-II rerun if any condition is met.
    """
    reasons = []

    # 1. Volatility spike: 5-day vol > 1.5× 60-day vol for any asset
    vol5  = returns_df.rolling(5).std().iloc[-1]
    vol60 = returns_df.rolling(60).std().iloc[-1]
    if ((vol5 / (vol60 + 1e-9)) > 1.5).any():
        reasons.append("Volatility spike detected")

    # 2. Portfolio drift
    if np.linalg.norm(current_weights - optimal_weights) > 0.10:
        reasons.append("Portfolio drift > 10%")

    # 3. Anomaly flag
    if is_anomaly:
        reasons.append("Anomaly detected by Isolation Forest")

    # 4. Scheduled trigger is now handled by the time-based auto-run (see main flow)

    if reasons:
        return True, " | ".join(reasons)
    return False, ""


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Configuration")
    risk_profile = st.select_slider(
        "Risk Profile",
        options=["Conservative", "Moderate", "Aggressive"],
        value="Moderate",
    )

    st.markdown("---")
    st.subheader("NSGA-II Control")

    # Show when the optimiser last ran
    last_run = st.session_state["last_nsga2_run"]
    if last_run is not None:
        elapsed = (datetime.now() - last_run).seconds
        st.caption(f"Last run: {last_run.strftime('%H:%M:%S')} ({elapsed}s ago)")
    else:
        st.caption("Not yet run")

    # Manual trigger button
    if st.button("▶ Run NSGA-II now", use_container_width=True):
        st.session_state["run_nsga2_now"] = True

    st.markdown("---")
    st.subheader("Current Portfolio")
    for i, ticker in enumerate(TICKERS):
        w = st.session_state.get("current_weights", np.ones(len(TICKERS)) / len(TICKERS))
        pct = w[i] * 100
        safe_w = float(np.clip(w[i], 0.0, 1.0))
        st.progress(safe_w, text=f"{ticker}: {pct:.1f}%")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN FLOW — TASK 11: Wire everything together
# ═════════════════════════════════════════════════════════════════════════════

# ── Step 1: Load prices ───────────────────────────────────────────────────────
prices_df  = load_market_data()
returns_df = prices_df.pct_change().dropna()

SENTIMENT_CONFIG = {
    "positive": ("Positive", "success"),
    "neutral":  ("Neutral",  "info"),
    "negative": ("Negative", "error"),
}

api_key_present = (
    bool(os.getenv("NEWS_API_KEY", ""))
    and os.getenv("NEWS_API_KEY") != "your_newsapi_key_here"
)

# ── Step 3: Sentiment — loaded lazily into session state (runs only ONCE) ─────
# Use cached values if already computed; otherwise default to neutral (0.0)
# so the rest of the dashboard renders immediately without waiting for FinBERT.
if st.session_state["sentiment_scores"] is None:
    # First run or first refresh: use neutral defaults so the UI shows right away
    sentiment_scores   = {t: 0.0 for t in TICKERS}
    article_sentiments = {t: [] for t in TICKERS}

    if api_key_present:
        # Load FinBERT + run sentiment in a background-style spinner
        # This only blocks on the VERY FIRST run; subsequent runs use cache
        with st.spinner("Loading FinBERT sentiment model (first run only)…"):
            try:
                finbert = load_finbert()
                for ticker in TICKERS:
                    articles = fetch_news(ticker)
                    if articles:
                        texts = [
                            f"{a['title']}. {a['description']}" if a["description"] else a["title"]
                            for a in articles
                        ]
                        results = analyze_sentiment(texts, finbert)
                        article_sentiments[ticker] = list(zip(articles, results))
                        scores = []
                        for r in results:
                            lbl = r["label"].lower()
                            s   = r["score"]
                            scores.append(s if lbl == "positive" else -s if lbl == "negative" else 0.0)
                        sentiment_scores[ticker] = float(np.mean(scores)) if scores else 0.0
            except Exception as e:
                st.warning(f"Sentiment analysis unavailable: {e}")

    # Only persist to session once fully computed
    st.session_state["sentiment_scores"]   = sentiment_scores
    st.session_state["article_sentiments"] = article_sentiments
else:
    # Subsequent runs: use cached values instantly (no FinBERT reload)
    sentiment_scores   = st.session_state["sentiment_scores"]
    article_sentiments = st.session_state["article_sentiments"]

# ── Step 2: Compute features ──────────────────────────────────────────────────
features_df = compute_features(prices_df, sentiment_scores)

# ── Step 4: Predict returns ───────────────────────────────────────────────────
with st.spinner("Running Random Forest return predictions…"):
    predicted_returns = predict_returns(features_df, prices_df)

# ── Step 5: Detect anomaly ────────────────────────────────────────────────────
with st.spinner("Running Isolation Forest anomaly detection…"):
    is_anomaly = detect_anomaly(returns_df)

# ── Current weights (fixed starting point) ───────────────────────────────────
if "current_weights" not in st.session_state:
    st.session_state["current_weights"] = np.array([0.35, 0.25, 0.15, 0.10, 0.15])
current_weights = st.session_state["current_weights"]

# ── Step 6: Triggers → NSGA-II ───────────────────────────────────────────────
# Build expected returns & covariance from predictions + historical data
# Blend: 50% RF prediction + 50% historical mean (more stable)
hist_mean = returns_df.mean().values
rf_pred = np.array([predicted_returns[t] for t in TICKERS])
exp_ret_vec = 0.5 * rf_pred + 0.5 * hist_mean
cov_matrix  = returns_df.cov().values

# Determine optimal weights from existing Pareto front (or equal-weight fallback)
if st.session_state["pareto_front"] is not None:
    pf = st.session_state["pareto_front"]
    if risk_profile == "Conservative":
        opt_row = pf.iloc[0]
    elif risk_profile == "Aggressive":
        opt_row = pf.iloc[-1]
    else:
        opt_row = pf.iloc[len(pf) // 2]
    optimal_weights = np.array([opt_row[f"w_{t}"] for t in TICKERS])
else:
    optimal_weights = np.ones(len(TICKERS)) / len(TICKERS)

# ── NSGA-II: runs ONLY when the sidebar button is pressed ────────────────────
if st.session_state.get("run_nsga2_now", False):
    st.session_state["run_nsga2_now"] = False  # reset flag immediately
    with st.spinner("Running NSGA-II optimisation…"):
        # Reduced pop_size and n_gen for fast UI responsiveness (prevents auto-refresh timeout)
        pareto_df = run_nsga2(exp_ret_vec, cov_matrix, pop_size=40, n_gen=50)
    st.session_state["pareto_front"]  = pareto_df
    st.session_state["last_nsga2_run"] = datetime.now()
    st.session_state["event_log"].append({
        "time":  datetime.now().strftime("%H:%M:%S"),
        "event": "NSGA-II triggered: Manual trigger via sidebar button",
    })

# Use stored Pareto front, or a static equal-weight placeholder if never run
if st.session_state["pareto_front"] is not None:
    pareto_df = st.session_state["pareto_front"]
else:
    # Placeholder so the rest of the UI renders without running NSGA-II
    eq = np.ones(len(TICKERS)) / len(TICKERS)
    var = float(eq @ cov_matrix @ eq)
    ret = float(exp_ret_vec @ eq)
    pareto_df = pd.DataFrame([{
        "Risk": np.sqrt(var) * np.sqrt(252),
        "Expected Return": (1 + ret) ** 252 - 1,
        **{f"w_{t}": eq[i] for i, t in enumerate(TICKERS)},
    }])

# Log anomaly detection
if is_anomaly:
    st.session_state["event_log"].append({
        "time":  datetime.now().strftime("%H:%M:%S"),
        "event": "Anomaly detected by Isolation Forest",
    })

# Pick optimal portfolio from Pareto front based on risk profile
if risk_profile == "Conservative":
    opt_row = pareto_df.iloc[0]
elif risk_profile == "Aggressive":
    opt_row = pareto_df.iloc[-1]
else:
    opt_row = pareto_df.iloc[len(pareto_df) // 2]

optimal_weights = np.array([opt_row[f"w_{t}"] for t in TICKERS])
current_risk    = float(opt_row["Risk"])
current_return  = float(opt_row["Expected Return"])

# Log rebalancing recommendation
rebalancing_needed = st.session_state.get("run_nsga2_now", False) or is_anomaly
if rebalancing_needed:
    st.session_state["event_log"].append({
        "time":  datetime.now().strftime("%H:%M:%S"),
        "event": f"Rebalancing recommended. Risk profile: {risk_profile}",
    })

# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

# ── Latest price metrics ──────────────────────────────────────────────────────
st.subheader("Latest Asset Prices")
latest = prices_df.iloc[-1]
prev   = prices_df.iloc[-2] if len(prices_df) > 1 else latest
cols   = st.columns(len(TICKERS))
for i, ticker in enumerate(TICKERS):
    if ticker in latest:
        current_price = latest[ticker]
        prev_price    = prev[ticker] if ticker in prev else current_price
        delta         = current_price - prev_price
        cols[i].metric(label=ticker, value=f"${current_price:.2f}", delta=f"${delta:.2f}")

st.markdown("---")

# ── TASK 8: KPI Cards ─────────────────────────────────────────────────────────
st.subheader("Portfolio KPIs")

w = optimal_weights

# Annualised expected return (using predicted daily log-returns × 252)
daily_return = float(exp_ret_vec @ w)
ann_return = (1 + daily_return) ** 252 - 1  # proper compounding

# Annualised volatility
port_var      = float(w @ cov_matrix @ w)
ann_vol       = np.sqrt(port_var * 252)

# Sharpe ratio (risk-free rate ≈ 0 for simplicity)
sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

# VaR 95% (parametric, daily)
daily_mu    = float(exp_ret_vec @ w)
daily_sigma = np.sqrt(port_var)
var_95      = -(daily_mu - 1.645 * daily_sigma)

# Max drawdown — equal-weight portfolio over the full data period
eq_weights    = np.ones(len(TICKERS)) / len(TICKERS)
port_returns  = (returns_df @ eq_weights)
cum_returns   = (1 + port_returns).cumprod()
roll_max      = cum_returns.cummax()
drawdown      = (cum_returns - roll_max) / roll_max
max_drawdown  = float(drawdown.min())

kpi_cols = st.columns(5)
kpi_cols[0].metric("Ann. Return",   f"{ann_return*100:.2f}%")
kpi_cols[1].metric("Ann. Volatility", f"{ann_vol*100:.2f}%")
kpi_cols[2].metric("Sharpe Ratio",  f"{sharpe:.2f}")
kpi_cols[3].metric("VaR 95% (daily)", f"{var_95*100:.2f}%")
kpi_cols[4].metric("Max Drawdown",  f"{max_drawdown*100:.2f}%")

st.markdown("---")

# ── Price Chart + TASK 7: Correlation Heatmap ────────────────────────────────
col_market, col_heatmap = st.columns(2)

with col_market:
    st.subheader("Market Monitor")
    try:
        fig_prices = px.line(
            prices_df,
            title="Asset Prices over Time",
            labels={"value": "Price", "index": "Date"},
        )
        fig_prices.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_prices, use_container_width=True)
    except Exception as e:
        import traceback
        st.error(f"Error drawing market monitor: {e}")
        st.text(traceback.format_exc())

with col_heatmap:
    st.subheader("60-Day Rolling Correlation")
    try:
        corr_60 = returns_df.tail(60).corr()
        fig_corr = px.imshow(
            corr_60,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            title="60-Day Correlation Heatmap",
        )
        fig_corr.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        import traceback
        st.error(f"Error drawing heatmap: {e}")
        st.text(traceback.format_exc())

st.markdown("---")

# ── TASK 9: Aggregate Sentiment Gauges ───────────────────────────────────────
st.subheader("Aggregate Sentiment per Asset")
sent_cols = st.columns(5)
for i, ticker in enumerate(TICKERS):
    score = sentiment_scores[ticker]
    if score > 0.1:
        indicator = "🟢 Bullish"
        delta_str = f"+{score:.3f}"
    elif score < -0.1:
        indicator = "🔴 Bearish"
        delta_str = f"{score:.3f}"
    else:
        indicator = "🟡 Neutral"
        delta_str = f"{score:.3f}"
    sent_cols[i].metric(
        label=f"{ticker} Sentiment",
        value=indicator,
        delta=delta_str,
    )

st.markdown("---")

# ── NSGA-II Pareto Front ──────────────────────────────────────────────────────
st.subheader("NSGA-II Pareto Front")
try:
    fig_pareto = px.scatter(
        pareto_df, x="Risk", y="Expected Return",
        title="Multi-Objective Optimisation (Pareto Front)",
        hover_data=["w_SPY", "w_TLT", "w_GLD", "w_VNQ", "w_USO"],
    )

    # Current portfolio (hardcoded baseline)
    daily_port_var = float(current_weights @ cov_matrix @ current_weights)
    daily_port_ret = float(exp_ret_vec @ current_weights)
    fig_pareto.add_trace(go.Scatter(
        x=[np.sqrt(np.clip(daily_port_var, 0, None)) * np.sqrt(252)],
        y=[(1 + daily_port_ret) ** 252 - 1],
        mode="markers",
        marker=dict(color="red", size=15, symbol="star"),
        name="Current Portfolio",
    ))

    # Target portfolio from risk profile
    fig_pareto.add_trace(go.Scatter(
        x=[opt_row["Risk"]], y=[opt_row["Expected Return"]],
        mode="markers",
        marker=dict(color="lime", size=15, symbol="star"),
        name=f"Optimal Target ({risk_profile})",
    ))

    fig_pareto.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_pareto, use_container_width=True)
except Exception as e:
    import traceback
    st.error(f"Error drawing Pareto front: {e}")
    st.text(traceback.format_exc())

st.markdown("---")

# ── AI Analysis + Rebalancing ─────────────────────────────────────────────────
col_ai, col_rebalance = st.columns(2)

with col_ai:
    st.subheader("AI Layer Analysis")
    st.markdown("**Isolation Forest Anomaly Status:**")
    if is_anomaly:
        st.error("**Anomaly Detected!** High volatility regime. Emergency Rebalancing Triggered.")
    else:
        st.success("**Normal Market Regime.** No anomalies detected.")

    st.markdown("**Random Forest Return Predictions (Next 1D Log-Return):**")
    pred_data = {
        "Asset":            TICKERS,
        "Predicted Return": [f"{predicted_returns[t]*100:.4f}%" for t in TICKERS],
    }
    st.dataframe(pd.DataFrame(pred_data), use_container_width=True)

with col_rebalance:
    st.subheader("Rebalancing Action")
    status_label = "Needs Rebalancing" if rebalancing_needed else "Optimal"
    st.markdown(f"**Current Status:** {status_label}")

    weights_data = {
        "Asset":          TICKERS,
        "Current Weight": [round(float(current_weights[TICKERS.index(t)]), 4) for t in TICKERS],
        "Target Weight":  [round(float(opt_row[f"w_{t}"]), 4) for t in TICKERS],
    }
    df_weights        = pd.DataFrame(weights_data)
    df_weights["Delta"] = df_weights["Target Weight"] - df_weights["Current Weight"]
    df_weights["Delta"] = df_weights["Delta"].round(4)

    # Use native Streamlit column mapping, avoids PyArrow styler bugs
    st.dataframe(
        df_weights,
        use_container_width=True,
    )

    if st.button("Execute Trades (Update Physical Twin)", use_container_width=True):
        st.session_state["current_weights"] = optimal_weights.copy()
        st.session_state["event_log"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "event": f"Trades executed. Portfolio rebalanced to {risk_profile} profile. New weights: " +
                     ", ".join([f"{t}={optimal_weights[i]:.1%}" for i, t in enumerate(TICKERS)])
        })
        st.success("Portfolio updated!")
        st.rerun()

st.markdown("---")

# ── News & FinBERT Sentiment Section ─────────────────────────────────────────
st.subheader("Financial News & FinBERT Sentiment")

if not api_key_present:
    st.warning("pb de clef api")
else:
    tabs = st.tabs(TICKERS)
    for tab, ticker in zip(tabs, TICKERS):
        with tab:
            article_data = article_sentiments.get(ticker, [])
            if not article_data:
                st.info("No articles found for this asset.")
            else:
                for article, sentiment in article_data:
                    label_raw  = sentiment["label"].lower()
                    score      = sentiment["score"]
                    label_text, badge_type = SENTIMENT_CONFIG.get(label_raw, ("Neutral", "info"))
                    with st.container():
                        col_text, col_badge = st.columns([5, 1])
                        with col_text:
                            st.markdown(
                                f"**{article['title']}**  \n"
                                f"<small>{article['published']} &nbsp;|&nbsp; {article['source']}</small>",
                                unsafe_allow_html=True,
                            )
                            if article["description"]:
                                desc = article["description"]
                                st.caption(desc[:200] + "…" if len(desc) > 200 else desc)
                            if article["url"]:
                                st.markdown(f"[Read more ↗]({article['url']})")
                        with col_badge:
                            getattr(st, badge_type)(f"{label_text}  \n`{score:.0%}`")
                    st.divider()

st.markdown("---")

# ── TASK 10: Event Log ────────────────────────────────────────────────────────
st.subheader("Event Log")

if st.session_state["event_log"]:
    event_df = pd.DataFrame(st.session_state["event_log"])
    # Show last 10 events (most recent first)
    st.dataframe(event_df.tail(10).iloc[::-1].reset_index(drop=True), use_container_width=True)
else:
    st.info("No events logged yet.")
