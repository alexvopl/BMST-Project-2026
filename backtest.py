import os
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from dotenv import load_dotenv
from newsapi import NewsApiClient
from sklearn.ensemble import RandomForestRegressor
from transformers import pipeline

load_dotenv()

TICKERS = ["SPY", "TLT", "GLD", "VNQ", "USO"]
TICKER_QUERIES = {
    "SPY": "S&P 500 stock market",
    "GLD": "Gold ETF commodity",
    "TLT": "US Treasury bonds TLT",
    "VNQ": "Real estate REIT VNQ",
    "USO": "crude oil USO ETF",
}

# ═════════════════════════════════════════════════════════════════════════════
# FUNCTIONS FROM APP.PY
# ═════════════════════════════════════════════════════════════════════════════

def compute_features(prices_df, sentiment_scores=None):
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

def predict_returns(features_df, prices_df):
    predictions = {}
    for ticker in TICKERS:
        cols = [c for c in features_df.columns if c.startswith(f"{ticker}_")]
        X = features_df[cols]

        log_ret = np.log(prices_df[ticker]).diff(1)
        y = log_ret.reindex(features_df.index).shift(-1)

        Xy = pd.concat([X, y.rename("target")], axis=1).dropna()

        if len(Xy) < 30:
            predictions[ticker] = 0.0
            continue

        X_train = Xy[cols].iloc[:-20]
        y_train = Xy["target"].iloc[:-20]
        X_pred  = X.iloc[[-1]]

        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        predictions[ticker] = float(model.predict(X_pred)[0])

    predictions = {t: float(np.clip(v, -0.005, 0.005)) for t, v in predictions.items()}
    return predictions

def _objectives(weights, expected_returns, cov_matrix):
    variance = float(weights @ cov_matrix @ weights)
    ret      = float(-expected_returns @ weights)
    return np.array([variance, ret])

def _fast_nondominated_sort(obj_vals):
    n = len(obj_vals)
    dom_count  = np.zeros(n, dtype=int)
    dom_set    = [[] for _ in range(n)]
    fronts     = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
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
    n = len(ranks)
    a, b = np.random.choice(n, 2, replace=False)
    if ranks[a] < ranks[b]: return a
    elif ranks[b] < ranks[a]: return b
    else: return a if distances[a] > distances[b] else b

def _sbx_crossover(p1, p2, eta=20):
    n = len(p1)
    child1, child2 = p1.copy(), p2.copy()
    for i in range(n):
        u = np.random.rand()
        if u <= 0.5: beta = (2 * u) ** (1 / (eta + 1))
        else: beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        child1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        child2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
    return child1, child2

def _polynomial_mutation(x, eta=20, prob=0.05):
    x = x.copy()
    for i in range(len(x)):
        if np.random.rand() < prob:
            u  = np.random.rand()
            if u < 0.5: delta = (2 * u) ** (1 / (eta + 1)) - 1
            else: delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            x[i] += delta
    return x

def _repair(x):
    x = np.clip(x, 0, 1)
    s = x.sum()
    return x / s if s > 0 else np.ones(len(x)) / len(x)

def run_nsga2(expected_returns, cov_matrix, pop_size=100, n_gen=200, crossover_prob=0.8, mutation_prob=0.05):
    n_assets = len(expected_returns)
    pop = np.random.dirichlet(np.ones(n_assets), size=pop_size)

    for _ in range(n_gen):
        obj = np.array([_objectives(ind, expected_returns, cov_matrix) for ind in pop])
        fronts = _fast_nondominated_sort(obj)
        ranks  = np.zeros(pop_size, dtype=int)
        for rank, front in enumerate(fronts):
            for idx in front: ranks[idx] = rank

        dists = np.zeros(pop_size)
        for front in fronts:
            cd = _crowding_distance(front, obj)
            for k, idx in enumerate(front): dists[idx] = cd[k]

        offspring = []
        while len(offspring) < pop_size:
            a = _tournament(ranks, dists)
            b = _tournament(ranks, dists)
            if np.random.rand() < crossover_prob: c1, c2 = _sbx_crossover(pop[a], pop[b])
            else: c1, c2 = pop[a].copy(), pop[b].copy()
            c1 = _repair(_polynomial_mutation(c1, prob=mutation_prob))
            c2 = _repair(_polynomial_mutation(c2, prob=mutation_prob))
            offspring.extend([c1, c2])

        combined     = np.vstack([pop, np.array(offspring[:pop_size])])
        obj_combined = np.array([_objectives(ind, expected_returns, cov_matrix) for ind in combined])

        fronts  = _fast_nondominated_sort(obj_combined)
        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend(front)
            else:
                needed = pop_size - len(new_pop)
                cd     = _crowding_distance(front, obj_combined)
                sorted_front = [front[i] for i in np.argsort(-cd)]
                new_pop.extend(sorted_front[:needed])
                break
        pop = combined[new_pop]

    obj_final = np.array([_objectives(ind, expected_returns, cov_matrix) for ind in pop])
    fronts    = _fast_nondominated_sort(obj_final)
    first     = fronts[0]

    rows = []
    for idx in first:
        w    = pop[idx]
        var  = obj_final[idx, 0]
        ret  = -obj_final[idx, 1]
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
# BACKTEST EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

def fetch_sentiment_history():
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key or api_key == "your_newsapi_key_here":
        print("No valid NEWS_API_KEY found, using 0.0 for sentiment.")
        return {t: [] for t in TICKERS}
    
    print("Loading FinBERT for sentiment analysis...")
    finbert = pipeline("text-classification", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")
    newsapi = NewsApiClient(api_key=api_key)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=28) # Max 1 month
    
    article_sentiments_history = {t: [] for t in TICKERS}
    print("Fetching news and computing sentiment...")
    for ticker in TICKERS:
        try:
            q = TICKER_QUERIES.get(ticker, ticker)
            response = newsapi.get_everything(
                q=q, language="en", sort_by="publishedAt", page_size=20, # Fetch more for history
                from_param=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d")
            )
            articles = response.get("articles", [])
            for a in articles:
                if not a.get("title"): continue
                text = f"{a['title']}. {a.get('description', '')}"
                truncated = text[:512]
                res = finbert([truncated])[0]
                lbl = res["label"].lower()
                s = res["score"]
                score = s if lbl == "positive" else -s if lbl == "negative" else 0.0
                date_str = a.get("publishedAt", "")[:10]
                if date_str:
                    article_sentiments_history[ticker].append({"date": date_str, "score": score})
        except Exception as e:
            print(f"Error fetching sentiment for {ticker}: {e}")
    return article_sentiments_history

def run_backtest():
    start_time = time.time()
    
    # 1. Load Data
    print("Downloading historical data...")
    # Need approx 6 months to have 60-day features ready, plus training window for rf
    history_start = datetime.now() - timedelta(days=180)
    history_end = datetime.now()
    prices_full = yf.download(TICKERS, start=history_start.strftime("%Y-%m-%d"), end=history_end.strftime("%Y-%m-%d"), progress=False)["Close"].dropna()
    
    article_history = fetch_sentiment_history()
    
    # The user mentions "1 month of daily prices" but also "Day 1/42" which is approx 2 months of trading days.
    # Let's handle testing length. Let's use 42 days (approx 2 months) as the test period!
    test_days_length = min(42, len(prices_full) - 100) # reserve 100 days min for cold start
    if test_days_length <= 0:
        print("Not enough data to run backtest.")
        return
        
    test_dates = prices_full.index[-test_days_length:]
    
    print(f"Test period: from {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')} ({len(test_dates)} days)")
    
    # Storage
    portfolio_values = {
        "Passive": [100.0],
        "One-Shot": [100.0],
        "NSGA-Twin": [100.0]
    }
    
    weights_history_twin = []
    rf_predictions_history = []
    
    # Init Strategy 1
    w_passive = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
    
    # Init Strategy 2 (One-Shot) computed ONCE at Day 1
    w_oneshot = None
    
    for i, current_date in enumerate(test_dates):
        print(f"Day {i+1}/{len(test_dates)}...")
        
        # Historical data strictly UP TO current_date
        current_idx = prices_full.index.get_loc(current_date)
        prices_window = prices_full.iloc[:current_idx+1]
        
        # Calculate daily sentiment based on articles published <= current_date
        daily_sentiment = {}
        for ticker in TICKERS:
            scores = [a["score"] for a in article_history.get(ticker, []) if a["date"] <= current_date.strftime("%Y-%m-%d")]
            daily_sentiment[ticker] = float(np.mean(scores)) if scores else 0.0
        
        # Compute features
        features_window = compute_features(prices_window, daily_sentiment)
        predicted_returns_dict = predict_returns(features_window, prices_window)
        
        rf_predictions_history.append((current_date, predicted_returns_dict))
        
        returns_window = prices_window.pct_change().dropna()
        hist_mean = returns_window.mean()[TICKERS].values
        rf_pred = np.array([predicted_returns_dict[t] for t in TICKERS])
        exp_ret_vec = 0.5 * rf_pred + 0.5 * hist_mean
        cov_matrix = returns_window.cov().loc[TICKERS, TICKERS].values
        
        # One-Shot runs only once
        if w_oneshot is None:
            print("  Running One-Shot optimization...")
            pareto_df_oneshot = run_nsga2(exp_ret_vec, cov_matrix, pop_size=50, n_gen=50)
            if len(pareto_df_oneshot) > 0:
                opt_row_oneshot = pareto_df_oneshot.iloc[len(pareto_df_oneshot) // 2] # Moderate
                w_oneshot = np.array([opt_row_oneshot[f"w_{t}"] for t in TICKERS])
            else:
                w_oneshot = w_passive
                
        # NSGA-Twin runs every day
        pareto_df_twin = run_nsga2(exp_ret_vec, cov_matrix, pop_size=50, n_gen=50)
        if len(pareto_df_twin) > 0:
            opt_row_twin = pareto_df_twin.iloc[len(pareto_df_twin) // 2] # Moderate
            w_twin = np.array([opt_row_twin[f"w_{t}"] for t in TICKERS])
        else:
            w_twin = w_passive
            
        weights_history_twin.append(w_twin)
        
        # Next day returns (if this is not the last day)
        if i < len(test_dates) - 1:
            next_date = test_dates[i+1]
            actual_next_returns = prices_full.pct_change().loc[next_date, TICKERS].values
            
            # Apply returns
            portfolio_values["Passive"].append(portfolio_values["Passive"][-1] * (1 + w_passive @ actual_next_returns))
            portfolio_values["One-Shot"].append(portfolio_values["One-Shot"][-1] * (1 + w_oneshot @ actual_next_returns))
            portfolio_values["NSGA-Twin"].append(portfolio_values["NSGA-Twin"][-1] * (1 + w_twin @ actual_next_returns))
            
    # Prepare charts data
    dates_for_plot = test_dates[:]
    
    # ── Outputs ─────────────────────────────────────────────────────────────
    
    # Chart 1: Cumulative Performance
    plt.figure(figsize=(12, 6))
    plt.style.use('default')
    plt.plot(dates_for_plot, portfolio_values["Passive"], label="Passive (Equal-Weight)", color="blue")
    plt.plot(dates_for_plot, portfolio_values["One-Shot"], label="One-Shot NSGA-II", color="orange")
    plt.plot(dates_for_plot, portfolio_values["NSGA-Twin"], label="NSGA-Twin (Daily)", color="green")
    plt.title("Backtest: Portfolio Performance Comparison", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (€)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cumulative_performance.png", dpi=150)
    plt.close()
    
    print("Saved cumulative_performance.png")
    
    # Chart 2: Daily Weights Evolution
    weights_matrix = np.array(weights_history_twin) # shape (days, 5)
    plt.figure(figsize=(12, 6))
    plt.stackplot(dates_for_plot, weights_matrix.T, labels=TICKERS, alpha=0.8)
    plt.title("NSGA-Twin: Daily Portfolio Allocation Over Time", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.legend(loc="upper left")
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig("weights_evolution.png", dpi=150)
    plt.close()
    
    print("Saved weights_evolution.png")
    
    # Chart 3: RF Predictions vs Reality
    from sklearn.metrics import r2_score
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    
    # Gather pred/actual pairs
    actual_returns_df = prices_full.pct_change().shift(-1).loc[test_dates[:-1]]
    
    predicted_df = pd.DataFrame([x[1] for x in rf_predictions_history[:-1]], index=test_dates[:-1])
    
    pred_vs_act = {t: {'pred': [], 'act': []} for t in TICKERS}
    
    for dt in test_dates[:-1]: # exclude last day as next day return is unknown
        act = actual_returns_df.loc[dt]
        pred = predicted_df.loc[dt]
        for t in TICKERS:
            pred_vs_act[t]['pred'].append(pred[t])
            pred_vs_act[t]['act'].append(act[t])
            
    for i, t in enumerate(TICKERS):
        preds = np.array(pred_vs_act[t]['pred'])
        acts = np.array(pred_vs_act[t]['act'])
        
        axs[i].scatter(preds, acts, alpha=0.5)
        # Red line
        min_val = min(np.min(preds), np.min(acts))
        max_val = max(np.max(preds), np.max(acts))
        axs[i].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
        
        # Metrics
        r2 = r2_score(acts, preds) if len(acts) > 1 else 0
        directional_acc = np.mean(np.sign(preds) == np.sign(acts)) * 100 if len(acts) > 0 else 0
        
        axs[i].set_title(t)
        axs[i].set_xlabel("Predicted")
        if i == 0: axs[i].set_ylabel("Actual")
        
        axs[i].text(0.05, 0.95, f"R²: {r2:.2f}\nDir Acc: {directional_acc:.1f}%", 
                    transform=axs[i].transAxes, verticalalignment='top')
                    
    fig.suptitle("Random Forest: Predicted vs Actual Returns", fontsize=16)
    plt.tight_layout()
    plt.savefig("rf_accuracy.png", dpi=150)
    plt.close()
    
    print("Saved rf_accuracy.png")
    
    # Table Summary Metrics
    def compute_metrics(vals):
        arr = np.array(vals)
        total_ret = (arr[-1] / arr[0] - 1) * 100
        
        daily_returns = arr[1:] / arr[:-1] - 1
        ann_ret = ((arr[-1] / arr[0]) ** (252 / len(daily_returns)) - 1) * 100
        ann_vol = np.std(daily_returns) * np.sqrt(252) * 100
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        cummax = np.maximum.accumulate(arr)
        drawdown = (cummax - arr) / cummax
        max_dd = np.max(drawdown) * 100
        
        return [f"{total_ret:.2f}%", f"{ann_ret:.2f}%", f"{ann_vol:.2f}%", f"{sharpe:.2f}", f"{max_dd:.2f}%"]
        
    metrics = ["Total return (%)", "Annualized return (%)", "Annualized volatility (%)", "Sharpe ratio", "Maximum drawdown (%)"]
    m_passive = compute_metrics(portfolio_values["Passive"])
    m_oneshot = compute_metrics(portfolio_values["One-Shot"])
    m_twin = compute_metrics(portfolio_values["NSGA-Twin"])
    
    df_metrics = pd.DataFrame({
        "Metric": metrics,
        "Passive": m_passive,
        "One-Shot": m_oneshot,
        "NSGA-Twin": m_twin
    })
    
    print("\n" + "="*60)
    print("BACKTEST SUMMARY")
    print("="*60)
    print(df_metrics.to_string(index=False))
    print("="*60)
    
    # Save as image
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df_metrics.values, colLabels=df_metrics.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.savefig("metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved metrics_summary.png")
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    run_backtest()
