import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

#  Page Config (Layout) 
st.set_page_config(
    page_title="NSGA-Twin Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title 
st.title("AI & NSGA-II Driven Portfolio Digital Twin")
st.markdown("### Real-Time Portfolio Management")
st.markdown("---")

# Fake Data Generation (a modif + tard quand on aura les vraies données)
@st.cache_data(ttl=5)
def simulate_market_data():
    """Simulates real-time market data for a few ETFs."""
    np.random.seed(int(time.time()))
    assets = ['S&P500', 'GLD', 'TLT', 'VNQ']
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='D')
    prices = pd.DataFrame(np.random.randn(100, len(assets)).cumsum(axis=0) + 100, columns=assets, index=dates)
    return prices

@st.cache_data(ttl=5)
def simulate_pareto_front():
    """Simulates the Pareto front produced by NSGA-II. (à adapter + tard quand on aura le programme)"""
    np.random.seed(int(time.time()))
    risks = np.linspace(0.05, 0.25, 50)
    returns = 0.02 + 0.5 * np.sqrt(risks) + np.random.normal(0, 0.01, 50)
    pareto_df = pd.DataFrame({'Risk': risks, 'Expected Return': returns})
    return pareto_df

@st.cache_data(ttl=5)
def simulate_ai_anomaly():
    """Simulates an Isolation Forest anomaly detection state. (à adapter + tard quand on aura le programme)"""
    is_anomaly = np.random.rand() > 0.90
    return is_anomaly


prices_df = simulate_market_data()
pareto_df = simulate_pareto_front()
is_anomaly = simulate_ai_anomaly()

# Current mock portfolio state
current_risk = 0.12
current_return = 0.16

# Sidebar Config
with st.sidebar:
    st.header("Configuration")
    risk_profile = st.select_slider(
        "Risk Profile",
        options=["Conservative", "Moderate", "Aggressive"],
        value="Moderate"
    )

# -- TOP SECTION : Market Monitor & Digital Twin Side-by-Side --
col_market, col_twin = st.columns(2)

with col_market:
    st.subheader("Market Monitor (RT)")
    fig_prices = px.line(prices_df, title="Asset Prices over time", labels={'value': 'Price', 'index': 'Date'})
    st.plotly_chart(fig_prices, use_container_width=True)

with col_twin:
    st.subheader("NSGA-II Pareto Front")
    fig_pareto = px.scatter(
        pareto_df, x="Risk", y="Expected Return", 
        title="Multi-Objective Optimization"
    )
    
    # Current portfolio
    fig_pareto.add_trace(go.Scatter(
        x=[current_risk], y=[current_return], 
        mode='markers', marker=dict(color='red', size=15, symbol='star'),
        name='Current Physical Portfolio'
    ))
    
    # Target Portfolio based on risk profile (slidebar)
    if risk_profile == "Conservative":
        target_risk = pareto_df['Risk'].min()
        target_return = pareto_df.loc[pareto_df['Risk'].idxmin()]['Expected Return']
    elif risk_profile == "Aggressive":
        target_risk = pareto_df['Risk'].max()
        target_return = pareto_df.loc[pareto_df['Risk'].idxmax()]['Expected Return']
    else:
        mid_idx = len(pareto_df) // 2
        target_risk = pareto_df.iloc[mid_idx]['Risk']
        target_return = pareto_df.iloc[mid_idx]['Expected Return']

    # Target portfolio
    fig_pareto.add_trace(go.Scatter(
        x=[target_risk], y=[target_return], 
        mode='markers', marker=dict(color='green', size=15, symbol='star'),
        name=f'Optimal Virtual Target ({risk_profile})'
    ))
    
    fig_pareto.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_pareto, use_container_width=True)

st.markdown("---")

# BOTTOM: AI Analysis & Rebalancing
col_ai, col_rebalance = st.columns(2)

with col_ai:
    st.subheader("AI Layer Analysis")
    st.markdown("**Isolation Forest Anomaly Status:**")
    if is_anomaly:
        st.error("**Anomaly Detected!** High volatility regime. Emergency Rebalancing Triggered.")
    else:
        st.success("**Normal Market Regime.** No anomalies detected.")
        
    st.markdown("**Random Forest Return Predictions (Next 1M):**")
    pred_data = {
        "Asset": ['S&P500', 'GLD', 'TLT', 'VNQ'], 
        "Predicted Return": [f"{np.random.normal(0.01, 0.02)*100:.2f}%" for _ in range(4)]
    }
    st.dataframe(pd.DataFrame(pred_data), use_container_width=True)

with col_rebalance:
    st.subheader("Rebalancing Action")
    st.markdown(f"**Current Status:** {'Needs Rebalancing' if is_anomaly or risk_profile != 'Moderate' else 'Optimal'}")
    
    weights_data = {
        "Asset": ['S&P500', 'GLD', 'TLT', 'VNQ'],
        "Current Weight": [0.30, 0.20, 0.30, 0.10],
        "Target Weight": [0.25, 0.15, 0.40, 0.15] if risk_profile == "Conservative" else 
                         [0.40, 0.30, 0.20, 0.05] if risk_profile == "Aggressive" else
                         [0.35, 0.25, 0.25, 0.05]
    }
    
    df_weights = pd.DataFrame(weights_data)
    df_weights["Delta"] = df_weights["Target Weight"] - df_weights["Current Weight"]
    
    def highlight_delta(val):
        color = '#00FF00' if val > 0 else '#FF0000' if val < 0 else 'white'
        return f'color: {color}'
    
    st.dataframe(df_weights.style.map(highlight_delta, subset=['Delta']), use_container_width=True)
    
    if st.button("Execute Trades (Update Physical Twin)", use_container_width=True):
        st.success("Transactions sent via MQTT!")
