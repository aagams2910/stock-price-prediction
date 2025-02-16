import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from portfolio_optimizer import PortfolioOptimizer

# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# # --- UI Configuration ---
# plt.style.use('seaborn-darkgrid')
# sns.set_palette("husl")
# st.set_page_config(
#     page_title="Quantum Wealth Manager",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     page_icon="💰"
# )

# --- Configuration ---
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", 
    "SBIN.NS", "ICICIBANK.NS", "INFY.NS", 
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS"
]
BENCHMARKS = ["^NSEI", "^NSEMDCP50", "GOLDBEES.NS"]

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Investment Parameters")
    prediction_date = st.date_input("Prediction Date", datetime.today() + timedelta(days=1))
    model_choice = st.selectbox("AI Model", ["Quantum Ensemble", "Random Forest", "Gradient Boosting", "Linear Regression"])
    risk_tolerance = st.slider("Risk Tolerance (1-10)", 1, 10, 7)
    refresh_data = st.button("🚀 Refresh Market Data")
    st.markdown("---")
    st.header("Portfolio Preferences")
    investment_amount = st.number_input("Investment Amount (₹)", min_value=1e6, value=5e6, step=1e5)
    sectors = st.multiselect("Preferred Sectors", ["Banking", "Technology", "Energy", "Consumer", "Infrastructure"])

# --- Data Fetching ---
@st.cache_data(show_spinner=False, ttl=3600)
def get_enhanced_data(ticker, days=730):
    """Fetch data using yfinance"""
    try:
        data = yf.download(ticker, period=f"{days}d")
        data = data.reset_index()
        return data
    except Exception as e:
        logging.error(f"Data fetch error for {ticker}: {e}")
        return pd.DataFrame()

# --- Main Interface ---
st.title("AI Wealth Management Platform")
st.markdown("### Intelligent Portfolio Optimization")

# Add these functions after the get_enhanced_data function

def train_model(X, y, model_type):
    """Train selected model with hyperparameter optimization"""
    if model_type == "Quantum Ensemble":
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
    elif model_type == "Random Forest":
        param_dist = {'n_estimators': randint(50, 200),
                     'max_depth': randint(2, 10)}
        model = RandomizedSearchCV(RandomForestRegressor(), param_dist, n_iter=10)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1)
    else:
        model = LinearRegression()
    
    model.fit(X, y)
    return model

def optimize_portfolio(predictions):
    """Calculate optimal portfolio allocations"""
    returns = pd.DataFrame([{
        'ticker': p['ticker'],
        'expected_return': p['expected_return'],
        'volatility': p['volatility']
    } for p in predictions])
    
    optimizer = PortfolioOptimizer(
        expected_returns=returns.set_index('ticker')['expected_return'],
        cov_matrix=pd.DataFrame(np.diag(returns['volatility']), 
                              index=returns['ticker'], 
                              columns=returns['ticker'])
    )
    return optimizer.optimize(risk_tolerance)

def run_quantum_predictions(pred_date):
    results = []
    progress = st.progress(0)
    
    for idx, ticker in enumerate(TICKERS):
        try:
            data = get_enhanced_data(ticker)
            if data.empty:
                continue
                
            # Feature Engineering
            data['MA_50'] = data['Close'].rolling(50).mean()
            data['MA_200'] = data['Close'].rolling(200).mean()
            data = data.dropna()
            
            # Prepare data
            X = data[['MA_50', 'MA_200']]
            y = data['Close']
            
            # Train model
            model = train_model(X, y, model_choice)
            
            # Generate predictions
            last_ma50 = data['MA_50'].iloc[-1]
            last_ma200 = data['MA_200'].iloc[-1]
            predicted_price = model.predict([[last_ma50, last_ma200]])[0]
            
            # Risk Analysis
            volatility = data['Close'].pct_change().std() * np.sqrt(252)
            sharpe_ratio = (data['Close'].pct_change().mean() / volatility) * np.sqrt(252)
            
            results.append({
                "ticker": ticker,
                "current_price": data['Close'].iloc[-1],
                "predicted_price": predicted_price,
                "expected_return": (predicted_price / data['Close'].iloc[-1] - 1),
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio
            })
            
            progress.progress((idx + 1) / len(TICKERS))
        except Exception as e:
            logging.error(f"Error processing {ticker}: {e}")
            continue
    
    return results

if refresh_data or st.sidebar.button("Run Analysis"):
    with st.spinner("Running Financial Models..."):
        predictions = run_quantum_predictions(prediction_date)
        portfolio = optimize_portfolio(predictions)
    
    # Display Results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Stock Predictions")
        cols = st.columns(3)
        for idx, res in enumerate(predictions):
            with cols[idx % 3]:
                with st.container():
                    st.subheader(res['ticker'])
                    st.metric("Current", f"₹{res['current_price']:,.2f}")
                    st.metric("Predicted", f"₹{res['predicted_price']:,.2f}", 
                             delta=f"{((res['predicted_price']/res['current_price'])-1):.2%}")
                    st.write(f"**Volatility:** {res['volatility']:.2f}")
                    st.write(f"**Sharpe Ratio:** {res['sharpe_ratio']:.2f}")
    
    with col2:
        st.header("Portfolio Allocation")
        st.write(f"**Investment:** ₹{investment_amount:,.2f}")
        for alloc in portfolio:
            st.progress(alloc['percentage'])
            st.write(f"{alloc['ticker']}: {alloc['percentage']:.1f}% (₹{investment_amount * alloc['percentage']/100:,.2f})")

st.caption("Disclaimer: This is for educational purposes only. Invest at your own risk.")