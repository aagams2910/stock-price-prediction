import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from portfolio_optimizer import PortfolioOptimizer  # Custom module

# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# --- Security Configuration ---
AUTHORIZED_USERS = {
    "admin": hashlib.sha256("$UP3R$3CR3T".encode()).hexdigest(),
    "client": hashlib.sha256("M1LL10N@1R3".encode()).hexdigest()
}

# --- Premium Data Sources ---
QUANDL_API_KEY = st.secrets.get("QUANDL_API_KEY", "")
BLOOMBERG_ENABLED = False  # Set to True if Bloomberg Terminal integration

# --- UI Configuration ---
plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")
st.set_page_config(
    page_title="NSE Quantum Wealth Manager",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💰"
)

# --- Authentication ---
def authenticate():
    with st.sidebar:
        st.subheader("Secure Access")
        user = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if user and password:
            if AUTHORIZED_USERS.get(user) == hashlib.sha256(password.encode()).hexdigest():
                return True
            st.error("Invalid credentials")
            return False
    return False

if not authenticate():
    st.stop()

# --- Premium Features Configuration ---
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", 
    "SBIN.NS", "ICICIBANK.NS", "INFY.NS", 
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS"
]
BENCHMARKS = ["^NSEI", "^NSEMDCP50", "GOLDBEES.NS"]

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Quantum Wealth Parameters")
    prediction_date = st.date_input("Prediction Date", datetime.today() + timedelta(days=1))
    model_choice = st.selectbox("AI Model", ["Quantum Ensemble", "Random Forest", "Gradient Boosting", "Linear Regression"])
    risk_tolerance = st.slider("Risk Tolerance (1-10)", 1, 10, 7)
    refresh_data = st.button("🚀 Refresh Market Data")
    st.markdown("---")
    st.header("Portfolio Preferences")
    investment_amount = st.number_input("Investment Amount (₹)", min_value=1e6, value=5e6, step=1e5)
    sectors = st.multiselect("Preferred Sectors", ["Banking", "Technology", "Energy", "Consumer", "Infrastructure"])

# --- Enhanced Data Fetching with Fallback ---
@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda _: None}, ttl=3600)
def get_enhanced_data(ticker, days=730):
    """Fetch data with premium sources fallback"""
    try:
        if BLOOMBERG_ENABLED:
            # Bloomberg Terminal integration
            data = pd.read_csv(f"bloomberg:{ticker}", parse_dates=['Date'])
        elif QUANDL_API_KEY:
            data = pd.read_csv(
                f"https://www.quandl.com/api/v3/datasets/NSE/{ticker}?api_key={QUANDL_API_KEY}",
                parse_dates=['Date']
            )
        else:
            data = yf.download(ticker, period=f"{days}d", prepost=True)
            data = data.reset_index()
    except Exception as e:
        logging.error(f"Data fetch error for {ticker}: {e}")
        data = yf.download(ticker, period=f"{days}d")
        data = data.reset_index()
    return data

# --- Advanced Model Training ---
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

# --- Portfolio Optimization ---
def optimize_portfolio(predictions):
    """Calculate optimal portfolio allocations"""
    returns = pd.DataFrame([{
        'ticker': p['ticker'],
        'expected_return': p['expected_return'],
        'volatility': p['volatility']
    } for p in predictions])
    
    optimizer = PortfolioOptimizer(
        expected_returns=returns['expected_return'],
        cov_matrix=returns['volatility'],
        risk_free_rate=0.05
    )
    return optimizer.optimize(risk_tolerance)

# --- Enhanced Prediction Engine ---
def run_quantum_predictions(pred_date):
    results = []
    progress = st.progress(0)
    
    for idx, ticker in enumerate(TICKERS):
        try:
            # Fetch enhanced historical data
            data = get_enhanced_data(ticker)
            data['Date_ordinal'] = data['Date'].dt.toordinal()
            
            # Feature Engineering
            data['MA_50'] = data['Close'].rolling(50).mean()
            data['MA_200'] = data['Close'].rolling(200).mean()
            data = data.dropna()
            
            # Prepare data
            train_size = int(len(data) * 0.8)
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]
            
            X_train = train_data[['Date_ordinal', 'MA_50', 'MA_200']]
            y_train = train_data['Close']
            
            # Train model
            model = train_model(X_train, y_train, model_choice)
            
            # Generate predictions
            pred_features = np.array([[pred_date.toordinal(),
                                     train_data['MA_50'].iloc[-1],
                                     train_data['MA_200'].iloc[-1]]])
            predicted_price = model.predict(pred_features)[0]
            
            # Risk Analysis
            volatility = data['Close'].pct_change().std() * np.sqrt(252)
            sharpe_ratio = (data['Close'].pct_change().mean() / volatility) * np.sqrt(252)
            
            # Benchmark Comparison
            bench_data = get_enhanced_data(BENCHMARKS[0])
            bench_return = bench_data['Close'].pct_change().mean() * 252
            
            results.append({
                "ticker": ticker,
                "current_price": data['Close'].iloc[-1],
                "predicted_price": predicted_price,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "alpha": (predicted_price - data['Close'].iloc[-1]) - bench_return,
                "benchmark": bench_return
            })
            
            progress.progress((idx + 1) / len(TICKERS))
        except Exception as e:
            logging.error(f"Error processing {ticker}: {e}")
            continue
    
    return results

# --- Main Interface ---
st.title("Quantum Wealth Management Platform")
st.markdown("### AI-Driven Portfolio Optimization for Ultra-High-Net-Worth Individuals")

if refresh_data or st.sidebar.button("Run Quantum Analysis"):
    with st.spinner("🚀 Running Quantum Financial Models..."):
        predictions = run_quantum_predictions(prediction_date)
        portfolio = optimize_portfolio(predictions)
    
    # Display Results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("AI Stock Predictions")
        cols = st.columns(3)
        for idx, res in enumerate(predictions):
            with cols[idx % 3]:
                with st.container():
                    st.subheader(res['ticker'])
                    st.metric("Current", f"₹{res['current_price']:,.2f}")
                    st.metric("Predicted", f"₹{res['predicted_price']:,.2f}",delta=f"{(res['predicted_price']/res['current_price'])-1:.2%}")
                    st.write(f"**Volatility:** {res['volatility']:.2f}")
                    st.write(f"**Sharpe Ratio:** {res['sharpe_ratio']:.2f}")
                    st.write(f"**Alpha vs Nifty:** {res['alpha']:.2%}")
    
    with col2:
        st.header("Optimal Portfolio Allocation")
        st.write(f"**Investment:** ₹{investment_amount:,.2f}")
        for alloc in portfolio:
            st.progress(alloc['percentage'])
            st.write(f"{alloc['ticker']}: {alloc['percentage']:.1f}% (₹{investment_amount * alloc['percentage']/100:,.2f})")
        
        st.subheader("Risk Analysis")
        st.vega_lite_chart({
            "mark": {"type": "circle", "tooltip": True},
            "encoding": {
                "x": {"field": "volatility", "type": "quantitative"},
                "y": {"field": "expected_return", "type": "quantitative"},
                "size": {"field": "allocation", "type": "quantitative"},
                "color": {"field": "ticker", "type": "nominal"}
            }
        })

# --- Premium Support Section ---
with st.expander("📞 Premium Support"):
    st.write("""
    **24/7 Wealth Management Support:**
    - 📞 +91 90000 00000
    - ✉️ elite-support@quantumwealth.com
    - 🚨 Emergency Portfolio Protection Line: +91 80000 00000
    """)

# --- Disclaimers ---
st.caption("""
*This system uses quantum-inspired algorithms for financial forecasting. 
Past performance is not indicative of future results. 
Consult your financial advisor before making investment decisions.*
""")