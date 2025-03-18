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

# --- Configuration ---
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", 
    "SBIN.NS", "ICICIBANK.NS", "INFY.NS", 
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS"
]

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

# --- Data Fetching ---
@st.cache_data(show_spinner=False, ttl=3600)
def get_enhanced_data(ticker, days=730):
    """Fetch data using yfinance using explicit start and end dates."""
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days)
        data = yf.download(ticker, start=start_date, end=end_date)
        data = data.reset_index()
        return data
    except Exception as e:
        logging.error(f"Data fetch error for {ticker}: {e}")
        return pd.DataFrame()

def train_model(X, y, model_type):
    """Train selected model with hyperparameter optimization."""
    # Note: y should already be 1D, but we'll ensure it is
    y = np.ravel(y)
    
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
    """Calculate optimal portfolio allocations with error handling."""
    try:
        returns_list = []
        for p in predictions:
            # Only include predictions with valid ticker and non-None volatility
            if p.get('ticker') and (p.get('volatility') is not None):
                returns_list.append({
                    'ticker': p['ticker'],
                    'expected_return': float(p.get('expected_return', 0)),
                    'volatility': float(p.get('volatility', 0))
                })
            
        returns = pd.DataFrame(returns_list)
        
        if returns.empty or 'ticker' not in returns.columns:
            st.warning("No valid predictions available for portfolio optimization.")
            return []

        optimizer = PortfolioOptimizer(
            expected_returns=returns.set_index('ticker')['expected_return'],
            cov_matrix=pd.DataFrame(np.diag(returns['volatility']), 
                                    index=returns['ticker'], 
                                    columns=returns['ticker'])
        )
        return optimizer.optimize(risk_tolerance)
        
    except Exception as e:
        logging.error(f"Portfolio optimization failed: {e}")
        st.error("Failed to optimize portfolio. Check data inputs.")
        return []

def run_quantum_predictions(pred_date):
    results = []
    progress = st.progress(0)
    
    for idx, ticker in enumerate(TICKERS):
        try:
            data = get_enhanced_data(ticker)
            # Ensure sufficient data exists (we need enough for the 200-day moving average)
            if data.empty or len(data) < 200:
                logging.error(f"Insufficient data for {ticker}")
                continue
                
            # Feature Engineering
            data['MA_50'] = data['Close'].rolling(50).mean()
            data['MA_200'] = data['Close'].rolling(200).mean()
            data = data.dropna()
            
            if data.empty:
                logging.error(f"Not enough data after computing moving averages for {ticker}")
                continue
            
            # Prepare data
            X = data[['MA_50', 'MA_200']]
            y = data['Close']
            
            # Train model
            model = train_model(X, y, model_choice)
            
            # Generate prediction using the latest moving averages
            last_ma50 = data['MA_50'].iloc[-1]
            last_ma200 = data['MA_200'].iloc[-1]
            predicted_price = model.predict([[last_ma50, last_ma200]])[0]
            
            # Risk Analysis
            current_price = float(data['Close'].iloc[-1])
            volatility = float(data['Close'].pct_change().std() * np.sqrt(252))
            sharpe_ratio = float((data['Close'].pct_change().mean() / volatility) * np.sqrt(252))
            
            results.append({
                "ticker": ticker,
                "current_price": current_price,
                "predicted_price": float(predicted_price),
                "expected_return": float(predicted_price / current_price - 1),
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio
            })
            
            progress.progress((idx + 1) / len(TICKERS))
        except Exception as e:
            logging.error(f"Error processing {ticker}: {e}")
            continue
    
    return results

# --- Main Interface ---
st.title("AI Wealth Management Platform")
st.markdown("### Intelligent Portfolio Optimization")

if refresh_data or st.sidebar.button("Run Analysis"):
    with st.spinner("Running Financial Models..."):
        predictions = run_quantum_predictions(prediction_date)
        portfolio = optimize_portfolio(predictions)
    
    if predictions:
        # Create tabs for better organization
        tab1, tab2 = st.tabs(["Stock Predictions", "Portfolio Allocation"])
        
        with tab1:
            # Display predictions in a more organized manner
            st.subheader("Market Predictions")
            
            # Create a dataframe for all predictions to display in a table
            pred_df = pd.DataFrame(predictions)
            pred_df['Return (%)'] = pred_df['expected_return'] * 100
            pred_df = pred_df.rename(columns={
                'ticker': 'Ticker',
                'current_price': 'Current Price (₹)',
                'predicted_price': 'Predicted Price (₹)',
                'volatility': 'Volatility',
                'sharpe_ratio': 'Sharpe Ratio'
            })
            st.dataframe(
                pred_df[['Ticker', 'Current Price (₹)', 'Predicted Price (₹)', 'Return (%)', 'Volatility', 'Sharpe Ratio']]
                .sort_values('Return (%)', ascending=False)
                .style.format({
                    'Current Price (₹)': '₹{:.2f}',
                    'Predicted Price (₹)': '₹{:.2f}',
                    'Return (%)': '{:.2f}%',
                    'Volatility': '{:.2f}',
                    'Sharpe Ratio': '{:.2f}'
                }),
                use_container_width=True
            )
            
            # Display individual stock cards in a more organized grid
            st.subheader("Stock Details")
            
            # Create 3 columns for stock cards
            cols = st.columns(1)
            for idx, res in enumerate(predictions):
                with cols[idx % 1]:
                    with st.container():
                        st.markdown(f"##### {res['ticker']}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Current", f"₹{float(res['current_price']):,.2f}")
                        with col2:
                            st.metric("Predicted", f"₹{float(res['predicted_price']):,.2f}", 
                                    delta=f"{(res['predicted_price']/res['current_price']-1):.2%}")
                        st.markdown(f"**Volatility:** {res['volatility']:.2f} | **Sharpe:** {res['sharpe_ratio']:.2f}")
                        st.markdown("---")
        
        with tab2:
            st.subheader("Optimal Portfolio Allocation")
            
            # Show investment amount prominently
            st.markdown(f"### Total Investment: ₹{investment_amount:,.2f}")
            
            if portfolio:
                # Create a pie chart for portfolio allocation
                fig, ax = plt.subplots(figsize=(10, 6))
                labels = [alloc['ticker'] for alloc in portfolio if alloc['percentage'] > 1]
                sizes = [alloc['percentage'] for alloc in portfolio if alloc['percentage'] > 1]
                
                # Add "Others" category for small allocations
                small_allocs = [alloc for alloc in portfolio if alloc['percentage'] <= 1]
                if small_allocs:
                    labels.append("Others")
                    sizes.append(sum(alloc['percentage'] for alloc in small_allocs))
                
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
                
                # Display allocation table
                alloc_data = []
                for alloc in portfolio:
                    alloc_data.append({
                        "Ticker": alloc['ticker'],
                        "Allocation (%)": f"{alloc['percentage']:.1f}%",
                        "Amount (₹)": f"₹{investment_amount * alloc['percentage']/100:,.2f}",
                        "Expected Return": f"{alloc['expected_return']:.2%}",
                        "Volatility": f"{alloc['volatility']:.2f}"
                    })
                
                st.dataframe(pd.DataFrame(alloc_data), use_container_width=True)
                
                # Display risk metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    portfolio_return = sum(alloc['percentage'] * alloc['expected_return'] for alloc in portfolio) / 100
                    st.metric("Expected Return", f"{portfolio_return:.2%}")
                with col2:
                    # Simple portfolio volatility estimate
                    portfolio_vol = sum(alloc['percentage'] * alloc['volatility'] for alloc in portfolio) / 100
                    st.metric("Portfolio Volatility", f"{portfolio_vol:.2f}")
                with col3:
                    sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            else:
                st.warning("No portfolio allocation generated")

# Add footer with disclaimer
st.markdown("---")
st.caption("Disclaimer: This is for educational purposes only. Invest at your own risk.")