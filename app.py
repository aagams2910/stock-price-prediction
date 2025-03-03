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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
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
    prediction_days = st.slider("Prediction Days Ahead", 1, 30, 7)
    prediction_date = datetime.today() + timedelta(days=prediction_days)
    st.write(f"Prediction Date: {prediction_date.strftime('%Y-%m-%d')}")
    model_choice = st.selectbox("AI Model", ["Advanced Ensemble", "Random Forest", "Gradient Boosting", "Linear Regression"])
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
        if data.empty:
            logging.warning(f"No data found for {ticker} from yfinance.")
        return data
    except Exception as e:
        logging.error(f"Data fetch error for {ticker}: {e}")
        return pd.DataFrame()

def create_features(data):
    """Create advanced technical indicators as features."""
    df = data.copy()
    # Price-based features
    df['Return'] = df['Close'].pct_change()
    df['Return_Lag1'] = df['Return'].shift(1)
    df['Return_Lag2'] = df['Return'].shift(2)
    df['Return_Lag3'] = df['Return'].shift(3)
    df['Return_Lag5'] = df['Return'].shift(5)
    # Moving averages
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['MA_200'] = df['Close'].rolling(200).mean()
    # Moving average crossovers
    df['MA_5_10_Ratio'] = df['MA_5'] / df['MA_10']
    df['MA_10_50_Ratio'] = df['MA_10'] / df['MA_50']
    df['MA_50_200_Ratio'] = df['MA_50'] / df['MA_200']
    # Volatility indicators
    df['Volatility_10'] = df['Return'].rolling(10).std()
    df['Volatility_30'] = df['Return'].rolling(30).std()
    # Volume indicators
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']
    # Trend indicators
    df['Price_200MA_Ratio'] = df['Close'] / df['MA_200']
    # Momentum indicators
    df['RSI_14'] = calculate_rsi(df['Close'], 14)
    # Drop NaN values
    df = df.dropna()
    return df

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=period-1, adjust=False).mean()
    ema_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def train_model(X, y, model_type, pred_days=1):
    """Train selected model with hyperparameter optimization."""
    # Convert to 1D array to avoid warnings
    y = np.ravel(y)
    # Use time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    if model_type == "Advanced Ensemble":
        # Create a pipeline with scaling and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                min_samples_split=10,
                min_samples_leaf=5
            ))
        ])
        model = pipeline
    elif model_type == "Random Forest":
        param_dist = {
            'n_estimators': randint(100, 300),
            'max_depth': randint(3, 15),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10)
        }
        base_model = RandomForestRegressor(random_state=42)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', base_model)
        ])
        model = RandomizedSearchCV(
            pipeline,
            param_distributions={'model__' + key: val for key, val in param_dist.items()},
            n_iter=10,
            cv=tscv,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "Gradient Boosting":
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            ))
        ])
        model = pipeline
    else: # Linear Regression
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])
        model = pipeline

    model.fit(X, y)
    return model

def predict_future_prices(model, X_last, pred_days, current_price, ticker):
    """Predict future prices for multiple days."""
    future_prices = []
    # Make a copy of the last data point to modify for future predictions
    X_future = X_last.copy()
    # Current date and price
    current_date = datetime.today()
    price = current_price

    for i in range(pred_days):
        next_date = current_date + timedelta(days=i+1)
        # Skip weekends for predictions
        if next_date.weekday() >= 5: # 5 is Saturday, 6 is Sunday
            continue
        # Predict next price
        next_price = float(model.predict(X_future)[0])
        # Update features that depend on the previous prediction for the next iteration
        # This is a simplification and would need to be adapted based on your feature set
        future_prices.append({
            'date': next_date,
            'price': next_price,
            'ticker': ticker
        })
        # Update price for next iteration
        price = next_price
    return future_prices

def run_advanced_predictions(pred_days):
    all_results = []
    all_future_predictions = []
    progress = st.progress(0)

    for idx, ticker in enumerate(TICKERS):
        try:
            data = get_enhanced_data(ticker)
            # Ensure sufficient data exists
            if data.empty or len(data) < 200:
                logging.warning(f"Insufficient data for {ticker}. Skipping.")
                continue

            # Feature Engineering
            data_with_features = create_features(data)
            if data_with_features.empty:
                logging.warning(f"Not enough data after computing features for {ticker}. Skipping.")
                continue

            # Define features to use
            feature_columns = [
                'MA_5', 'MA_10', 'MA_20', 'MA_50', 'MA_200',
                'Return_Lag1', 'Return_Lag2', 'Return_Lag3',
                'MA_5_10_Ratio', 'MA_10_50_Ratio', 'MA_50_200_Ratio',
                'Volatility_10', 'Volatility_30',
                'Volume_Ratio', 'Price_200MA_Ratio', 'RSI_14'
            ]

            # Prepare data
            X = data_with_features[feature_columns]
            y = data_with_features['Close']

            # Train model
            model = train_model(X, y, model_choice, pred_days)

            # Get the last data point for prediction
            X_last = X.iloc[[-1]]

            # Current price
            current_price = float(data['Close'].iloc[-1])

            # Generate future price predictions
            future_predictions = predict_future_prices(model, X_last, pred_days, current_price, ticker)
            all_future_predictions.extend(future_predictions)

            # For final prediction date
            final_pred_date = prediction_date

            # Find the closest prediction date if the exact one doesn't exist
            pred_prices = [p['price'] for p in future_predictions
                           if abs((p['date'] - final_pred_date).days) < 2]

            predicted_price = pred_prices[-1] if pred_prices else current_price * 1.01  # Use a fallback

            # Risk Analysis
            returns = data['Close'].pct_change().dropna()
            volatility = float(returns.std() * np.sqrt(252))
            sharpe_ratio = float((returns.mean() / returns.std()) * np.sqrt(252))

            # Confidence interval based on volatility
            confidence = 0.95
            z_score = 1.96  # 95% confidence
            std_dev = volatility * current_price * np.sqrt(pred_days / 252)
            lower_bound = predicted_price - z_score * std_dev
            upper_bound = predicted_price + z_score * std_dev

            # Append results
            result = {
                "ticker": ticker,
                "company_name": ticker.split('.')[0],  # Simple name extraction
                "current_price": current_price,
                "predicted_price": float(predicted_price),
                "expected_return": float(predicted_price / current_price - 1),
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "lower_bound": max(0, float(lower_bound)),
                "upper_bound": float(upper_bound),
                "confidence": confidence,
                "historical_data": data[['Date', 'Close']].rename(columns={'Close': 'price'}).tail(90),
                "future_predictions": future_predictions
            }
            all_results.append(result)
            logging.info(f"Successfully processed {ticker}. Predicted Price: {predicted_price}, Volatility: {volatility}")

        except Exception as e:
            logging.error(f"Error processing {ticker}: {e}")
            continue  # Continue to the next ticker

        progress.progress((idx + 1) / len(TICKERS))
    print(f"Number of results before returning: {len(all_results)}")  # Debugging line

    return all_results, all_future_predictions

def optimize_portfolio(predictions):
    """Calculate optimal portfolio allocations with error handling."""
    try:
        returns_list = []
        for p in predictions:
            # Check for missing or invalid data
            if not all(key in p for key in ['ticker', 'expected_return', 'volatility']):
                logging.warning(f"Skipping prediction due to missing keys: {p}")
                continue

            if p['ticker'] is None:
                logging.warning(f"Skipping prediction due to None ticker: {p}")
                continue

            if not isinstance(p['volatility'], (int, float)):
                logging.warning(f"Skipping prediction due to invalid volatility: {p}")
                continue

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

        # Log the tickers used for portfolio optimization
        logging.info(f"Tickers used for portfolio optimization: {returns['ticker'].tolist()}")

        # Debugging: Print the expected returns and covariance matrix
        expected_returns = returns.set_index('ticker')['expected_return']
        cov_matrix = pd.DataFrame(np.diag(returns['volatility']),
                                    index=returns['ticker'],
                                    columns=returns['ticker'])
        logging.debug(f"Expected Returns:\n{expected_returns}")
        logging.debug(f"Covariance Matrix:\n{cov_matrix}")

        optimizer = PortfolioOptimizer(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix
        )
        portfolio = optimizer.optimize(risk_tolerance)

        if not portfolio:
             st.warning("Portfolio optimization failed. Check the logs for details.")
             return []

        return portfolio

    except Exception as e:
        logging.error(f"Portfolio optimization failed: {e}")
        st.error("Failed to optimize portfolio. Check data inputs.")
        return []

# --- Main Interface ---
st.title("AI Wealth Management Platform")
st.markdown("### Intelligent Portfolio Optimization")

if refresh_data or st.sidebar.button("Run Analysis"):
    with st.spinner("Running Advanced Financial Models..."):
        predictions, all_future_data = run_advanced_predictions(prediction_days)
        print(f"Number of predictions after run_advanced_predictions: {len(predictions)}") # Debugging
        portfolio = optimize_portfolio(predictions)

    if predictions:
        # Create tabs for better organization
        tab1, tab2 = st.tabs(["Stock Predictions", "Portfolio Allocation"])

        with tab1:
            # Display predictions in a more organized manner
            st.subheader(f"Market Predictions for {prediction_date.strftime('%Y-%m-%d')}")

            # Create a dataframe for all predictions to display in a table
            pred_df = pd.DataFrame([{
                'Ticker': p['ticker'],
                'Current Price (₹)': p['current_price'],
                'Predicted Price (₹)': p['predicted_price'],
                'Return (%)': p['expected_return'] * 100,
                'Volatility': p['volatility'],
                'Sharpe Ratio': p['sharpe_ratio']
            } for p in predictions])

            st.dataframe(
                pred_df.sort_values('Return (%)', ascending=False)
                .style.format({
                    'Current Price (₹)': '₹{:.2f}',
                    'Predicted Price (₹)': '₹{:.2f}',
                    'Return (%)': '{:.2f}%',
                    'Volatility': '{:.2f}',
                    'Sharpe Ratio': '{:.2f}'
                }),
                use_container_width=True
            )

            # Display individual stock predictions with line charts
            st.subheader("Individual Stock Forecasts")
            for stock in predictions:
                with st.container():
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        st.subheader(stock['ticker'])
                        st.metric("Current", f"₹{stock['current_price']:.2f}")
                        st.metric("Predicted", f"₹{stock['predicted_price']:.2f}",
                                  delta=f"{stock['expected_return']:.2%}")
                        st.write(f"**Volatility:** {stock['volatility']:.2f}")
                        st.write(f"**Sharpe Ratio:** {stock['sharpe_ratio']:.2f}")
                        st.write(f"**95% Confidence Interval:**")
                        st.write(f"Lower: ₹{stock['lower_bound']:.2f}")
                        st.write(f"Upper: ₹{stock['upper_bound']:.2f}")

                    with col2:
                        # Prepare data for plotting
                        hist_data = stock['historical_data'].copy()
                        hist_data['source'] = 'historical'

                        # Prepare future data
                        future_data = pd.DataFrame(stock['future_predictions'])
                        if not future_data.empty:
                            future_data['source'] = 'prediction'

                            # Combine historical and future data
                            plot_data = pd.concat([
                                hist_data[['Date', 'price', 'source']].rename(columns={'Date': 'date'}),
                                future_data[['date', 'price', 'source']]
                            ])

                            # Create the plot
                            fig, ax = plt.subplots(figsize=(10, 6))

                            # Plot historical data
                            sns.lineplot(
                                data=plot_data[plot_data['source'] == 'historical'],
                                x='date', y='price', ax=ax, label='Historical', color='blue'
                            )

                            # Plot prediction data
                            sns.lineplot(
                                data=plot_data[plot_data['source'] == 'prediction'],
                                x='date', y='price', ax=ax, label='Prediction', color='red'
                            )

                            # Add confidence interval for predictions
                            future_dates = future_data['date']
                            if not future_dates.empty:
                                lower_bound = stock['lower_bound']
                                upper_bound = stock['upper_bound']

                                # Calculate a simple confidence band
                                conf_data = future_data.copy()
                                price_range = stock['predicted_price'] - stock['current_price']
                                days = len(conf_data)
                                conf_data['lower'] = [
                                    stock['current_price'] + (price_range * i / days) - (stock['volatility'] * stock['current_price'] * np.sqrt((i + 1) / 252) * 1.96)
                                    for i in range(days)
                                ]

                                conf_data['upper'] = [
                                    stock['current_price'] + (price_range * i / days) + (stock['volatility'] * stock['current_price'] * np.sqrt((i + 1) / 252) * 1.96)
                                    for i in range(days)
                                ]

                                ax.fill_between(
                                    conf_data['date'],
                                    conf_data['lower'],
                                    conf_data['upper'],
                                    alpha=0.2, color='red',
                                    label='95% Confidence Interval'
                                )

                            # Format the plot
                            plt.title(f"{stock['ticker']} Price Forecast")
                            plt.xlabel("Date")
                            plt.ylabel("Price (₹)")
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            plt.legend()

                            # Display the plot
                            st.pyplot(fig)
                    with col3:
                        # Calculate additional metrics
                        returns = stock['historical_data']['price'].pct_change().dropna()
                        # Monthly returns
                        monthly_return = returns.mean() * 21
                        # Maximum drawdown
                        cum_returns = (1 + returns).cumprod()
                        rolling_max = cum_returns.cummax()
                        drawdown = (cum_returns / rolling_max) - 1
                        max_drawdown = drawdown.min()

                        st.write("### Additional Metrics")
                        st.write(f"**Monthly Return:** {monthly_return:.2%}")
                        st.write(f"**Max Drawdown:** {max_drawdown:.2%}")

                        # Momentum indicator
                        current = stock['current_price']
                        ma50 = stock['historical_data']['price'].rolling(50).mean().iloc[-1]
                        ma200 = stock['historical_data']['price'].rolling(200).mean().iloc[-1]
                        momentum = "⬆️ Bullish" if current > ma50 > ma200 else "⬇️ Bearish"
                        st.write(f"**Momentum:** {momentum}")

        with tab2:
            st.subheader("Optimal Portfolio Allocation")
            if portfolio:
                portfolio_df = pd.DataFrame(portfolio)
                portfolio_df['Investment (₹)'] = (portfolio_df['percentage'] / 100) * investment_amount
                st.dataframe(
                    portfolio_df.sort_values('percentage', ascending=False)
                    .style.format({
                        'percentage': '{:.2f}%',
                        'expected_return': '{:.4f}',
                        'volatility': '{:.4f}',
                        'Investment (₹)': '₹{:.2f}'
                    }),
                    use_container_width=True
                )

                total_expected_return = (portfolio_df['percentage'] / 100 * portfolio_df['expected_return']).sum()
                total_volatility = (portfolio_df['percentage'] / 100 * portfolio_df['volatility']).sum()

                st.write(f"**Total Expected Return:** {total_expected_return:.4f}")
                st.write(f"**Total Volatility:** {total_volatility:.4f}")
            else:
                st.warning("No portfolio allocation available.")
