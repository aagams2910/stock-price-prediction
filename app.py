import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="NSE Stock Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("NSE Stock Price Prediction Grid")

# Fixed list of 9 NSE stock tickers (Yahoo Finance format)
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", 
    "SBIN.NS", "ICICIBANK.NS", "INFY.NS", 
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS"
]

# Sidebar for user inputs
st.sidebar.header("Prediction Settings")
prediction_date = st.sidebar.date_input(
    "Select Prediction Date",
    value=datetime.today() + timedelta(days=1)
)

@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    """Download stock data using yfinance with caching."""
    return yf.download(ticker, start=start_date, end=end_date)

def run_prediction(pred_date, tickers):
    """Run predictions for the provided tickers and prediction date."""
    # Define periods
    training_end = pred_date - timedelta(days=30)
    training_start = training_end - timedelta(days=365 * 2)
    testing_start = training_end
    testing_end = pred_date - timedelta(days=1)

    # Format dates as strings for yfinance
    training_start_str = training_start.strftime("%Y-%m-%d")
    training_end_str = training_end.strftime("%Y-%m-%d")
    testing_start_str = testing_start.strftime("%Y-%m-%d")
    testing_end_str = testing_end.strftime("%Y-%m-%d")

    st.write(f"**Training Period:** {training_start_str} to {training_end_str}")
    st.write(f"**Testing Period:** {testing_start_str} to {testing_end_str}")

    results = []

    # Optional: add a progress bar
    progress_bar = st.progress(0)
    total = len(tickers)
    
    for idx, ticker in enumerate(tickers, start=1):
        st.write(f"Processing **{ticker}**...")
        # Download training and testing data using caching
        train_data = get_stock_data(ticker, training_start_str, training_end_str)
        test_data = get_stock_data(ticker, testing_start_str, testing_end_str)
        
        if train_data.empty or test_data.empty:
            st.error(f"No sufficient data for {ticker}. Skipping.")
            progress_bar.progress(idx / total)
            continue

        # Prepare data
        train_data = train_data.reset_index()
        test_data = test_data.reset_index()
        train_data["Date_ordinal"] = train_data["Date"].apply(lambda x: x.toordinal())
        test_data["Date_ordinal"] = test_data["Date"].apply(lambda x: x.toordinal())

        X_train = train_data[["Date_ordinal"]]
        y_train = train_data["Close"]

        # Train the model
        model = LinearRegression()
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            st.error(f"Error training model for {ticker}: {e}")
            progress_bar.progress(idx / total)
            continue

        # Get the current price and predict for the prediction date
        current_price = float(test_data["Close"].iloc[-1])
        pred_ordinal = np.array([[pred_date.toordinal()]])
        predicted_price = float(model.predict(pred_ordinal)[0])
        expected_return = ((predicted_price - current_price) / current_price) * 100

        # Create a plot
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.scatter(train_data["Date"], y_train, label="Training Data", s=5)
        sorted_train = train_data.sort_values("Date")
        ax.plot(
            sorted_train["Date"],
            model.predict(sorted_train[["Date_ordinal"]]),
            label="Regression Line",
            color="red",
        )
        ax.set_title(ticker, fontsize=8)
        ax.tick_params(labelsize=6)
        plt.tight_layout()

        results.append({
            "ticker": ticker,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "expected_return": expected_return,
            "figure": fig,
        })
        
        progress_bar.progress(idx / total)

    return results

if st.sidebar.button("Run Prediction"):
    with st.spinner("Running predictions..."):
        prediction_date_dt = prediction_date  # Already a datetime.date object
        results = run_prediction(prediction_date_dt, TICKERS)

    if results:
        st.header("Prediction Results")
        num_cols = 3  # Display in a 3-column grid
        for i in range(0, len(results), num_cols):
            cols = st.columns(num_cols)
            for j, res in enumerate(results[i: i + num_cols]):
                with cols[j]:
                    st.subheader(res["ticker"])
                    st.write(f"**Current Price:** {res['current_price']:.2f}")
                    st.write(f"**Predicted Price:** {res['predicted_price']:.2f}")
                    st.write(f"**Expected Return:** {res['expected_return']:.2f}%")
                    st.pyplot(res["figure"])
