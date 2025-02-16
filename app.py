import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.title("NSE Stock Price Prediction Grid")

# Fixed list of 9 NSE stock tickers (Yahoo Finance format)
tickers = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", 
    "SBIN.NS", "ICICIBANK.NS", "INFY.NS", 
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS"
]

# Sidebar input for the common prediction date
st.sidebar.header("Prediction Date")
prediction_date = st.sidebar.date_input(
    "Select Prediction Date",
    value=datetime.today() + timedelta(days=1)
)

if st.sidebar.button("Run Prediction"):
    prediction_date_dt = prediction_date
    training_end = prediction_date_dt - timedelta(days=30)
    training_start = training_end - timedelta(days=365 * 2)
    testing_start = training_end
    testing_end = prediction_date_dt - timedelta(days=1)

    training_start_str = training_start.strftime("%Y-%m-%d")
    training_end_str = training_end.strftime("%Y-%m-%d")
    testing_start_str = testing_start.strftime("%Y-%m-%d")
    testing_end_str = testing_end.strftime("%Y-%m-%d")

    st.write(f"**Training Period:** {training_start_str} to {training_end_str}")
    st.write(f"**Testing Period:** {testing_start_str} to {testing_end_str}")

    results = []

    # Process each ticker from the fixed list
    for ticker in tickers:
        st.write(f"Processing **{ticker}**...")
        # Download training data
        train_data = yf.download(ticker, start=training_start_str, end=training_end_str)
        if train_data.empty:
            st.error(f"No training data found for {ticker}. Skipping.")
            continue

        # Download testing data
        test_data = yf.download(ticker, start=testing_start_str, end=testing_end_str)
        if test_data.empty:
            st.error(f"No testing data found for {ticker}. Skipping.")
            continue

        # Reset index so Date becomes a column
        train_data = train_data.reset_index()
        test_data = test_data.reset_index()

        # Convert Date to a numeric value using .toordinal()
        train_data["Date_ordinal"] = train_data["Date"].apply(lambda x: x.toordinal())
        test_data["Date_ordinal"] = test_data["Date"].apply(lambda x: x.toordinal())

        # Prepare features and target using the closing price
        X_train = train_data[["Date_ordinal"]]
        y_train = train_data["Close"]

        # Train a simple linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Get the current price from the latest closing price in the testing period
        current_price = float(test_data["Close"].iloc[-1])
        
        # Predict the closing price for the prediction date
        pred_ordinal = np.array([[prediction_date_dt.toordinal()]])
        predicted_price = float(model.predict(pred_ordinal)[0])
        
        # Calculate expected percentage return
        expected_return = ((predicted_price - current_price) / current_price) * 100

        # Create a small plot of training data and regression line for this ticker
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

        # Save the results for display
        results.append({
            "ticker": ticker,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "expected_return": expected_return,
            "figure": fig,
        })

    # Display the results 
    if results:
        st.header("Prediction Results")
        num_cols = 3  
        for i in range(0, len(results), num_cols):
            cols = st.columns(num_cols)
            for j, res in enumerate(results[i: i + num_cols]):
                with cols[j]:
                    st.subheader(res["ticker"])
                    st.write(f"**Current Price:** {res['current_price']:.2f}")
                    st.write(f"**Predicted Price:** {res['predicted_price']:.2f}")
                    st.write(f"**Expected Return:** {res['expected_return']:.2f}%")
                    st.pyplot(res["figure"])
