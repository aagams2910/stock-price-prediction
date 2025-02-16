import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.title("NSE Stock Price Prediction Grid")

st.sidebar.header("User Input Parameters")
# Enter tickers as a comma-separated list.
# For NSE stocks, use Yahoo Finance tickers (e.g., TCS.NS, INFY.NS, etc.)
tickers_input = st.sidebar.text_input(
    "Stock Tickers (comma separated)",
    "RELIANCE.NS, TCS.NS, HDFC.NS, SBIN.NS, ICICIBANK.NS, INFY.NS, LT.NS, AXISBANK.NS, BAJFINANCE.NS"
)
# Common prediction date for all stocks
prediction_date = st.sidebar.date_input(
    "Prediction Date",
    value=datetime.today() + timedelta(days=1)
)

if st.sidebar.button("Run Prediction"):
    # Parse ticker list (strip whitespace)
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        st.error("Please provide at least one ticker.")
        st.stop()

    # Define periods based on prediction date:
    # - Training period: ~2 years ending 30 days before prediction date.
    # - Testing period: the 30 days before the prediction date.
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

    # Process each ticker individually
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

        # Create a numeric feature from the Date (using .toordinal())
        train_data["Date_ordinal"] = train_data["Date"].apply(lambda x: x.toordinal())
        test_data["Date_ordinal"] = test_data["Date"].apply(lambda x: x.toordinal())

        # Prepare features (X) and target (y) using the closing price
        X_train = train_data[["Date_ordinal"]]
        y_train = train_data["Close"]
        X_test = test_data[["Date_ordinal"]]
        y_test = test_data["Close"]

        # Train a simple linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate the model on testing data
        y_pred_test = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)

        # Predict the closing price for the prediction date
        pred_ordinal = np.array([[prediction_date_dt.toordinal()]])
        predicted_price = float(model.predict(pred_ordinal)[0])

        # Create a plot for the training data and regression line (smaller figure for grid)
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

        # Store result for this ticker
        results.append({
            "ticker": ticker,
            "predicted_price": predicted_price,
            "mse": mse,
            "r2": r2,
            "figure": fig,
        })

    # Display the results in a grid layout: 3 columns per row (3 x 3 grid for 9 stocks)
    if results:
        st.header("Prediction Results")
        num_cols = 3  # change as needed
        for i in range(0, len(results), num_cols):
            cols = st.columns(num_cols)
            for j, res in enumerate(results[i : i + num_cols]):
                with cols[j]:
                    st.subheader(res["ticker"])
                    st.write(f"**Predicted Price:** {res['predicted_price']:.2f}")
                    st.write(f"**MSE:** {res['mse']:.2f}")
                    st.write(f"**R²:** {res['r2']:.2f}")
                    st.pyplot(res["figure"])
