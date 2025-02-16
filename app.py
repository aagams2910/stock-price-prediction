# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.title("NSE Stock Price Prediction App")

# Sidebar inputs
st.sidebar.header("User Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker (e.g., RELIANCE.NS)", "RELIANCE.NS")
prediction_date = st.sidebar.date_input(
    "Prediction Date (YYYY-MM-DD)",
    value=datetime.today() + timedelta(days=1)
)

if st.sidebar.button("Run Prediction"):
    # Convert prediction date to datetime (if not already)
    prediction_date_dt = prediction_date

    # Define the training and testing periods:
    # Training: ~2 years of data ending 30 days before prediction date
    # Testing: 30 days immediately before the prediction date
    training_end = prediction_date_dt - timedelta(days=30)
    training_start = training_end - timedelta(days=365*2)  # approx two years
    testing_start = training_end
    testing_end = prediction_date_dt - timedelta(days=1)

    # Format dates as strings for yfinance
    training_start_str = training_start.strftime("%Y-%m-%d")
    training_end_str = training_end.strftime("%Y-%m-%d")
    testing_start_str = testing_start.strftime("%Y-%m-%d")
    testing_end_str = testing_end.strftime("%Y-%m-%d")

    st.write(f"**Training Period:** {training_start_str} to {training_end_str}")
    st.write(f"**Testing Period:** {testing_start_str} to {testing_end_str}")

    # Download training data
    st.write("Downloading training data...")
    train_data = yf.download(ticker, start=training_start_str, end=training_end_str)
    if train_data.empty:
        st.error("No training data found for the given period. Please check the ticker and dates.")
        st.stop()
    else:
        st.success("Training data downloaded successfully.")

    # Download testing data
    st.write("Downloading testing data...")
    test_data = yf.download(ticker, start=testing_start_str, end=testing_end_str)
    if test_data.empty:
        st.error("No testing data found for the given period. Please check the ticker and dates.")
        st.stop()
    else:
        st.success("Testing data downloaded successfully.")

    # Reset index to have Date as a column
    train_data = train_data.reset_index()
    test_data = test_data.reset_index()

    # Create a numeric feature from the Date using toordinal()
    train_data['Date_ordinal'] = train_data['Date'].apply(lambda x: x.toordinal())
    test_data['Date_ordinal'] = test_data['Date'].apply(lambda x: x.toordinal())

    # Define features (X) and target (y) using the closing price
    X_train = train_data[['Date_ordinal']]
    y_train = train_data['Close']
    X_test = test_data[['Date_ordinal']]
    y_test = test_data['Close']

    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model on testing data
    y_pred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    st.subheader("Testing Performance")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    # Predict the closing price for the prediction date
    pred_ordinal = np.array([[prediction_date_dt.toordinal()]])
    predicted_price = float(model.predict(pred_ordinal)[0])
    st.subheader("Prediction")
    st.write(f"Predicted closing price for {prediction_date_dt.strftime('%Y-%m-%d')}: **{predicted_price:.2f}**")


    # Plot the training data and regression line
    st.subheader("Training Data & Regression Line")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(train_data['Date'], y_train, label="Training Data", color="blue", s=10)
    # Plot regression line on sorted dates for clarity
    sorted_train = train_data.sort_values('Date')
    ax.plot(sorted_train['Date'], model.predict(sorted_train[['Date_ordinal']]), label="Regression Line", color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price")
    ax.legend()
    st.pyplot(fig)
