# Stock Price Prediction with Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![yfinance](https://img.shields.io/badge/yfinance-latest-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)

## Overview

This project implements an AI-powered stock price prediction and portfolio optimization system using machine learning models and sentiment analysis. The application provides predictions for Indian stock market equities (NSE) and recommends optimal portfolio allocations based on risk tolerance.

## Features

- **Stock Price Prediction**: Uses multiple ML models including Random Forest, Gradient Boosting, and Linear Regression
- **Portfolio Optimization**: Optimizes asset allocation based on expected returns, volatility, and user-defined risk tolerance
- **Interactive Dashboard**: Built with Streamlit for easy visualization and interaction
- **Risk Analysis**: Calculates volatility, Sharpe ratio, and expected returns for each stock
- **Customizable Parameters**: Adjust prediction date, risk tolerance, and investment amount

## Models

The application offers several prediction models:
- **Quantum Ensemble**: A custom ensemble model combining multiple prediction techniques
- **Random Forest Regressor**: With hyperparameter optimization
- **Gradient Boosting Regressor**: For high-accuracy predictions
- **Linear Regression**: As a baseline model

## Installation

```bash
# Clone the repository
git clone https://github.com/aagams2910/stock-price-prediction.git
cd stock-price-prediction

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the Streamlit app
streamlit run app.py
```

## Requirements

- Python 3.7+
- streamlit
- yfinance
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

## Project Structure

- `app.py`: Main application file with Streamlit interface
- `portfolio_optimizer.py`: Portfolio optimization algorithms
- `app.log`: Application logs

## How It Works

1. **Data Collection**: Fetches historical stock data using yfinance
2. **Feature Engineering**: Calculates technical indicators like moving averages
3. **Model Training**: Trains selected ML model on historical data
4. **Prediction**: Generates price predictions for specified date
5. **Portfolio Optimization**: Calculates optimal asset allocation based on predictions and risk tolerance


## Future Improvements

- Incorporate sentiment analysis from news and social media
- Add more technical indicators for better prediction accuracy
- Implement backtesting functionality
- Support for international markets

