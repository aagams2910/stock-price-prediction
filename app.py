from flask import Flask, render_template
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# List of Nifty 50 and Sensex 30 companies (you can expand this list)
NIFTY_50_TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BPCL.NS",
    "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS",
    "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SHRIRAMFIN.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS"
]

SENSEX_30_TICKERS = [
    "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJFINANCE.NS", "BHARTIARTL.NS", "HCLTECH.NS",
    "HDFCBANK.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS",
    "ITC.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS",
    "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "BAJAJFINSV.NS", "ADANIPORTS.NS"
]

def predict_stock_price(data):
    """Simple linear regression prediction for next 5 days"""
    try:
        # Create features (days) and target (close price)
        df = data[['Close']].reset_index()
        df['days'] = (df['Date'] - df['Date'].min()).dt.days
        
        # Train linear regression model
        X = np.array(df['days']).reshape(-1, 1)
        y = np.array(df['Close'])
        model = LinearRegression().fit(X, y)
        
        # Predict next 5 days
        future_days = np.array([X[-1][0] + 1, X[-1][0] + 2, X[-1][0] + 3, 
                              X[-1][0] + 4, X[-1][0] + 5]).reshape(-1, 1)
        predictions = model.predict(future_days)
        
        # Create prediction DataFrame
        last_date = df['Date'].max()
        dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=5)
        return pd.DataFrame({
            'Date': dates,
            'Predicted Close': predictions.round(2)
        }).set_index('Date')
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return pd.DataFrame()

def fetch_stock_data(tickers):
    stock_data = {}
    for ticker in tickers:
        try:
            print(f"Fetching data for {ticker}...")
            data = yf.download(ticker, start="2023-01-01", end=pd.Timestamp.today().strftime('%Y-%m-%d'))
            if not data.empty:
                prediction = predict_stock_price(data)
                stock_data[ticker] = {
                    'historical': data.tail(5).reset_index().to_dict('records'),
                    'prediction': prediction.reset_index().to_dict('records')
                }
            else:
                stock_data[ticker] = {'error': f"No data for {ticker}"}
        except Exception as e:
            stock_data[ticker] = {'error': str(e)}
    return stock_data

def fetch_news_headlines(query):
    url = f"https://www.businesstoday.in/search/{query.replace(' ', '%20')}"
    try:
        print(f"\nScraping financial news headlines for {query} from {url}...")
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        headlines = []
        for headline in soup.select("h2 a"):
            headlines.append(headline.text.strip())

        return headlines[:5]  # Return top 5 headlines
    except Exception as e:
        return [f"Error fetching news headlines for {query}: {str(e)}"]

def analyze_sentiment(headlines):
    sentiment_results = []
    for headline in headlines:
        blob = TextBlob(headline)
        sentiment = blob.sentiment
        sentiment_results.append({
            "headline": headline,
            "polarity": sentiment.polarity,
            "subjectivity": sentiment.subjectivity
        })
    return sentiment_results

@app.route("/")
def home():
    try:
        # Fetch stock data for Nifty 50 and Sensex 30 companies
        print("Fetching Nifty 50 stock data...")
        nifty_stock_data = fetch_stock_data(NIFTY_50_TICKERS)
        print("Nifty 50 data:", nifty_stock_data)  # Debugging

        print("Fetching Sensex 30 stock data...")
        sensex_stock_data = fetch_stock_data(SENSEX_30_TICKERS)
        print("Sensex 30 data:", sensex_stock_data)  # Debugging

        # Fetch news headlines for each company
        print("Fetching news headlines...")
        nifty_news = {}
        sensex_news = {}

        for ticker in NIFTY_50_TICKERS:
            query = ticker.replace(".NS", "")  # Remove .NS for news search
            nifty_news[ticker] = fetch_news_headlines(query)

        for ticker in SENSEX_30_TICKERS:
            query = ticker.replace(".NS", "")  # Remove .NS for news search
            sensex_news[ticker] = fetch_news_headlines(query)

        # Analyze sentiment for the fetched headlines
        print("Analyzing sentiment...")
        nifty_sentiment = {ticker: analyze_sentiment(headlines) for ticker, headlines in nifty_news.items()}
        sensex_sentiment = {ticker: analyze_sentiment(headlines) for ticker, headlines in sensex_news.items()}

        # Render the data in the template
        print("Rendering template...")
        return render_template("index.html", 
                              nifty_stock_data=nifty_stock_data, 
                              sensex_stock_data=sensex_stock_data,
                              nifty_news=nifty_news,
                              sensex_news=sensex_news,
                              nifty_sentiment=nifty_sentiment,
                              sensex_sentiment=sensex_sentiment)
    
    except Exception as e:
        print(f"Error in home function: {str(e)}")
        return render_template("error.html", error=str(e))
if __name__ == "__main__":
    app.run(debug=True)