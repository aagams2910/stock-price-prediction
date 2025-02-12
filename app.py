from flask import Flask, render_template
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

app = Flask(__name__)

def fetch_stock_data():
    ticker = "RELIANCE.NS"
    try:
        print(f"Fetching historical stock prices for {ticker}...")
        stock_data = yf.download(ticker, start="2020-01-01", end="2023-10-01")
        if stock_data.empty:
            return f"No data found for {ticker} in the specified date range."
        return stock_data.head().to_html()
    except Exception as e:
        return f"Error fetching stock data: {str(e)}"

def fetch_news_headlines():
    url = "https://www.businesstoday.in/latest/economy"
    try:
        print(f"\nScraping financial news headlines from {url}...")
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        headlines = []
        for headline in soup.select("h2 a"):
            headlines.append(headline.text.strip())

        return headlines[:5]  # Return top 5 headlines
    except Exception as e:
        return [f"Error fetching news headlines: {str(e)}"]

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
    # Fetch data
    stock_data_html = fetch_stock_data()
    headlines = fetch_news_headlines()
    sentiment_results = analyze_sentiment(headlines)

    # Render the data in the template
    return render_template("index.html", 
                           stock_data=stock_data_html, 
                           headlines=headlines, 
                           sentiment_results=sentiment_results)

if __name__ == "__main__":
    app.run(debug=True)