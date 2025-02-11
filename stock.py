import yfinance as yf
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

ticker = "RELIANCE.NS" 
print(f"Fetching historical stock prices for {ticker}...")
stock_data = yf.download(ticker, start="2020-01-01", end="2023-10-01")
print("Historical Stock Data:")
print(stock_data.head())

url = "https://www.businesstoday.in/latest/economy"  
print(f"\nScraping financial news headlines from {url}...")
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

headlines = []
for headline in soup.select("h2 a"): 
    headlines.append(headline.text.strip())

print("\nLatest Financial News Headlines:")
for i, headline in enumerate(headlines[:5], 1):  
    print(f"{i}. {headline}")

print("\nSentiment Analysis of News Headlines:")
for i, headline in enumerate(headlines[:5], 1): 
    blob = TextBlob(headline)
    sentiment = blob.sentiment
    print(f"{i}. Headline: {headline}")
    print(f"   Sentiment Polarity: {sentiment.polarity:.2f} (Range: -1 to 1)")
    print(f"   Sentiment Subjectivity: {sentiment.subjectivity:.2f} (Range: 0 to 1)\n")