import requests
from textblob import TextBlob


def fetch_news(api_key, ticker):
    # Fetch news data from Alpha Vantage
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}"
    # Make a GET request to the URL
    response = requests.get(url)
    # Parse the response
    news_data = response.json()
    return news_data['feed']

def analyze_news_sentiment(news_data):
    sentiments = []
    # Analyze the sentiment of each news headline using TextBlob
    for article in news_data:
        blob = TextBlob(article['summary'])
        article['polarity'] = blob.sentiment.polarity
        article['subjectivity'] = blob.sentiment.subjectivity
        sentiments.append(blob.sentiment.polarity)
    return sentiments