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

def format_and_display_news(news_data):
    formatted_news = []
    # Format and display the news headline and sentiment polarity
    for article in news_data:
        formatted_article = {
            "title": article.get("title", "No title"),
            # "summary": article.get("summary", "No summary"),
            "url": article.get("link", "#"),
            "polarity": article.get("polarity", 0),
            "subjectivity": article.get("subjectivity", 0),
            "Date": article.get("publishedDate")
        }
        formatted_news.append(formatted_article)
    return formatted_news
