import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv(override=True)

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")


@tool
def fetch_ticker_url(ticker: str) -> str:
    """
    Fetches the first finnhub.io URL from news data for a ticker symbol.
    
    Args:
        ticker: Stock ticker symbol (e.g., "TSLA")
    
    Returns:
        A finnhub.io URL string, or None if not found
    """
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/news/{ticker}?limit=1", timeout=10)
        if response.status_code == 200:
            news_data = response.json().get("news", [])
            if news_data and len(news_data) > 0:
                url = news_data[0].get("url", "")
                if "finnhub.io" in url:
                    return url
    except Exception as e:
        pass
    
    return None

