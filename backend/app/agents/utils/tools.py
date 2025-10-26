import json
import os
from typing import Any, Dict, List, Union

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


@tool
def fetch_ticker_data(tickers: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Fetches comprehensive financial data for a list of ticker symbols from the FastAPI backend.

    Args:
        tickers: Single ticker string (e.g., "TSLA") or List of ticker symbols (e.g., ["TSLA", "AAPL"])

    Returns:
        Dictionary with ticker symbols as keys, each containing all relevant financial data in nested JSON format.
        Returns error information for tickers that fail to fetch.
    """
    result = {}

    if isinstance(tickers, str):
        ticker_list = [tickers]
    else:
        ticker_list = tickers

    for ticker in ticker_list:
        try:
            ticker_data = _fetch_single_ticker(ticker)
            result[ticker] = ticker_data
        except Exception as e:
            result[ticker] = {
                "error": str(e),
                "status": "failed"
            }

    return result


def _fetch_single_ticker(ticker: str) -> Dict[str, Any]:
    """
    Fetches all available data for a single ticker from FastAPI endpoints.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")

    Returns:
        Nested dictionary containing all financial data for the ticker
    """
    data = {
        "ticker": ticker,
        "company": None,
        "market_data": None,
        "basic_financials": None,
        "earnings_reports": None,
        "earnings_surprises": None,
        "news": None,
        "sec_filings": None,
        "financials_reported": None,
        "insider_sentiment": None
    }

    # Fetch company profile
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/company/{ticker}", timeout=10)
        if response.status_code == 200:
            data["company"] = response.json()
    except Exception as e:
        data["company"] = {"error": str(e)}

    # Fetch market data
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/market-data/{ticker}", timeout=10)
        if response.status_code == 200:
            data["market_data"] = response.json()
    except Exception as e:
        data["market_data"] = {"error": str(e)}

    # Fetch basic financials
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/basic-financials/{ticker}", timeout=10)
        if response.status_code == 200:
            data["basic_financials"] = response.json()
    except Exception as e:
        data["basic_financials"] = {"error": str(e)}

    # Fetch earnings reports
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/earnings/{ticker}", timeout=10)
        if response.status_code == 200:
            data["earnings_reports"] = response.json().get("earnings", [])
    except Exception as e:
        data["earnings_reports"] = {"error": str(e)}

    # Fetch earnings surprises
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/earnings-surprises/{ticker}", timeout=10)
        if response.status_code == 200:
            data["earnings_surprises"] = response.json().get("earnings_surprises", [])
    except Exception as e:
        data["earnings_surprises"] = {"error": str(e)}

    # Fetch news (limited to 20 most recent)
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/news/{ticker}?limit=20", timeout=10)
        if response.status_code == 200:
            data["news"] = response.json().get("news", [])
    except Exception as e:
        data["news"] = {"error": str(e)}

    # Fetch SEC filings
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/filings/{ticker}", timeout=10)
        if response.status_code == 200:
            data["sec_filings"] = response.json().get("filings", [])
    except Exception as e:
        data["sec_filings"] = {"error": str(e)}

    # Fetch financials reported
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/financials-reported/{ticker}", timeout=10)
        if response.status_code == 200:
            data["financials_reported"] = response.json().get("financials_reported", [])
    except Exception as e:
        data["financials_reported"] = {"error": str(e)}

    # Fetch insider sentiment
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/insider-sentiment/{ticker}", timeout=10)
        if response.status_code == 200:
            data["insider_sentiment"] = response.json().get("insider_sentiment", [])
    except Exception as e:
        data["insider_sentiment"] = {"error": str(e)}

    return data


@tool
def fetch_peers(ticker: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Fetch peer tickers for a given symbol from the FastAPI backend.

    Args:
        ticker: Single ticker string (e.g., "TSLA") or single-element list (e.g., ["TSLA"])

    Returns:
        List of peer ticker symbols
    """
    
    if isinstance(ticker, list):
        ticker = ticker[0] if ticker else ""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/peers/{ticker}", timeout=10)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 404:
            return {"ticker": ticker, "peers": []}
        return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


@tool
def web_search(query: str) -> Dict[str, Any]:
    """
    Perform a web search via Tavily API and scrape text content from result pages.

    Input: free-text keyword like "quantum stocks" or "macro events today".
    Output: A dict with keys: query (string) and results (list of objects with url and text).
    """
    try:

        max_results = 5
        tavily_url = "https://api.tavily.com/search"
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "advanced",
            "include_answer": False,
            "max_results": max_results,
        }
        sr = requests.post(tavily_url, json=payload, timeout=30)
        if sr.status_code != 200:
            return {"error": f"Tavily HTTP {sr.status_code}", "details": sr.text}

        data = sr.json()
        links: List[str] = []
        for item in data.get("results", [])[:max_results]:
            url = item.get("url")
            if url and url.startswith("http"):
                links.append(url)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/114.0.0.0 Safari/537.36"
            )
        }
        results = []
        for url in links:
            try:
                page = requests.get(url, headers=headers, timeout=20)
                psoup = BeautifulSoup(page.text, "html.parser")
                text = "\n".join(p.get_text(strip=True) for p in psoup.find_all("p"))
                results.append({"url": url, "text": text[:20000]})
            except Exception as e:
                results.append({"url": url, "error": str(e)})

        return {"query": query, "results": results}
    except Exception as e:
        return {"error": str(e)}


@tool
def fetch_ticker_summary(tickers: List[str]) -> Dict[str, Any]:
    """
    Fetches comprehensive summary data for a list of tickers using the /ticker-summary endpoint.
    This is more efficient than fetch_ticker_data as it makes fewer API calls.

    Args:
        tickers: List of ticker symbols (e.g., ["TSLA", "AAPL"])

    Returns:
        Dictionary with ticker symbols as keys, each containing summary data with latest information
        from all collections (company, market_data, basic_financials, latest_earnings, latest_news,
        recent_filings, insider_sentiment).
    """
    result = {}

    for ticker in tickers:
        try:
            response = requests.get(f"{FASTAPI_BASE_URL}/ticker-summary/{ticker}", timeout=10)
            if response.status_code == 200:
                result[ticker] = response.json()
            else:
                result[ticker] = {
                    "error": f"HTTP {response.status_code}",
                    "status": "failed"
                }
        except Exception as e:
            result[ticker] = {
                "error": str(e),
                "status": "failed"
            }

    return result


# Standalone helper function for non-LangChain usage
def get_ticker_data(tickers: List[str], use_summary: bool = True) -> Dict[str, Any]:
    """
    Helper function to fetch ticker data without LangChain tool decorator.

    Args:
        tickers: List of ticker symbols
        use_summary: If True, uses the efficient /ticker-summary endpoint.
                     If False, fetches all data separately.

    Returns:
        Dictionary with ticker data
    """
    if use_summary:
        return fetch_ticker_summary.invoke({"tickers": tickers})
    else:
        return fetch_ticker_data.invoke({"tickers": tickers})


if __name__ == "__main__":
    # Example usage
    test_tickers = ["PDLMF", "PLSE"]

    # print("=" * 80)
    # print("TESTING fetch_ticker_summary")
    # print("=" * 80)
    # summary_data = fetch_ticker_summary.invoke({"tickers": test_tickers})
    # print(json.dumps(summary_data, indent=2, default=str))

    print("\n" + "=" * 80)
    print("TESTING fetch_ticker_data (comprehensive)")
    print("=" * 80)
    full_data = fetch_ticker_data.invoke({"tickers": ["PDLMF"]})
    print(json.dumps(full_data, indent=2, default=str))
