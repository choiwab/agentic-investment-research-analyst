import os
import json
import requests
from typing import List, Dict, Any
from langchain.tools import tool

# FastAPI base URL - defaults to localhost, override with env var
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")


@tool
def fetch_ticker_data(tickers: List[str]) -> Dict[str, Any]:
    """
    Fetches comprehensive financial data for a list of ticker symbols from the FastAPI backend.

    Args:
        tickers: List of ticker symbols (e.g., ["TSLA", "AAPL"])

    Returns:
        Dictionary with ticker symbols as keys, each containing all relevant financial data in nested JSON format.
        Returns error information for tickers that fail to fetch.
    """
    result = {}

    for ticker in tickers:
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


@tool
def filter_by_timeframe(data: Dict[str, Any], year: int = None, quarter: int = None) -> Dict[str, Any]:
    """
    Filters financial data by specific timeframe (year and/or quarter).

    Args:
        data: Dictionary containing ticker financial data
        year: Target year (e.g., 2024)
        quarter: Target quarter (1-4)

    Returns:
        Filtered data matching the specified timeframe
    """
    filtered = {}

    # Filter earnings reports
    if data.get("earnings_reports"):
        filtered["earnings_reports"] = [
            e for e in data["earnings_reports"]
            if (year is None or e.get("year") == year) and
               (quarter is None or e.get("quarter") == quarter)
        ]

    # Filter earnings surprises
    if data.get("earnings_surprises"):
        filtered["earnings_surprises"] = [
            e for e in data["earnings_surprises"]
            if (year is None or e.get("year") == year) and
               (quarter is None or e.get("quarter") == quarter)
        ]

    # Filter financials reported
    if data.get("financials_reported"):
        filtered["financials_reported"] = [
            f for f in data["financials_reported"]
            if (year is None or f.get("year") == year) and
               (quarter is None or f.get("quarter") == quarter)
        ]

    # Filter insider sentiment by year
    if data.get("insider_sentiment") and year:
        filtered["insider_sentiment"] = [
            i for i in data["insider_sentiment"]
            if i.get("year") == year
        ]

    # Keep non-time-series data as-is
    for key in ["company", "market_data", "basic_financials", "news", "sec_filings"]:
        if data.get(key):
            filtered[key] = data[key]

    return filtered


def calculate_growth_metrics(current_value: float, previous_value: float) -> float:
    """
    Calculate year-over-year or quarter-over-quarter growth rate.

    Args:
        current_value: Current period value
        previous_value: Previous period value

    Returns:
        Growth rate as decimal (e.g., 0.05 for 5% growth)
    """
    if previous_value is None or previous_value == 0:
        return None
    return (current_value - previous_value) / previous_value


def detect_anomalies(data: Dict[str, Any], threshold: float = 0.20) -> Dict[str, List[str]]:
    """
    Detect significant anomalies in financial data.

    Args:
        data: Dictionary containing ticker financial data
        threshold: Threshold for flagging anomalies (default 20%)

    Returns:
        Dictionary with 'positive_anomalies' and 'negative_anomalies' lists
    """
    anomalies = {
        "positive_anomalies": [],
        "negative_anomalies": []
    }

    # Check earnings surprises
    if data.get("earnings_surprises") and isinstance(data["earnings_surprises"], list):
        for surprise in data["earnings_surprises"]:
            if not isinstance(surprise, dict):
                continue
            surprise_pct = surprise.get("surprisePercent")
            if surprise_pct is not None:
                if surprise_pct > threshold * 100:
                    anomalies["positive_anomalies"].append(
                        f"EPS beat by {surprise_pct:.1f}% in {surprise.get('period', 'unknown period')}"
                    )
                elif surprise_pct < -threshold * 100:
                    anomalies["negative_anomalies"].append(
                        f"EPS miss by {abs(surprise_pct):.1f}% in {surprise.get('period', 'unknown period')}"
                    )

    # Check insider sentiment
    if data.get("insider_sentiment") and isinstance(data["insider_sentiment"], list):
        for sentiment in data["insider_sentiment"]:
            if not isinstance(sentiment, dict):
                continue
            mspr = sentiment.get("mspr")
            if mspr is not None and mspr < -0.5:
                anomalies["negative_anomalies"].append(
                    f"Heavy insider selling detected (MSPR: {mspr:.2f}) in {sentiment.get('year')}-{sentiment.get('month'):02d}"
                )
            elif mspr is not None and mspr > 0.5:
                anomalies["positive_anomalies"].append(
                    f"Strong insider buying detected (MSPR: {mspr:.2f}) in {sentiment.get('year')}-{sentiment.get('month'):02d}"
                )

    # Check valuation metrics
    if (data.get("basic_financials") and
        isinstance(data["basic_financials"], dict) and
        data["basic_financials"].get("metric")):
        metrics = data["basic_financials"]["metric"]
        pe_ratio = metrics.get("peBasicExclExtraTTM")

        if pe_ratio is not None and pe_ratio != "N/A":
            try:
                pe_ratio = float(pe_ratio)
                if pe_ratio > 50:
                    anomalies["negative_anomalies"].append(
                        f"Elevated P/E ratio of {pe_ratio:.1f} suggests stretched valuation"
                    )
                elif pe_ratio < 10 and pe_ratio > 0:
                    anomalies["positive_anomalies"].append(
                        f"Low P/E ratio of {pe_ratio:.1f} may indicate undervaluation"
                    )
            except (ValueError, TypeError):
                pass  # Skip if can't convert to float

    return anomalies


def normalize_financial_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and clean financial data to ensure consistent units and handle missing values.

    Args:
        data: Raw ticker financial data

    Returns:
        Normalized data with consistent formatting
    """
    normalized = data.copy()

    # Normalize revenue to millions (if in different units)
    if normalized.get("earnings_reports") and isinstance(normalized["earnings_reports"], list):
        for report in normalized["earnings_reports"]:
            # Skip if report is not a dict (e.g., error message)
            if not isinstance(report, dict):
                continue
            # Ensure revenue is in consistent units (millions)
            if report.get("revenueActual"):
                # Assume values > 1B are in dollars, convert to millions
                if report["revenueActual"] > 1_000_000_000:
                    report["revenueActual"] = report["revenueActual"] / 1_000_000
            if report.get("revenueEstimate"):
                if report["revenueEstimate"] > 1_000_000_000:
                    report["revenueEstimate"] = report["revenueEstimate"] / 1_000_000

    # Handle missing values in basic financials
    if normalized.get("basic_financials") and isinstance(normalized["basic_financials"], dict):
        if normalized["basic_financials"].get("metric"):
            metrics = normalized["basic_financials"]["metric"]
            # Replace None/null with 'N/A' for display purposes
            for key, value in metrics.items():
                if value is None:
                    metrics[key] = "N/A"

    # Ensure market data has all required fields
    if normalized.get("market_data") and isinstance(normalized["market_data"], dict):
        required_fields = ["c", "h", "l", "o", "pc"]
        for field in required_fields:
            if field not in normalized["market_data"]:
                normalized["market_data"][field] = None

    return normalized


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
