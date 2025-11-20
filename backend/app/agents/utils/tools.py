import json
import os
from typing import Any, Dict, List, Union
from yahooquery import Ticker
import pandas as pd

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv(override=True)

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


def _should_use_alpha_vantage_fallback(data: Dict[str, Any]) -> bool:
    """
    Determine if Alpha Vantage fallback should be used based on MongoDB data quality.

    Args:
        data: Data dictionary returned from MongoDB fetch attempts

    Returns:
        True if fallback is needed, False otherwise
    """
    # Critical fields that must be present for meaningful analysis
    critical_fields = ["company", "basic_financials", "market_data"]

    for field in critical_fields:
        # Check if field is None
        if data.get(field) is None:
            return True

        # Check if field contains an error dict
        if isinstance(data.get(field), dict) and data[field].get("error"):
            return True

    return False

def _is_alpha_vantage_low_quality(data: Dict[str, Any]) -> bool:
    """
    Returns True if Alpha Vantage data is incomplete and Yahoo fallback should be used.
    """
    if not data:
        return True

    missing_score = 0

    # --- Company ---
    company = data.get("company")
    if not company or not company.get("industry") or not company.get("description"):
        missing_score += 1

    # --- Market Data ---
    m = data.get("market_data") or {}
    if not m.get("c") or not m.get("h") or not m.get("l"):
        missing_score += 1
    if not m.get("volume"):
        missing_score += 1

    # --- Basic Financials ---
    bf = data.get("basic_financials") or {}
    metric = bf.get("metric") or {}

    critical_metrics = [
        "peBasicExclExtraTTM",
        "pbRatio",
        "psTTM",
        "52WeekHigh",
        "52WeekLow",
        "beta",
    ]
    for key in critical_metrics:
        if metric.get(key) is None:
            missing_score += 1

    # --- Earnings ---
    if not data.get("earnings_reports"):
        missing_score += 1
    
    # ---Insider Sentiment---
    ins = data.get("insider_sentiment")
    if not ins or ins.get("change") in (None, "", "N/A") or ins.get("mspr") in (None, "N/A"):
        missing_score += 1

    # Threshold: if >= 3 weaknesses, AV is low-quality
    return missing_score >= 3

def _should_use_yahoo_fallback(data: Dict[str, Any]) -> bool:
    """
    Use Yahoo fallback if Alpha Vantage is missing *any* important fields,
    even if the outer dict exists.
    """
    if not data:
        return True
    
    # First Condition:
    if _is_alpha_vantage_low_quality(data):
        return True

    # --- Company ---
    comp = data.get("company") or {}
    if not comp.get("industry") or not comp.get("description"):
        return True
    if comp.get("marketCapitalization") in (None, 0):
        return True

    # --- Market ---
    m = data.get("market_data") or {}
    market_required = ["c", "h", "l", "o", "pc", "volume"]
    for k in market_required:
        if m.get(k) in (None, 0):
            return True

    # --- Financial Metrics ---
    bf = data.get("basic_financials") or {}
    metric = bf.get("metric") or {}
    financial_required = [
        "peBasicExclExtraTTM",
        "pbRatio",
        "psTTM",
        "52WeekHigh",
        "52WeekLow",
        "beta",
    ]
    for k in financial_required:
        if metric.get(k) is None:
            return True

    # --- Earnings Reports ---
    if not data.get("earnings_reports") or len(data["earnings_reports"]) < 2:
        return True

    # --- Insider sentiment ---
    ins = data.get("insider_sentiment") or {}
    if not ins or ins.get("change") in (None, "", "N/A") or ins.get("mspr") in (None, "N/A"):
        return True

    return False

def _merge_alpha_vantage_and_yahoo_finance(av_data: Dict[str, Any], yahoo_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge: Use Yahoo to fill missing fields inside nested dicts/lists.
    This version correctly handles:
      • list of dicts (earnings_reports, earnings_surprises)
      • nested fields inside each element
      • merging Yahoo values only into missing AV fields
    """
    result = av_data.copy()

    for key, yahoo_value in yahoo_data.items():
        if key == "ticker":
            continue

        av_value = result.get(key)
        result[key] = _deep_merge_value(av_value, yahoo_value, path=key)

    return result


def _deep_merge_value(av_value, yahoo_value, path=""):
    """
    Recursively merge Yahoo into AV:
      - Fill missing AV values only
      - Preserve valid AV values
      - Merge lists intelligently (list of dicts especially)
    """

    # Edge condition: Always prefer Yahoo for insider_sentiment if it has real data
    if path.startswith("insider_sentiment"):
        if yahoo_value not in (None, [], {}, "N/A"):
            print(f"[FALLBACk] Yahoo overriding insider sentiment at '{path}'")
            return yahoo_value
        
        return av_value
    # 0. AV missing → use Yahoo
    if av_value is None:
        print(f"[FALLBACK] Yahoo filled missing '{path}'")
        return yahoo_value

    # 1. AV error dict → replace
    if isinstance(av_value, dict) and av_value.get("error"):
        print(f"[FALLBACK] Yahoo replaced error field '{path}'")
        return yahoo_value

    # 2. Merge dict → recurse into fields
    if isinstance(av_value, dict) and isinstance(yahoo_value, dict):
        merged = av_value.copy()
        for k, yv in yahoo_value.items():
            merged[k] = _deep_merge_value(
                av_value.get(k),
                yv,
                path=f"{path}.{k}"
            )
        return merged

    # 3. Merge lists
    if isinstance(av_value, list) and isinstance(yahoo_value, list):

        # Empty AV list → use complete Yahoo list
        if len(av_value) == 0 and len(yahoo_value) > 0:
            print(f"[FALLBACK] Yahoo filled missing list '{path}'")
            return yahoo_value

        # Case: list of dicts (earnings, financials)
        if all(isinstance(i, dict) for i in av_value) and all(isinstance(i, dict) for i in yahoo_value):
            merged_list = []

            max_len = max(len(av_value), len(yahoo_value))
            for i in range(max_len):

                av_item = av_value[i] if i < len(av_value) else None
                yahoo_item = yahoo_value[i] if i < len(yahoo_value) else None

                merged_item = _deep_merge_value(
                    av_item,
                    yahoo_item,
                    path=f"{path}[{i}]"
                )
                merged_list.append(merged_item)

            return merged_list

        # Fallback: AV list valid → keep AV
        return av_value

    # 4. Primitive type handling
    # Fill only if AV is None or empty string
    if av_value in (None, "", "N/A"):
        print(f"[FALLBACK] Yahoo filled missing '{path}'")
        return yahoo_value

    return av_value


def _merge_mongodb_and_alpha_vantage(mongodb_data: Dict[str, Any], av_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Intelligently merge MongoDB and Alpha Vantage data, preferring MongoDB when available.

    Strategy:
    - For each field, use MongoDB data if it exists and is valid
    - Use Alpha Vantage data only if MongoDB field is None or contains error
    - Preserve the ticker field from original data

    Args:
        mongodb_data: Data from MongoDB (may have missing/error fields)
        av_data: Data from Alpha Vantage (transformed to MongoDB format)

    Returns:
        Merged dictionary with best available data from both sources
    """
    result = mongodb_data.copy()

    # Merge each top-level field
    for key in av_data:
        # Skip ticker field (already set)
        if key == "ticker":
            continue

        mongodb_value = result.get(key)

        # Use Alpha Vantage data if:
        # 1. MongoDB value is None
        # 2. MongoDB value is an error dict
        # 3. MongoDB value is an empty list and AV has data
        should_use_av = False

        if mongodb_value is None:
            should_use_av = True
        elif isinstance(mongodb_value, dict) and mongodb_value.get("error"):
            should_use_av = True
        elif isinstance(mongodb_value, list) and len(mongodb_value) == 0 and isinstance(av_data[key], list) and len(av_data[key]) > 0:
            should_use_av = True

        if should_use_av:
            result[key] = av_data[key]
            print(f"[FALLBACK] Using Alpha Vantage data for field '{key}'")

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

    # ============================================================================
    # Alpha Vantage Fallback Strategy
    # ============================================================================
    # Check if we need to use Alpha Vantage as fallback for missing MongoDB data
    if _should_use_alpha_vantage_fallback(data):
        print(f"[INFO] MongoDB data incomplete for {ticker}, attempting Alpha Vantage fallback")

        # Get Alpha Vantage API key from environment
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

        if api_key:
            try:
                # Fetch data from all Alpha Vantage endpoints
                print(f"[FALLBACK] Fetching Alpha Vantage data for {ticker}")
                overview = _fetch_alpha_vantage_overview(ticker, api_key)
                earnings = _fetch_alpha_vantage_earnings(ticker, api_key)
                quote = _fetch_alpha_vantage_quote(ticker, api_key)

                # Check if all Alpha Vantage endpoints returned errors
                all_failed = all(
                    (not resp) or isinstance(resp, dict) and resp.get("error")
                    for resp in [overview, earnings, quote]
                )

                if all_failed:
                    print(f"[ERROR] All Alpha Vantage endpoints failed for {ticker}")
                    # Check if ticker not found specifically
                    if any(resp.get("error") == "ticker_not_found" for resp in [overview, earnings, quote]):
                        data["_fallback_status"] = "ticker_not_found"
                        print(f"[ERROR] Ticker {ticker} not found in Alpha Vantage")
                    else:
                        data["_fallback_status"] = "api_error"
                        print(f"[WARNING] Alpha Vantage API errors for {ticker}, using MongoDB data only")
                else:
                    # Transform Alpha Vantage data to MongoDB format
                    av_data = _transform_av_to_mongodb_format(ticker, overview, earnings, quote)

                    # Merge with MongoDB data (prefer MongoDB when available)
                    data = _merge_mongodb_and_alpha_vantage(data, av_data)

                    # Validate that critical fields are now populated
                    still_missing = _should_use_alpha_vantage_fallback(data)
                    if still_missing:
                        print(f"[WARNING] Alpha Vantage fallback incomplete for {ticker}, some critical fields still missing")
                        data["_fallback_status"] = "partial_success"
                    else:
                        print(f"[FALLBACK] Successfully merged Alpha Vantage data for {ticker}")
                        data["_fallback_status"] = "success"

            except Exception as e:
                print(f"[ERROR] Alpha Vantage fallback failed for {ticker}: {e}")
                data["_fallback_status"] = "exception"
                # Continue with MongoDB data even if fallback fails
        else:
            print(f"[WARNING] ALPHA_VANTAGE_API_KEY not set, cannot use fallback for {ticker}")
            data["_fallback_status"] = "no_api_key"

    # ============================================================================
    # Yahoo Finance API Fallback Strategy - Second-Level
    # ============================================================================
    if _should_use_yahoo_fallback(data):
        print(f"[INFO] Alpha Vantage data incomplete or low-quality for {ticker}, attempting Yahoo Finance fallback")

        try:
            yahoo_overview = _fetch_yahoo_overview(ticker)
            yahoo_earnings = _fetch_yahoo_earnings(ticker)
            yahoo_quote = _fetch_yahoo_quote(ticker)
            yahoo_sentiment = _fetch_yahoo_insider_sentiment(ticker)

            all_failed = all(
                (not resp) or isinstance(resp, dict) and resp.get("error")
                for resp in [yahoo_overview, yahoo_earnings, yahoo_quote, yahoo_sentiment]
            )

            if all_failed:
                print(f"[ERROR] All Yahoo Finance endpoints failed for {ticker}")
                # Check if ticker not found specifically
                data["_yahoo_fallback_status"] = "api_error"
            else:
                yahoo_data = _transform_yahoo_to_mongodb_format(ticker, yahoo_overview, yahoo_earnings, yahoo_quote, yahoo_sentiment)
                data = _merge_alpha_vantage_and_yahoo_finance(data, yahoo_data)

                # Validate that critical fields are now populated
                still_missing = _should_use_yahoo_fallback(data)
                if still_missing:
                    print(f"[WARNING] Yahoo Finance fallback incomplete for {ticker}, some critical fields still missing")
                    data["_fallback_status_yahoo"] = "partial_success"
                else:
                    print(f"[FALLBACK] Successfully merged Yahoo Finance data for {ticker}")
                    data["_fallback_status_yahoo"] = "success"

        except Exception as e:
            print(f"[ERROR] Yahoo Finance fallback failed for {ticker}: {e}")
            data["_fallback_status_yahoo"] = "exception"

    return data    

## Yahoo Finance Fallback Functions

def _fetch_yahoo_overview(ticker: str) -> Dict[str, Any]:
    """
    Fetch overview data from Yahoo Finance.
    Returns:
        {
            "summary": {...},
            "profile": {...},
            "financial": {...},
            "price": {...}
        }
    Or standardized error: {"error": "...", "details": "..."}
    """
    try:
        t = Ticker(ticker)

        # Yahooquery stores errors in .symbols = None or empty
        if not t.symbols or ticker.upper() not in t.symbols:
            return {"error": "ticker_not_found", "details": "Invalid ticker or no Yahoo data"}

        summary = t.summary_detail.get(ticker)
        profile = t.asset_profile.get(ticker)
        financial = t.financial_data.get(ticker)
        price = t.price.get(ticker)

        # If all sections missing → error
        if all(v is None for v in [summary, profile, financial, price]):
            return {"error": "ticker_not_found", "details": "Yahoo returned no data"}

        return {
            "summary": summary,
            "profile": profile,
            "financial": financial,
            "price": price
        }

    except Exception as e:
        return {"error": "api_error", "details": str(e)}

def _fetch_yahoo_earnings(ticker: str) -> Dict[str, Any]:
    """
    Fetch earnings data from Yahoo Finance using yahooquery.Ticker.earnings.

    Returns:
        earnings dict for this ticker (inner object of t.earnings[ticker]),
        or an error dict.
    """
    try:
        t = Ticker(ticker)
        all_earnings = t.earnings

        if not isinstance(all_earnings, dict) or ticker not in all_earnings:
            return {"error" : "no_data", "details": "No Yahoo earnings data for ticker"}
        
        earnings = all_earnings.get(ticker)
        if not earnings:
            return {"error": "no_data", "details": "Empty Yahoo earnings for ticker"}
        
        return earnings
    
    except Exception as e:
        return {"error": "api_error", "details": str(e)}

def _fetch_yahoo_quote(ticker: str) -> Dict[str, Any]:
    """
    Fetch real-time quote/price data from Yahoo Finance using yahooquery.Ticker.price.
    Returns raw quote dict or standardized error dict.
    """
    try:
        t = Ticker(ticker)
        quote = t.price.get(ticker)

        if not quote or "regularMarketPrice" not in quote:
            return {"error": "ticker_not_found", "details": "No quote data available"}

        return quote

    except Exception as e:
        return {"error": "api_error", "details": str(e)}

def _fetch_yahoo_insider_sentiment(ticker: str) -> Dict[str, Any]:
    """
    Extract insider sentiment (net insider buying/selling + MSPR) using Yahoo Finance data.
    """
    try:
        t = Ticker(ticker)
        tx = t.insider_transactions
        if tx is None or len(tx) == 0:
            return {"change": None, 
                    "mspr": None,
                    "totalBuys": None,
                    "totalSells": None,
                    "lastInsiderTrade": None}
        
        # Clean up
        df = tx.copy()
        if "transactionText" not in df.columns:
            return {
                "change": None,
                "mspr": None
            }

        df["text"] = df["transactionText"].astype(str).str.lower()
        df["isBuy"] = df["text"].str.contains("buy")
        df["isGift"] = df["text"].str.contains("gift") # Treat as neutral
        df["isSell"] = df["text"].str.contains("sale") | df["text"].str.contains("sell")

        # Extracting share amount
        df["shares"] = pd.to_numeric(df["shares"], errors = "coerce").fillna(0)

        # Computing
        total_buys = df[df["isBuy"]]["shares"].sum()
        total_sells = df[df["isSell"]]["shares"].sum()

        # Insider sentiment = net shares bought - sold
        net_change = float(total_buys - total_sells)

        # Last insider trade date
        last_date = None
        if "startDate" in df.columns:
            try:
                df["startDate"] = pd.to_datetime(df["startDate"], errors = "coerce")
                last_date = df["startDate"].max().strftime("%Y-%m-%d")
            except Exception:
                last_date = None
        
        # MSPR: buys vs sells last 90 days
        df_recent = df.dropna(subset = ["startDate"])
        recent_cutoff = df_recent["startDate"].max() - pd.Timedelta(days = 90)
        df_recent = df_recent[df_recent["startDate"] >= recent_cutoff]

        recent_buys = df_recent[df_recent["isBuy"]]["shares"].sum()
        recent_sells = df_recent[df_recent["isSell"]]["shares"].sum()

        if recent_buys == 0 and recent_sells == 0:
            mspr = None
        
        else:
            mspr = float(recent_buys / (recent_sells + 1e-9))

        return {
            "change": net_change,    # Net insider buying
            "mspr": mspr,            # Buy/Sell ratio
            "totalBuys": _safe_float(total_buys),     # Total shares bought
            "totalSells": _safe_float(total_sells),   # Total shares sold
            "lastInsiderTrade": last_date,      # Most recent transaction date
        }
    
    except Exception as e:
        return {
            "change": None,
            "mspr": None,
            "error": str(e)
        }

def _transform_yahoo_to_mongodb_format(
        ticker: str, 
        overview: Dict[str, Any] = None,
        earnings: Dict[str, Any] = None,
        quote: Dict[str, Any] = None,
        insider_sentiment: Dict[str, Any] = None
    ) -> Dict[str, Any]:
    """
    Transform Yahoo Finance data (overview + earnings + quote) into the MongoDB defined schema
    used by the metric extractor agent.

    The output structure matches _transform_av_to_mongodb_ format.
    """
    result: Dict[str, Any] = {
        "ticker": ticker,
        "company": None,
        "market_data": None,
        "basic_financials": None,
        "earnings_reports": [],
        "earnings_surprises": [],
        "news": [],
        "sec_filings": [],
        "financials_reported": [],
        "insider_sentiment": []
    }

    overview_has_error = overview is None or (isinstance(overview, dict) and overview.get("error"))
    
    if not overview_has_error and overview:
        summary = overview.get("summary")
        profile = overview.get("profile")
        financial = overview.get("financial")
        price = overview.get("price")

        # Transform overview to company data and basic financials data
        result["company"] = {
            "ticker": ticker,
            "name": price.get("longName") or price.get("shortName"),
            "country": profile.get("country"),
            "currency": price.get("currency") or summary.get("currency"),
            "exchange": price.get("exchange") or price.get("exchangeName"),
            "ipo": None,  # Yahoo doesn't provide IPO date here
            "marketCapitalization": _safe_float(summary.get("marketCap") or price.get("marketCap")),
            "phone": profile.get("phone"),
            "shareOutstanding": None,    # Yahoo doesn't provide
            "weburl": profile.get("website"),
            "logo": None,
            "finnhubIndustry": profile.get("sector"),
            "industry": profile.get("industry"),
            "description": profile.get("longBusinessSummary"),
        }

        # Basic financials (metrics)
        current_price = _safe_float(financial.get("currentPrice"))
        book_value = _safe_float(financial.get("bookValue"))

        pb_ratio = _safe_float(financial.get("priceToBook"))
        if pb_ratio is None and current_price is not None and book_value not in (None, 0):
            pb_ratio = current_price / book_value    # calculating pb_ratio manually if missing
            
        result["basic_financials"] = {
            "metric": {
                # Valuation
                "peBasicExclExtraTTM": _safe_float(summary.get("trailingPE") or financial.get("forwardPE")),
                "pbRatio": pb_ratio,
                "psTTM": _safe_float(summary.get("priceToSalesTrailing12Months")),
                "pegRatio": None,  # not in Yahoo Finance API

                # Profitability
                "profitMarginTTM": _safe_float(financial.get("profitMargins")),
                "operatingMarginTTM": _safe_float(financial.get("operatingMargins")),
                "roaTTM": _safe_float(financial.get("returnOnAssets")),
                "roeTTM": _safe_float(financial.get("returnOnEquity")),

                # Growth
                "revenueGrowthTTM": _safe_float(financial.get("revenueGrowth")),
                "revenueGrowthQuarterlyYoy": None,
                "epsGrowthTTMYoy": _safe_float(financial.get("earningsGrowth")),

                # Per share
                "bookValuePerShareQuarterly": book_value,
                "dividendPerShareTTM": _safe_float(summary.get("dividendRate")),
                # Other
                "52WeekHigh": _safe_float(summary.get("fiftyTwoWeekHigh")),
                "52WeekLow": _safe_float(summary.get("fiftyTwoWeekLow")),
                "beta": _safe_float(summary.get("beta")),
            },
            "series": {}
        }

        # Market Data from Quote 
        quote_has_error = quote is None or (isinstance(quote, dict) and quote.get("error"))

        # Prefer explicit quote; fall back to overview['price']
        source = None
        if not quote_has_error and quote:
            source = quote
        elif not overview_has_error and overview:
            source = (overview.get("price"))
        
        if source:
            result["market_data"] = {
                "c": _safe_float(source.get("regularMarketPrice")),
                "h": _safe_float(source.get("regularMarketDayHigh")),
                "l": _safe_float(source.get("regularMarketDayLow")),
                "o": _safe_float(source.get("regularMarketOpen")),
                "pc": _safe_float(source.get("regularMarketPreviousClose")),
                "t": source.get("regularMarketTime"),
                "volume": _safe_float(source.get("regularMarketVolume")),
                "change": _safe_float(source.get("regularMarketChange")),
                "changePercent": _safe_float(source.get("regularMarketChangePercent")),
            }
        
        # earnings -> earnings_reports + earnings_surprises
        earnings_has_error = earnings is None or (isinstance(earnings, dict) and earnings.get("error"))

        if not earnings_has_error and earnings:
            chart = earnings.get("earningsChart") or {}
            quarterly = chart.get("quarterly") or []
            fin_chart = earnings.get("financialsChart") or {}
            quarterly_financials = fin_chart.get("quarterly") or []

            # Build revenue_map
            revenue_map = {}
            for item in quarterly_financials:
                period_key = item.get("date") # e.g. 4Q2024
                revenue_val = item.get("revenue")
                if period_key:
                    revenue_map[period_key] = revenue_val

            for q in quarterly:
                cq = q.get("calendarQuarter") or q.get("date")
                year, quarter_num = None, None

                if isinstance(cq, str) and "Q" in cq:
                    try:
                        q_part, y_part = cq.split("Q")
                        quarter_num = int(q_part)
                        year = int(y_part)
                    except Exception:
                        pass
                
                eps_actual = _safe_float(q.get("actual"))
                eps_estimate = _safe_float(q.get("estimate"))
                surprise_pct_raw = _safe_float(q.get("surprisePct"))
                
                # Insert revenueActual from financialsChart if exists
                revenue_actual = revenue_map.get(cq)

                earnings_entry = {
                    "period": cq,      # e.g. "4Q2024"
                    "year": year,
                    "quarter": quarter_num,
                    "epsActual": eps_actual,
                    "epsEstimate": eps_estimate,
                    "revenueActual": revenue_actual,
                    "revenueEstimate": None,
                    "surprisePercent": surprise_pct_raw,
                }
                result["earnings_reports"].append(earnings_entry)
            
            for e in result["earnings_reports"]:
                if e["epsActual"] is not None and e["epsEstimate"] is not None:
                    result["earnings_surprises"].append({
                        "period": e["period"],
                        "year": e["year"],
                        "quarter": e["quarter"],
                        "actual": e["epsActual"],
                        "estimate": e["epsEstimate"],
                        "surprise": e["epsActual"] - e["epsEstimate"],
                        "surprisePercent": e["surprisePercent"],
                    })
        
        # Insider sentiment
        insider_sentiment_has_error = (
            insider_sentiment is None or 
            (isinstance(insider_sentiment, dict) and insider_sentiment.get("error"))
        )

        if not insider_sentiment_has_error and insider_sentiment:
            result['insider_sentiment'] = insider_sentiment
        
    return result
    
# ============================================================================
# Alpha Vantage Fallback Functions
# ============================================================================

def _fetch_alpha_vantage_overview(ticker: str, api_key: str) -> Dict[str, Any]:
    """
    Fetch company overview and financial data from Alpha Vantage OVERVIEW endpoint.

    Args:
        ticker: Stock ticker symbol
        api_key: Alpha Vantage API key

    Returns:
        Dictionary with overview data, or error dict if request fails
        Error types: {"error": "ticker_not_found"}, {"error": "rate_limit"},
                     {"error": "api_error", "details": str}
    """
    if not api_key:
        return {"error": "no_api_key", "details": "ALPHA_VANTAGE_API_KEY not configured"}

    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "OVERVIEW",
            "symbol": ticker,
            "apikey": api_key
        }
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return {"error": "api_error", "details": f"HTTP {response.status_code}"}

        data = response.json()

        # Check for rate limit (Alpha Vantage returns "Note" field)
        if "Note" in data:
            return {"error": "rate_limit", "message": data.get("Note")}

        # Check for error message (invalid ticker or other errors)
        if "Error Message" in data:
            return {"error": "ticker_not_found", "details": data.get("Error Message")}

        # Check if data is empty or invalid
        if not data or not data.get("Symbol"):
            return {"error": "ticker_not_found", "details": "No data returned for ticker"}

        # Verify the ticker matches (case-insensitive)
        if data.get("Symbol", "").upper() != ticker.upper():
            return {"error": "ticker_not_found", "details": f"Ticker mismatch: requested {ticker}, got {data.get('Symbol')}"}

        return data

    except requests.exceptions.Timeout:
        return {"error": "api_error", "details": "Request timeout"}
    except requests.exceptions.ConnectionError:
        return {"error": "api_error", "details": "Connection error"}
    except Exception as e:
        return {"error": "api_error", "details": str(e)}


def _fetch_alpha_vantage_earnings(ticker: str, api_key: str) -> Dict[str, Any]:
    """
    Fetch earnings data from Alpha Vantage EARNINGS endpoint.

    Args:
        ticker: Stock ticker symbol
        api_key: Alpha Vantage API key

    Returns:
        Dictionary with earnings data, or error dict if request fails
        Error types: {"error": "ticker_not_found"}, {"error": "rate_limit"},
                     {"error": "api_error", "details": str}
    """
    if not api_key:
        return {"error": "no_api_key", "details": "ALPHA_VANTAGE_API_KEY not configured"}

    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "EARNINGS",
            "symbol": ticker,
            "apikey": api_key
        }
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return {"error": "api_error", "details": f"HTTP {response.status_code}"}

        data = response.json()

        # Check for rate limit
        if "Note" in data:
            return {"error": "rate_limit", "message": data.get("Note")}

        # Check for error message
        if "Error Message" in data:
            return {"error": "ticker_not_found", "details": data.get("Error Message")}

        # Validate we got earnings data
        if not data or (not data.get("quarterlyEarnings") and not data.get("annualEarnings")):
            return {"error": "no_data", "details": "No earnings data available"}

        return data

    except requests.exceptions.Timeout:
        return {"error": "api_error", "details": "Request timeout"}
    except requests.exceptions.ConnectionError:
        return {"error": "api_error", "details": "Connection error"}
    except Exception as e:
        return {"error": "api_error", "details": str(e)}


def _fetch_alpha_vantage_quote(ticker: str, api_key: str) -> Dict[str, Any]:
    """
    Fetch real-time market data from Alpha Vantage GLOBAL_QUOTE endpoint.

    Args:
        ticker: Stock ticker symbol
        api_key: Alpha Vantage API key

    Returns:
        Dictionary with quote data, or error dict if request fails
        Error types: {"error": "ticker_not_found"}, {"error": "rate_limit"},
                     {"error": "api_error", "details": str}
    """
    if not api_key:
        return {"error": "no_api_key", "details": "ALPHA_VANTAGE_API_KEY not configured"}

    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker,
            "apikey": api_key
        }
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return {"error": "api_error", "details": f"HTTP {response.status_code}"}

        data = response.json()

        # Check for rate limit
        if "Note" in data:
            return {"error": "rate_limit", "message": data.get("Note")}

        # Check for error message
        if "Error Message" in data:
            return {"error": "ticker_not_found", "details": data.get("Error Message")}

        # Extract the Global Quote object
        quote = data.get("Global Quote", {})
        if not quote or not quote.get("01. symbol"):
            return {"error": "ticker_not_found", "details": "No quote data available"}

        return quote

    except requests.exceptions.Timeout:
        return {"error": "api_error", "details": "Request timeout"}
    except requests.exceptions.ConnectionError:
        return {"error": "api_error", "details": "Connection error"}
    except Exception as e:
        return {"error": "api_error", "details": str(e)}


def _safe_float(value: Any) -> Any:
    """
    Safely convert a value to float, returning None if conversion fails.

    Args:
        value: Value to convert (can be string, number, None, etc.)

    Returns:
        Float value or None if conversion fails or value is invalid
    """
    if value is None or value == "None" or value == "" or value == "N/A":
        return None
    try:
        # Handle percentage strings (e.g., "5.2%")
        if isinstance(value, str) and "%" in value:
            value = value.replace("%", "")
        return float(value)
    except (ValueError, TypeError):
        return None


def _transform_av_to_mongodb_format(
    ticker: str,
    overview: Dict[str, Any] = None,
    earnings: Dict[str, Any] = None,
    quote: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Transform Alpha Vantage API responses to match MongoDB data structure.

    This function maps Alpha Vantage data to the expected format used by the
    metric extractor agent, matching the structure returned by _fetch_single_ticker().

    Handles missing data gracefully:
    - If an entire endpoint fails (error dict), that section is set to None
    - If specific fields are missing, they default to None via _safe_float()
    - Returns valid structure even if all inputs are None/errors
    - Compatible with normalize_financial_data() and detect_anomalies()

    Args:
        ticker: Stock ticker symbol
        overview: Data from OVERVIEW endpoint (can be None or error dict)
        earnings: Data from EARNINGS endpoint (can be None or error dict)
        quote: Data from GLOBAL_QUOTE endpoint (can be None or error dict)

    Returns:
        Dictionary matching MongoDB structure with all available data
    """
    result = {
        "ticker": ticker,
        "company": None,
        "market_data": None,
        "basic_financials": None,
        "earnings_reports": [],  # Default to empty list, not None
        "earnings_surprises": [],  # Default to empty list, not None
        "news": [],  # Not available from Alpha Vantage free tier
        "sec_filings": [],  # Not available from Alpha Vantage free tier
        "financials_reported": [],
        "insider_sentiment": []  # Not available from Alpha Vantage
    }

    # Check if overview contains error
    overview_has_error = overview is None or (isinstance(overview, dict) and overview.get("error"))

    # Transform OVERVIEW data into company and basic_financials
    if not overview_has_error and overview:
        result["company"] = {
            "ticker": ticker,
            "name": overview.get("Name"),
            "country": overview.get("Country"),
            "currency": overview.get("Currency"),
            "exchange": overview.get("Exchange"),
            "ipo": overview.get("IPODate"),
            "marketCapitalization": _safe_float(overview.get("MarketCapitalization")),
            "phone": overview.get("Phone"),
            "shareOutstanding": _safe_float(overview.get("SharesOutstanding")),
            "weburl": overview.get("OfficialSite"),
            "logo": None,  # Not available from Alpha Vantage
            "finnhubIndustry": overview.get("Sector"),
            "industry": overview.get("Industry"),
            "description": overview.get("Description")
        }

        # Create basic_financials with metric dict
        # normalize_financial_data expects this structure
        result["basic_financials"] = {
            "metric": {
                # Valuation metrics (used by detect_anomalies)
                "peBasicExclExtraTTM": _safe_float(overview.get("PERatio")),
                "pbRatio": _safe_float(overview.get("PriceToBookRatio")),
                "psTTM": _safe_float(overview.get("PriceToSalesRatioTTM")),
                "pegRatio": _safe_float(overview.get("PEGRatio")),

                # Profitability metrics
                "profitMarginTTM": _safe_float(overview.get("ProfitMargin")),
                "operatingMarginTTM": _safe_float(overview.get("OperatingMarginTTM")),
                "roaTTM": _safe_float(overview.get("ReturnOnAssetsTTM")),
                "roeTTM": _safe_float(overview.get("ReturnOnEquityTTM")),

                # Growth metrics
                "revenueGrowthTTM": _safe_float(overview.get("QuarterlyRevenueGrowthYOY")),
                "revenueGrowthQuarterlyYoy": _safe_float(overview.get("QuarterlyRevenueGrowthYOY")),
                "epsGrowthTTMYoy": _safe_float(overview.get("QuarterlyEarningsGrowthYOY")),

                # Per-share metrics
                "bookValuePerShareQuarterly": _safe_float(overview.get("BookValue")),
                "dividendPerShareTTM": _safe_float(overview.get("DividendPerShare")),

                # Other metrics (used by detect_anomalies for 52-week high/low)
                "52WeekHigh": _safe_float(overview.get("52WeekHigh")),
                "52WeekLow": _safe_float(overview.get("52WeekLow")),
                "beta": _safe_float(overview.get("Beta"))
            },
            "series": {}  # Alpha Vantage doesn't provide time series in OVERVIEW
        }

    # Check if quote contains error
    quote_has_error = quote is None or (isinstance(quote, dict) and quote.get("error"))

    # Transform GLOBAL_QUOTE data into market_data
    # normalize_financial_data expects fields: c, h, l, o, pc
    if not quote_has_error and quote:
        # Safe extraction with fallback to None
        change_percent_raw = quote.get("10. change percent", "")
        if isinstance(change_percent_raw, str):
            change_percent_raw = change_percent_raw.replace("%", "")

        result["market_data"] = {
            "c": _safe_float(quote.get("05. price")),  # Current price
            "h": _safe_float(quote.get("03. high")),   # High
            "l": _safe_float(quote.get("04. low")),    # Low
            "o": _safe_float(quote.get("02. open")),   # Open
            "pc": _safe_float(quote.get("08. previous close")),  # Previous close
            "t": None,  # Timestamp not directly available
            "volume": _safe_float(quote.get("06. volume")),
            "change": _safe_float(quote.get("09. change")),
            "changePercent": _safe_float(change_percent_raw)
        }

    # Check if earnings contains error
    earnings_has_error = earnings is None or (isinstance(earnings, dict) and earnings.get("error"))

    # Transform EARNINGS data into earnings_reports
    # normalize_financial_data and detect_anomalies expect list format
    if not earnings_has_error and earnings and earnings.get("quarterlyEarnings"):
        quarterly = earnings.get("quarterlyEarnings", [])

        # Ensure we have a list
        if not isinstance(quarterly, list):
            quarterly = []

        result["earnings_reports"] = []

        for report in quarterly[:8]:  # Get last 8 quarters
            # Skip if report is not a dict
            if not isinstance(report, dict):
                continue

            fiscal_date = report.get("fiscalDateEnding", "")

            # Parse year and quarter from fiscalDateEnding (format: YYYY-MM-DD)
            year = None
            quarter = None
            if fiscal_date:
                try:
                    year = int(fiscal_date[:4])
                    month = int(fiscal_date[5:7])
                    quarter = (month - 1) // 3 + 1
                except (ValueError, IndexError):
                    # If parsing fails, leave as None
                    pass

            eps_actual = _safe_float(report.get("reportedEPS"))
            eps_estimate = _safe_float(report.get("estimatedEPS"))

            # Build earnings entry - matches MongoDB structure
            earnings_entry = {
                "period": fiscal_date,
                "year": year,
                "quarter": quarter,
                "epsActual": eps_actual,
                "epsEstimate": eps_estimate,
                "revenueActual": None,  # Not available in Alpha Vantage quarterly earnings
                "revenueEstimate": None,
                "surprisePercent": None
            }

            # Calculate surprise percentage if we have both values
            if eps_actual is not None and eps_estimate is not None and eps_estimate != 0:
                earnings_entry["surprisePercent"] = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100

            result["earnings_reports"].append(earnings_entry)

        # Also populate earnings_surprises with the same data
        # detect_anomalies expects this format with 'surprisePercent' field
        result["earnings_surprises"] = []
        for e in result["earnings_reports"]:
            # Only include if we have actual and estimate values
            if e.get("epsActual") is not None and e.get("epsEstimate") is not None:
                result["earnings_surprises"].append({
                    "period": e["period"],
                    "year": e["year"],
                    "quarter": e["quarter"],
                    "actual": e["epsActual"],
                    "estimate": e["epsEstimate"],
                    "surprise": e["epsActual"] - e["epsEstimate"] if e["epsActual"] and e["epsEstimate"] else None,
                    "surprisePercent": e["surprisePercent"]
                })

    return result


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
def fetch_ticker_summary(tickers: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Fetches comprehensive summary data for a list of tickers using the /ticker-summary endpoint.
    This is more efficient than fetch_ticker_data as it makes fewer API calls.

    Args:
        tickers: Single ticker string (e.g., "TSLA") or List of ticker symbols (e.g., ["TSLA", "AAPL"])

    Returns:
        Dictionary with ticker symbols as keys, each containing summary data with latest information
        from all collections (company, market_data, basic_financials, latest_earnings, latest_news,
        recent_filings, insider_sentiment).
    """
    result = {}

    # Normalize input to always be a list
    if isinstance(tickers, str):
        ticker_list = [tickers]
    else:
        ticker_list = tickers

    for ticker in ticker_list:
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

    # Remove internal metadata fields that shouldn't be passed to LLMs
    normalized.pop("_fallback_status", None)
    normalized.pop("_metadata", None)

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
