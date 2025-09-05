import finnhub
from pymongo import MongoClient
import time
from bson import Int64

finnhub_client = finnhub.Client(api_key="d2pjmthr01qnf9nlcku0d2pjmthr01qnf9nlckug")

client = MongoClient("mongodb://root:password@localhost:27017/")
db = client["test"]

# ---- EXTRACT ----
def extract_company_profile(ticker):
    return finnhub_client.company_profile2(symbol=ticker)

def extract_earnings(ticker):
    return finnhub_client.company_earnings(ticker, limit=10)

def extract_news(ticker):
    return finnhub_client.company_news(ticker, _from="2024-01-01", to="2024-12-31")

def extract_earnings_surprises(ticker):
    return finnhub_client.company_earnings(ticker, limit=10)

def extract_financials_reported(ticker):
    return finnhub_client.financials_reported(symbol=ticker, freq="annual")

def extract_insider_sentiment(ticker):
    return finnhub_client.stock_insider_sentiment(ticker, "2021-01-01", "2022-12-31")

def extract_basic_financials(ticker):
    return finnhub_client.company_basic_financials(ticker, "all")

def extract_sec_filings(ticker):
    return finnhub_client.filings(symbol=ticker, _from="2020-01-01", to="2025-12-31")

def extract_market_data(ticker):
    return finnhub_client.quote(ticker)

# ---- TRANSFORM ----
def transform_earnings(data, ticker):
    for item in data:
        item["ticker"] = ticker
    return data

def transform_news(data, ticker):
    for item in data:
        item["ticker"] = ticker
    return data

def transform_earnings_surprises(data, ticker):
    for item in data:
        item["symbol"] = ticker
    return data

def transform_financials_reported(data, ticker):
    docs = []
    for item in data.get("data", []):
        item["symbol"] = ticker
        docs.append(item)
    return docs

def transform_insider_sentiment(data, ticker):
    docs = []
    for item in data.get("data", []):
        item["symbol"] = ticker
        docs.append(item)
    return docs

def transform_basic_financials(data, ticker):
    doc = data
    doc["symbol"] = ticker
    return doc

def transform_company_profile(data):
    if not data:
        return None
    doc = data.copy()
    doc["_id"] = doc.pop("ticker")  
    return doc

def transform_sec_filings(data):
    docs = []
    for item in data:
        item["ticker"] = item.pop("symbol")
        docs.append(item)
    return docs

def transform_market_data(data, ticker):
    data["ticker"] = ticker
    if "t" in data and data["t"] is not None:
        data["t"] = Int64(data["t"])
    return data


# ---- LOAD ----
def load_collection(name, docs):
    if not docs:
        return

    if name == "companies":
        db[name].update_one({"_id": docs["_id"]}, {"$set": docs}, upsert=True)

    elif name == "earnings_reports":
        for doc in docs:
            db[name].update_one(
                {"ticker": doc["ticker"], "year": doc["year"], "quarter": doc["quarter"]},
                {"$set": doc},
                upsert=True
            )

    elif name == "news":
        for doc in docs:
            db[name].update_one(
                {"ticker": doc["ticker"], "id": doc["id"]},
                {"$set": doc},
                upsert=True
            )

    elif name == "earnings_surprises":
        for doc in docs:
            db[name].update_one(
                {"symbol": doc["symbol"], "year": doc["year"], "quarter": doc["quarter"]},
                {"$set": doc},
                upsert=True
            )

    elif name == "financials_reported":
        for doc in docs:
            db[name].update_one(
                {"symbol": doc["symbol"], "accessNumber": doc["accessNumber"]},
                {"$set": doc},
                upsert=True
            )

    elif name == "insider_sentiment":
        for doc in docs:
            db[name].update_one(
                {"symbol": doc["symbol"], "year": doc["year"], "month": doc["month"]},
                {"$set": doc},
                upsert=True
            )

    elif name == "basic_financials":
        db[name].update_one(
            {"symbol": docs["symbol"]},
            {"$set": docs},
            upsert=True
        )

    elif name == "sec_fillings":
        for doc in docs:
            db[name].update_one(
                {"ticker": doc["ticker"], "accessNumber": doc["accessNumber"]},
                {"$set": doc},
                upsert=True
            )

    elif name == "market_data":
        db[name].update_one(
            {"ticker": docs["ticker"]},
            {"$set": docs},
            upsert=True
        )

# ---- PIPELINE ----
def run_pipeline(tickers):
    for ticker in tickers:
        print(f"Processing {ticker}...")

        # ---- Companies ----
        profile = transform_company_profile(extract_company_profile(ticker))
        load_collection("companies", profile)

        # ---- Earnings Reports ----
        earnings = transform_earnings(extract_earnings(ticker), ticker)
        load_collection("earnings_reports", earnings)

        # ---- News ----
        news = transform_news(extract_news(ticker), ticker)
        load_collection("news", news)

        # ---- Earnings Surprises ----
        earnings_surprises = transform_earnings_surprises(extract_earnings_surprises(ticker), ticker)
        load_collection("earnings_surprises", earnings_surprises)

        # ---- Financials Reported ----
        financials = transform_financials_reported(extract_financials_reported(ticker), ticker)
        load_collection("financials_reported", financials)

        # ---- Insider Sentiment ----
        insider = transform_insider_sentiment(extract_insider_sentiment(ticker), ticker)
        load_collection("insider_sentiment", insider)

        # ---- Basic Financials ----
        basic = transform_basic_financials(extract_basic_financials(ticker), ticker)
        load_collection("basic_financials", basic)

        # ---- SEC Filings ----
        sec = transform_sec_filings(extract_sec_filings(ticker))
        load_collection("sec_fillings", sec)

        # ---- Market Data ----
        market = transform_market_data(extract_market_data(ticker), ticker)
        load_collection("market_data", market)

        time.sleep(1)  

# ---- RUN ----
tickers = ["AAPL", "TSLA", "AMZN"]
run_pipeline(tickers)
