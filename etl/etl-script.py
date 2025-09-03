import finnhub
from pymongo import MongoClient
import time

# Setup Finnhub client
finnhub_client = finnhub.Client(api_key="d2pjmthr01qnf9nlcku0d2pjmthr01qnf9nlckug")

# Setup Mongo
client = MongoClient("mongodb://root:password@localhost:27017/")
db = client["test"]

# ---- EXTRACT ----
def extract_company_profile(ticker):
    return finnhub_client.company_profile2(symbol=ticker)

def extract_earnings(ticker):
    return finnhub_client.company_earnings(ticker, limit=10)

def extract_news(ticker):
    return finnhub_client.company_news(ticker, _from="2024-01-01", to="2024-12-31")

# ---- TRANSFORM ----
def transform_profile(data):
    # set _id = ticker for uniqueness
    data["_id"] = data["ticker"]
    return data

def transform_earnings(data, ticker):
    for item in data:
        item["ticker"] = ticker
    return data

def transform_news(data, ticker):
    for item in data:
        item["ticker"] = ticker
    return data

# ---- LOAD ----
def load_collection(name, docs):
    if not docs:
        return
    if isinstance(docs, dict):  # single document
        db[name].update_one({"_id": docs["_id"]}, {"$set": docs}, upsert=True)
    else:  # list of documents
        for doc in docs:
            db[name].update_one(
                {"ticker": doc["ticker"], "id": doc.get("id")}, 
                {"$set": doc}, 
                upsert=True
            )

# ---- PIPELINE ----
def run_pipeline(tickers):
    for ticker in tickers:
        print(f"Processing {ticker}...")

        profile = transform_profile(extract_company_profile(ticker))
        load_collection("companies", profile)

        earnings = transform_earnings(extract_earnings(ticker), ticker)
        load_collection("earnings_reports", earnings)

        news = transform_news(extract_news(ticker), ticker)
        load_collection("news", news)

        time.sleep(1)  # respect API rate limits

# Example usage
tickers = ["AAPL", "TSLA", "AMZN"]
run_pipeline(tickers)
