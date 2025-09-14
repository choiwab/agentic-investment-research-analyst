import os
import random
import time
from datetime import date, timedelta

import finnhub
from bson.int64 import Int64
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    raise RuntimeError("FINNHUB_API_KEY is not set. Add it to your .env or environment.")

ATLAS_URI = os.getenv("ATLAS_URI")
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# client = MongoClient("mongodb://root:password@localhost:27017/?authSource=admin")
# db = client["test"]
def get_mongo_client():
    return MongoClient(ATLAS_URI)
    # Uncomment below to use different connection strings based on environment
    # if os.getenv('AIRFLOW__CORE__EXECUTOR'):
    #     # Use Docker service name
    #     return MongoClient("mongodb://root:password@mongo:27017/?authSource=admin")
    # else:
    #     # Use localhost for local development
    #     return MongoClient("mongodb://root:password@localhost:27017/?authSource=admin")

# Replace the existing client initialization
client = get_mongo_client()
db = client["test"]

# ---- EXTRACT ----
def extract_symbols(exchange="US"):
    # get a list of tickers
    rows = finnhub_client.stock_symbols(exchange=exchange) or []
    # Keep only stocks with a non-empty symbol
    symbols = [r["symbol"] for r in rows if r.get("symbol") and r.get("type") == "Common Stock"]
    return symbols


def extract_company_profile(ticker):
    return finnhub_client.company_profile2(symbol=ticker)

def extract_earnings(ticker, days=30):
    start = (date.today() - timedelta(days=days)).isoformat()
    end   = date.today().isoformat()
    res = finnhub_client.earnings_calendar(_from=start, to=end, symbol=ticker) or {}
    # unwrap the list
    return res.get("earningsCalendar", [])

def extract_news(ticker, start=None, end=None):
    # if no date/year is specified, then take last 1 year
    if not start or not end:
        start = (date.today()-timedelta(days=365)).isoformat()
        end = date.today().isoformat()

    return finnhub_client.company_news(ticker, _from=start, to=end)

def extract_filings(ticker, start=None, end=None):
    # if no date/year is specified, then take 2024-01-01 to 2024-12-31
    if not start or not end:
        start = (date.today()-timedelta(days=365)).isoformat()
        end = date.today().isoformat()

    if hasattr(finnhub_client, "stock_filings"):
        return finnhub_client.stock_filings(symbol=ticker, _from=start, to=end) or []
    return finnhub_client.filings(symbol=ticker, **{"_from": start, "to": end}) or []    

def extract_quote(ticker):
    q = finnhub_client.quote(symbol=ticker) or {}
    q["ticker"] = ticker
    return q

def extract_earnings_surprises(ticker, limit=8):
    return finnhub_client.company_earnings(ticker, limit=limit) or []

def extract_financials_reported(ticker, freq="quarterly", days=365):
    _from = (date.today()-timedelta(days=days)).isoformat()
    _to   = date.today().isoformat()
    res = finnhub_client.financials_reported(symbol=ticker, freq=freq, _from=_from, to=_to) or {}
    return res.get("data", [])  

def extract_insider_sentiment(ticker, days=365):
    _from = (date.today()-timedelta(days=days)).isoformat()
    _to   = date.today().isoformat()
    res = finnhub_client.stock_insider_sentiment(ticker, _from, _to) or {}
    return res.get("data", [])

def extract_basic_financials(ticker, metric="all"):
    return finnhub_client.company_basic_financials(symbol=ticker, metric=metric) or {}


# ---- TRANSFORM ----
def transform_company_profile(raw):
    if not raw or not raw.get("ticker") or not raw.get("name"):
        return None
    doc = {"_id": raw["ticker"], "name": raw["name"]}
    if raw.get("country"): doc["country"] = raw["country"]
    if raw.get("currency"): doc["currency"] = raw["currency"]
    if raw.get("exchange"): doc["exchange"] = raw["exchange"]
    if raw.get("finnhubIndustry"): doc["finnhubIndustry"] = raw["finnhubIndustry"]
    if raw.get("ipo"): doc["ipo"] = raw["ipo"]
    if raw.get("logo"): doc["logo"] = raw["logo"]
    if raw.get("marketCapitalization") is not None: doc["marketCapitalization"] = float(raw["marketCapitalization"])
    if raw.get("phone"): doc["phone"] = raw["phone"]
    if raw.get("shareOutstanding") is not None: doc["shareOutstanding"] = float(raw["shareOutstanding"])
    if raw.get("weburl"): doc["weburl"] = raw["weburl"]
    return doc

def transform_earnings(rows, ticker):
    cleaned = []
    for r in rows or []:
        if r.get("year") is None or r.get("quarter") is None:
            continue
        d = {"ticker": ticker, "year": int(r["year"]), "quarter": int(r["quarter"])}
        if r.get("date"): d["date"] = r["date"]            # string only if present
        if r.get("hour"): d["hour"] = r["hour"]            # string only if present
        if r.get("epsActual") is not None: d["epsActual"] = float(r["epsActual"])
        if r.get("epsEstimate") is not None: d["epsEstimate"] = float(r["epsEstimate"])
        if r.get("revenueActual") is not None: d["revenueActual"] = float(r["revenueActual"])
        if r.get("revenueEstimate") is not None: d["revenueEstimate"] = float(r["revenueEstimate"])
        cleaned.append(d)
    return cleaned

def transform_news(rows, ticker):
    cleaned = []
    for r in rows or []:
        if r.get("id") is None:
            continue
        d = {"ticker": ticker, "id": int(r["id"])}
        if r.get("datetime") is not None: d["datetime"] = int(r["datetime"])
        if r.get("category"): d["category"] = r["category"]
        if r.get("headline"): d["headline"] = r["headline"]
        if r.get("image"): d["image"] = r["image"]
        if r.get("related"): d["related"] = r["related"]
        if r.get("source"): d["source"] = r["source"]
        if r.get("summary"): d["summary"] = r["summary"]
        if r.get("url"): d["url"] = r["url"]
        cleaned.append(d)
    return cleaned

def transform_sec_filings(rows, ticker):
    cleaned = []
    for r in rows or []:
        if not r.get("accessNumber"):
            continue
        d = {"ticker": ticker, "accessNumber": r["accessNumber"]}
        if r.get("cik"): d["cik"] = r["cik"]
        if r.get("form"): d["form"] = r["form"]
        if r.get("filedDate"): d["filedDate"] = r["filedDate"]
        if r.get("acceptedDate"): d["acceptedDate"] = r["acceptedDate"]
        if r.get("reportUrl"): d["reportUrl"] = r["reportUrl"]
        if r.get("filingUrl"): d["filingUrl"] = r["filingUrl"]
        cleaned.append(d)
    return cleaned

def transform_market_data(raw, ticker):
    if not raw or not raw.get("t"):
        return None
    d = {"ticker": ticker, "t": int(raw["t"])}
    if raw.get("c") is not None: d["c"] = float(raw["c"])
    if raw.get("d") is not None: d["d"] = float(raw["d"])
    if raw.get("dp") is not None: d["dp"] = float(raw["dp"])
    if raw.get("h") is not None: d["h"] = float(raw["h"])
    if raw.get("l") is not None: d["l"] = float(raw["l"])
    if raw.get("o") is not None: d["o"] = float(raw["o"])
    if raw.get("pc") is not None: d["pc"] = float(raw["pc"])

    # ensure 64 bit
    d["t"] = Int64(int(raw["t"]))
    return d

def transform_earnings_surprises(rows, ticker):
    cleaned = []
    for r in rows or []:
        if r.get("year") is None or r.get("quarter") is None:
            continue
        d = {
            "ticker": ticker,
            "year": int(r["year"]),
            "quarter": int(r["quarter"])
        }
        if r.get("period"): d["period"] = r["period"]
        for f in ("actual","estimate","surprise","surprisePercent"):
            if r.get(f) is not None:
                d[f] = float(r[f])
        cleaned.append(d)
    return cleaned

def transform_financials_reported(rows, ticker):
    cleaned = []
    for r in rows or []:
        access = r.get("accessNumber")
        if not access:
            continue
        d = {
            "ticker": ticker,
            "accessNumber": access
        }
        # metadata
        for k in ("cik","form","year","quarter","startDate","endDate","filedDate","acceptedDate"):
            if r.get(k) is not None:
                d[k] = r[k]
        # the full report object (bs/cf/ic) can be large; keep as-is
        if r.get("report"): d["report"] = r["report"]
        cleaned.append(d)
    return cleaned

def transform_insider_sentiment(rows, ticker):
    cleaned = []
    for r in rows or []:
        if r.get("year") is None or r.get("month") is None:
            continue
        d = {
            "ticker": ticker,
            "year": int(r["year"]),
            "month": int(r["month"])
        }
        # change (net buys), mspr (monthly share purchase ratio)
        if r.get("change") is not None: d["change"] = float(r["change"])
        if r.get("mspr")  is not None: d["mspr"]  = float(r["mspr"])
        cleaned.append(d)
    return cleaned

def transform_basic_financials(raw, ticker):
    if not raw:
        return None
    # Store one doc per ticker
    doc = {"_id": ticker, "ticker": ticker}   
    if raw.get("metric"):  doc["metric"] = raw["metric"]      # point-in-time KPIs (P/E, 52W ranges, etc.)
    if raw.get("series"):  doc["series"] = raw["series"]      # time-series ratios by period
    return doc


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
def run_pipeline(exchange="US", max_symbols=5):
    tickers = extract_symbols(exchange)[:max_symbols]
    for ticker in tickers:
        print(f"Processing {ticker}...")

        # companies
        profile = transform_company_profile(extract_company_profile(ticker))
        if profile:
            print(f"Upserting company profile: {profile}")
            db.companies.update_one({"_id": profile["_id"]}, {"$set": profile}, upsert=True)

        # earnings
        for d in transform_earnings(extract_earnings(ticker), ticker):
            db.earnings_reports.update_one(
                {"ticker": d["ticker"], "year": d["year"], "quarter": d["quarter"]},
                {"$set": d},
                upsert=True
            )

        # news
        for d in transform_news(extract_news(ticker), ticker):
            db.news.update_one(
                {"ticker": d["ticker"], "id": d["id"]},
                {"$set": d},
                upsert=True
            )

        # sec filings
        for d in transform_sec_filings(extract_filings(ticker), ticker):
            db.sec_fillings.update_one(
                {"ticker": d["ticker"], "accessNumber": d["accessNumber"]},
                {"$set": d},
                upsert=True
            )

        # market data
        mkt = transform_market_data(extract_quote(ticker), ticker)
        if mkt:
            db.market_data.update_one(
                {"ticker": mkt["ticker"], "t": mkt["t"]},
                {"$set": mkt},
                upsert=True
            )

        # earnings surprises (historical)
        for d in transform_earnings_surprises(extract_earnings_surprises(ticker), ticker):
            # print(f"Upserting earnings surprise: {d}") 
            db.earnings_surprises.update_one(
                {"ticker": d["ticker"], "year": d["year"], "quarter": d["quarter"]},
                {"$set": d},
                upsert=True
            )

        # financials as reported
        for d in transform_financials_reported(extract_financials_reported(ticker, freq="quarterly"), ticker):
            # print(f"Upserting financials as reported: {d}") 
            db.financials_reported.update_one(
                {"ticker": d["ticker"], "accessNumber": d["accessNumber"]},
                {"$set": d},
                upsert=True
            )

        # insider sentiment
        for d in transform_insider_sentiment(extract_insider_sentiment(ticker), ticker):
            print(f"Upserting insider sentiment: {d}") 
            db.insider_sentiment.update_one(
                {"ticker": d["ticker"], "year": d["year"], "month": d["month"]},
                {"$set": d},
                upsert=True
            )

        # basic financials
        bf = transform_basic_financials(extract_basic_financials(ticker), ticker)
        if bf:
            # print(f"Upserting basic financials: {bf}") 
            db.basic_financials.update_one({"_id": bf["_id"]}, {"$set": bf}, upsert=True)

        time.sleep(random.uniform(1.0,3.0))

#tryout
if __name__ == "__main__":
    # print(finnhub_client.earnings_calendar(_from="2025-08-06", to="2025-09-06", symbol="EMAX", international=False))
    run_pipeline(exchange="US", max_symbols=5)
