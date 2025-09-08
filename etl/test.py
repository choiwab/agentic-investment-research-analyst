import finnhub
from pymongo import MongoClient
from datetime import date, timedelta
from bson.int64 import Int64
import time

finnhub_client = finnhub.Client(api_key="d2uotr1r01qq994gloi0d2uotr1r01qq994gloig")

try:
    print("Attempting to connect to MongoDB...")
    client = MongoClient("mongodb://root:password@localhost:27017/?authSource=admin")
    # Test the connection
    client.server_info()
    print("Successfully connected to MongoDB")
    db = client["test"]
except Exception as e:
    print(f"MongoDB Connection Error: {e}")
    exit(1)

# Test Finnhub connection
try:
    print("Testing Finnhub connection...")
    test_symbol = finnhub_client.stock_symbols('US')[0]
    print(f"Successfully connected to Finnhub. Retrieved symbol: {test_symbol['symbol']}")
except Exception as e:
    print(f"Finnhub Error: {e}")
    exit(1)

def extract_earnings(ticker, days=30):
    start = (date.today() - timedelta(days=days)).isoformat()
    end   = date.today().isoformat()
    res = finnhub_client.earnings_calendar(_from=start, to=end, symbol=ticker) or {}
    # unwrap the list
    return res.get("earningsCalendar", [])

# ---- NEW: Earnings Surprises (history; last 8 by default) ----
def extract_earnings_surprises(ticker, limit=8):
    return finnhub_client.company_earnings(ticker, limit=limit) or []

# ---- NEW: Financials As Reported (quarterly by default; last year) ----
def extract_financials_reported(ticker, freq="quarterly", days=365):
    _from = (date.today()-timedelta(days=days)).isoformat()
    _to   = date.today().isoformat()
    res = finnhub_client.financials_reported(symbol=ticker, freq=freq, _from=_from, to=_to) or {}
    return res.get("data", [])  # Finnhub returns {"data": [...], "cik": "...", "symbol": "..."}

# ---- NEW: Insider Sentiment (last year) ----
def extract_insider_sentiment(ticker, days=365):
    _from = (date.today()-timedelta(days=days)).isoformat()
    _to   = date.today().isoformat()
    res = finnhub_client.stock_insider_sentiment(ticker, _from, _to) or {}
    return res.get("data", [])

# ---- NEW: Basic Financials (point-in-time KPIs + timeseries) ----
def extract_basic_financials(ticker, metric="all"):
    return finnhub_client.company_basic_financials(symbol=ticker, metric=metric) or {}

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

# ---- NEW: transform earnings surprises ----
def transform_earnings_surprises(rows, ticker):
    cleaned = []
    for r in rows or []:
        # Finnhub fields: actual, estimate, period, quarter, surprise, surprisePercent, symbol, year
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

# ---- NEW: transform financials as reported ----
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

# ---- NEW: transform insider sentiment ----
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

# ---- NEW: transform basic financials ----
def transform_basic_financials(raw, ticker):
    if not raw:
        return None
    # Store one doc per ticker
    doc = {"_id": ticker}
    if raw.get("metric"):  doc["metric"] = raw["metric"]      # point-in-time KPIs (P/E, 52W ranges, etc.)
    if raw.get("series"):  doc["series"] = raw["series"]      # time-series ratios by period
    return doc

if __name__ == "__main__":
    ticker = "AAPL"   # test with Apple (or any symbol you like)

    print(finnhub_client.earnings_calendar(_from="2025-08-06", to="2025-09-06", symbol=ticker))

    print("\n=== Earnings Surprises ===")
    es = transform_earnings_surprises(extract_earnings_surprises(ticker), ticker)
    print(es[:2])   # print first 2 docs

    print("\n=== Financials Reported ===")
    fr = transform_financials_reported(extract_financials_reported(ticker, freq="quarterly"), ticker)
    print(fr[:1])   # first doc only (these can be large)

    print("\n=== Insider Sentiment ===")
    ins = transform_insider_sentiment(extract_insider_sentiment(ticker), ticker)
    print(ins[:3])  # first 3 months

    # print("\n=== Basic Financials ===")
    # bf = transform_basic_financials(extract_basic_financials(ticker), ticker)
    # print(bf)   # single doc, not a list