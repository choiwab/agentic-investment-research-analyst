from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymongo import UpdateOne

sys.path.append('/opt/airflow/etl')
#import etl_script
from etl import etl_script

# Import your ETL functionss
from etl.etl_script import (
    extract_symbols, extract_company_profile, extract_earnings,
    extract_news, extract_filings, extract_quote, extract_earnings_surprises,
    extract_financials_reported, extract_insider_sentiment, extract_basic_financials,
    extract_peers,
    transform_company_profile, transform_earnings, transform_news,
    transform_sec_filings, transform_market_data, transform_earnings_surprises,
    transform_financials_reported, transform_insider_sentiment, transform_basic_financials,
    transform_peers,
    db
)
import time
import random

# Default arguments
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'financial_data_etl',
    default_args=default_args,
    description='ETL pipeline for financial data from Finnhub',
    schedule_interval='@continuous',  # Continuous execution - runs immediately after previous run completes
        # schedule_interval='0 9 * * 1-5' - Weekdays at 9 AM
        # schedule_interval=timedelta(hours=4) - Every 4 hours
        # schedule_interval='@daily' - Once per day
        # schedule_interval=None - Manual trigger only
    catchup=False,
    max_active_runs=1,  # Ensures only one instance runs at a time
)

def process_single_basic_ticker(ticker):
    """Process basic data for a single ticker - used for parallel execution"""
    results = {
        'ticker': ticker,
        'success': False,
        'companies': [],
        'earnings': [],
        'news': [],
        'market_data': [],
        'peers': []
    }

    try:
        # Companies
        profile = transform_company_profile(extract_company_profile(ticker))
        if profile:
            results['companies'].append(profile)

        # Earnings
        earnings_data = transform_earnings(extract_earnings(ticker), ticker)
        if earnings_data:
            results['earnings'].extend(earnings_data)

        # News
        news_data = transform_news(extract_news(ticker), ticker)
        if news_data:
            results['news'].extend(news_data)

        # Market Data
        mkt = transform_market_data(extract_quote(ticker), ticker)
        if mkt:
            results['market_data'].append(mkt)

        # Peers
        peers_data = transform_peers(extract_peers(ticker), ticker)
        if peers_data:
            results['peers'].append(peers_data)

        # Rate limiting - reduced from 1-2 seconds to 0.3-0.5 seconds
        time.sleep(random.uniform(0.3, 0.5))

        results['success'] = True
        print(f"✓ Processed {ticker}")

    except Exception as e:
        print(f"✗ Error processing {ticker}: {str(e)}")

    return results

def process_batch_symbols(**context):
    """Process a batch of symbols with parallel execution and bulk MongoDB operations"""
    batch_size = context.get('batch_size', 10)
    exchange = context.get('exchange', 'US')
    max_workers = context.get('max_workers', 5)  # Number of parallel workers

    tickers = extract_symbols(exchange)[:batch_size]

    # Parallel processing with ThreadPoolExecutor
    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_ticker = {executor.submit(process_single_basic_ticker, ticker): ticker for ticker in tickers}

        # Collect results as they complete
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"✗ Exception for {ticker}: {str(e)}")

    # Bulk write to MongoDB - preserves exact same data structure
    successful_tickers = []

    for result in all_results:
        if not result['success']:
            continue

        ticker = result['ticker']

        try:
            # Companies - single document per ticker
            if result['companies']:
                for profile in result['companies']:
                    db.companies.update_one({"_id": profile["_id"]}, {"$set": profile}, upsert=True)
                print(f"✓ Updated company profile for {ticker}")

            # Earnings - bulk operation
            if result['earnings']:
                operations = [
                    UpdateOne(
                        {"ticker": d["ticker"], "year": d["year"], "quarter": d["quarter"]},
                        {"$set": d},
                        upsert=True
                    )
                    for d in result['earnings']
                ]
                db.earnings_reports.bulk_write(operations)
                print(f"✓ Updated {len(result['earnings'])} earnings records for {ticker}")

            # News - bulk operation
            if result['news']:
                operations = [
                    UpdateOne(
                        {"ticker": d["ticker"], "id": d["id"]},
                        {"$set": d},
                        upsert=True
                    )
                    for d in result['news']
                ]
                db.news.bulk_write(operations)
                print(f"✓ Updated {len(result['news'])} news records for {ticker}")

            # Market Data - single document per ticker
            if result['market_data']:
                for mkt in result['market_data']:
                    db.market_data.update_one(
                        {"ticker": mkt["ticker"], "t": mkt["t"]},
                        {"$set": mkt},
                        upsert=True
                    )
                print(f"✓ Updated market data for {ticker}")

            # Peers - single document per ticker
            if result['peers']:
                for peers_data in result['peers']:
                    db.peers.update_one({"_id": peers_data["_id"]}, {"$set": peers_data}, upsert=True)
                print(f"✓ Updated peers data for {ticker}")

            successful_tickers.append(ticker)

        except Exception as e:
            print(f"✗ Error writing to MongoDB for {ticker}: {str(e)}")

    return f"Processed {len(successful_tickers)}/{len(tickers)} symbols successfully"

def process_single_detailed_ticker(ticker):
    """Process detailed data for a single ticker - used for parallel execution"""
    results = {
        'ticker': ticker,
        'success': False,
        'filings': [],
        'earnings_surprises': [],
        'financials_reported': [],
        'insider_sentiment': [],
        'basic_financials': []
    }

    try:
        # SEC Filings
        filings_data = transform_sec_filings(extract_filings(ticker), ticker)
        if filings_data:
            results['filings'].extend(filings_data)

        # Earnings Surprises
        surprises_data = transform_earnings_surprises(extract_earnings_surprises(ticker), ticker)
        if surprises_data:
            results['earnings_surprises'].extend(surprises_data)

        # Financials Reported
        financials_data = transform_financials_reported(
            extract_financials_reported(ticker, freq="quarterly"), ticker
        )
        if financials_data:
            results['financials_reported'].extend(financials_data)

        # Insider Sentiment
        sentiment_data = transform_insider_sentiment(extract_insider_sentiment(ticker), ticker)
        if sentiment_data:
            results['insider_sentiment'].extend(sentiment_data)

        # Basic Financials
        bf = transform_basic_financials(extract_basic_financials(ticker), ticker)
        if bf:
            results['basic_financials'].append(bf)

        # Rate limiting - reduced from 2-4 seconds to 0.5-1 seconds
        time.sleep(random.uniform(0.5, 1.0))

        results['success'] = True
        print(f"✓ Processed detailed data for {ticker}")

    except Exception as e:
        print(f"✗ Error processing detailed data for {ticker}: {str(e)}")

    return results

def process_detailed_data(**context):
    """Process detailed data for a smaller batch of symbols with parallel execution"""
    batch_size = context.get('batch_size', 5)
    exchange = context.get('exchange', 'US')
    offset = context.get('offset', 0)  # Start offset to avoid duplicate processing
    max_workers = context.get('max_workers', 3)  # Number of parallel workers

    # Get tickers with offset to avoid processing same tickers as basic task
    all_tickers = extract_symbols(exchange)
    tickers = all_tickers[offset:offset + batch_size]

    # Parallel processing
    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(process_single_detailed_ticker, ticker): ticker for ticker in tickers}

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"✗ Exception for {ticker}: {str(e)}")

    # Bulk write to MongoDB
    successful_tickers = []

    for result in all_results:
        if not result['success']:
            continue

        ticker = result['ticker']

        try:
            # SEC Filings - bulk operation
            if result['filings']:
                operations = [
                    UpdateOne(
                        {"ticker": d["ticker"], "accessNumber": d["accessNumber"]},
                        {"$set": d},
                        upsert=True
                    )
                    for d in result['filings']
                ]
                db.sec_fillings.bulk_write(operations)
                print(f"✓ Updated {len(result['filings'])} filing records for {ticker}")

            # Earnings Surprises - bulk operation
            if result['earnings_surprises']:
                operations = [
                    UpdateOne(
                        {"ticker": d["ticker"], "year": d["year"], "quarter": d["quarter"]},
                        {"$set": d},
                        upsert=True
                    )
                    for d in result['earnings_surprises']
                ]
                db.earnings_surprises.bulk_write(operations)
                print(f"✓ Updated {len(result['earnings_surprises'])} earnings surprise records for {ticker}")

            # Financials Reported - bulk operation
            if result['financials_reported']:
                operations = [
                    UpdateOne(
                        {"ticker": d["ticker"], "accessNumber": d["accessNumber"]},
                        {"$set": d},
                        upsert=True
                    )
                    for d in result['financials_reported']
                ]
                db.financials_reported.bulk_write(operations)
                print(f"✓ Updated {len(result['financials_reported'])} financials records for {ticker}")

            # Insider Sentiment - bulk operation
            if result['insider_sentiment']:
                operations = [
                    UpdateOne(
                        {"ticker": d["ticker"], "year": d["year"], "month": d["month"]},
                        {"$set": d},
                        upsert=True
                    )
                    for d in result['insider_sentiment']
                ]
                db.insider_sentiment.bulk_write(operations)
                print(f"✓ Updated {len(result['insider_sentiment'])} insider sentiment records for {ticker}")

            # Basic Financials - single document per ticker
            if result['basic_financials']:
                for bf in result['basic_financials']:
                    db.basic_financials.update_one({"_id": bf["_id"]}, {"$set": bf}, upsert=True)
                print(f"✓ Updated basic financials for {ticker}")

            successful_tickers.append(ticker)

        except Exception as e:
            print(f"✗ Error writing to MongoDB for {ticker}: {str(e)}")

    return f"Processed detailed data for {len(successful_tickers)}/{len(tickers)} symbols successfully"

def check_mongodb_connection(**context):
    """Check if MongoDB is accessible"""
    try:
        # Test connection
        db.command('ping')
        collections = db.list_collection_names()
        print(f"✓ MongoDB connection successful. Found {len(collections)} collections.")
        return True
    except Exception as e:
        print(f"✗ MongoDB connection failed: {str(e)}")
        raise

# Define tasks
check_db_task = PythonOperator(
    task_id='check_mongodb_connection',
    python_callable=check_mongodb_connection,
    dag=dag,
)

process_basic_data_task = PythonOperator(
    task_id='process_basic_data',
    python_callable=process_batch_symbols,
    op_kwargs={'batch_size': 50, 'exchange': 'US', 'max_workers': 5},  # Increased batch size, added parallel workers
    dag=dag,
)

process_detailed_data_task = PythonOperator(
    task_id='process_detailed_data',
    python_callable=process_detailed_data,
    op_kwargs={'batch_size': 20, 'exchange': 'US', 'offset': 50, 'max_workers': 3},  # Increased batch size, offset to avoid duplicates
    dag=dag,
)

# Set task dependencies
check_db_task >> process_basic_data_task >> process_detailed_data_task
