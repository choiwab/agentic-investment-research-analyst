from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
import sys
import os

sys.path.append('/opt/airflow/etl')
#import etl_script
from etl import etl_script

# Import your ETL functionss
from etl.etl_script import (
    extract_symbols, extract_company_profile, extract_earnings,
    extract_news, extract_filings, extract_quote, extract_earnings_surprises,
    extract_financials_reported, extract_insider_sentiment, extract_basic_financials,
    transform_company_profile, transform_earnings, transform_news,
    transform_sec_filings, transform_market_data, transform_earnings_surprises,
    transform_financials_reported, transform_insider_sentiment, transform_basic_financials,
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
    schedule_interval=timedelta(hours=6), 
        # schedule_interval='0 9 * * 1-5' - Weekdays at 9 AM
        # schedule_interval=timedelta(hours=4) - Every 4 hours
        # schedule_interval='@daily' - Once per day
        # schedule_interval=None - Manual trigger only
    catchup=False,
    max_active_runs=1,
)

def process_batch_symbols(**context):
    """Process a batch of symbols"""
    batch_size = context.get('batch_size', 10)
    exchange = context.get('exchange', 'US')
    
    tickers = extract_symbols(exchange)[:batch_size]
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        
        try:
            # Companies
            profile = transform_company_profile(extract_company_profile(ticker))
            if profile:
                db.companies.update_one({"_id": profile["_id"]}, {"$set": profile}, upsert=True)
                print(f"✓ Updated company profile for {ticker}")
            
            # Earnings
            earnings_data = transform_earnings(extract_earnings(ticker), ticker)
            for d in earnings_data:
                db.earnings_reports.update_one(
                    {"ticker": d["ticker"], "year": d["year"], "quarter": d["quarter"]},
                    {"$set": d},
                    upsert=True
                )
            if earnings_data:
                print(f"✓ Updated {len(earnings_data)} earnings records for {ticker}")
            
            # News
            news_data = transform_news(extract_news(ticker), ticker)
            for d in news_data:
                db.news.update_one(
                    {"ticker": d["ticker"], "id": d["id"]},
                    {"$set": d},
                    upsert=True
                )
            if news_data:
                print(f"✓ Updated {len(news_data)} news records for {ticker}")
            
            # Market Data
            mkt = transform_market_data(extract_quote(ticker), ticker)
            if mkt:
                db.market_data.update_one(
                    {"ticker": mkt["ticker"], "t": mkt["t"]},
                    {"$set": mkt},
                    upsert=True
                )
                print(f"✓ Updated market data for {ticker}")
            
            # Rate limiting
            time.sleep(random.uniform(1.0, 2.0))
            
        except Exception as e:
            print(f"✗ Error processing {ticker}: {str(e)}")
            continue
    
    return f"Processed {len(tickers)} symbols successfully"

def process_detailed_data(**context):
    """Process detailed data for a smaller batch of symbols"""
    batch_size = context.get('batch_size', 5)
    exchange = context.get('exchange', 'US')
    
    tickers = extract_symbols(exchange)[:batch_size]
    
    for ticker in tickers:
        print(f"Processing detailed data for {ticker}...")
        
        try:
            # SEC Filings
            filings_data = transform_sec_filings(extract_filings(ticker), ticker)
            for d in filings_data:
                db.sec_fillings.update_one(
                    {"ticker": d["ticker"], "accessNumber": d["accessNumber"]},
                    {"$set": d},
                    upsert=True
                )
            if filings_data:
                print(f"✓ Updated {len(filings_data)} filing records for {ticker}")
            
            # Earnings Surprises
            surprises_data = transform_earnings_surprises(extract_earnings_surprises(ticker), ticker)
            for d in surprises_data:
                db.earnings_surprises.update_one(
                    {"ticker": d["ticker"], "year": d["year"], "quarter": d["quarter"]},
                    {"$set": d},
                    upsert=True
                )
            if surprises_data:
                print(f"✓ Updated {len(surprises_data)} earnings surprise records for {ticker}")
            
            # Financials Reported
            financials_data = transform_financials_reported(
                extract_financials_reported(ticker, freq="quarterly"), ticker
            )
            for d in financials_data:
                db.financials_reported.update_one(
                    {"ticker": d["ticker"], "accessNumber": d["accessNumber"]},
                    {"$set": d},
                    upsert=True
                )
            if financials_data:
                print(f"✓ Updated {len(financials_data)} financials records for {ticker}")
            
            # Insider Sentiment
            sentiment_data = transform_insider_sentiment(extract_insider_sentiment(ticker), ticker)
            for d in sentiment_data:
                db.insider_sentiment.update_one(
                    {"ticker": d["ticker"], "year": d["year"], "month": d["month"]},
                    {"$set": d},
                    upsert=True
                )
            if sentiment_data:
                print(f"✓ Updated {len(sentiment_data)} insider sentiment records for {ticker}")
            
            # Basic Financials
            bf = transform_basic_financials(extract_basic_financials(ticker), ticker)
            if bf:
                db.basic_financials.update_one({"_id": bf["_id"]}, {"$set": bf}, upsert=True)
                print(f"✓ Updated basic financials for {ticker}")
            
            # Rate limiting
            time.sleep(random.uniform(2.0, 4.0))
            
        except Exception as e:
            print(f"✗ Error processing detailed data for {ticker}: {str(e)}")
            continue
    
    return f"Processed detailed data for {len(tickers)} symbols successfully"

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
    op_kwargs={'batch_size': 20, 'exchange': 'US'},
    dag=dag,
)

process_detailed_data_task = PythonOperator(
    task_id='process_detailed_data',
    python_callable=process_detailed_data,
    op_kwargs={'batch_size': 5, 'exchange': 'US'},
    dag=dag,
)

# Set task dependencies
check_db_task >> process_basic_data_task >> process_detailed_data_task