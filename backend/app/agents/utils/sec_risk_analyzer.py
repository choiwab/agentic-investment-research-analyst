import os
import re
import time
from typing import Dict, List
from dotenv import load_dotenv
import yfinance as yf
from pymongo import MongoClient
import certifi


class SECRiskAnalyzer:

    def __init__(self):
        load_dotenv()
        self.risk_keywords = {
            'market_risk': [
                'market volatility', 'economic downturn', 'recession', 'inflation',
                'interest rate', 'currency fluctuation', 'commodity price',
                'market competition', 'demand fluctuation', 'supply chain'
            ],
            'operational_risk': [
                'operational failure', 'system failure', 'cybersecurity',
                'data breach', 'regulatory compliance', 'key personnel',
                'technology disruption', 'manufacturing', 'quality control',
                'supply chain disruption'
            ],
            'financial_risk': [
                'liquidity', 'credit risk', 'debt', 'cash flow',
                'working capital', 'covenant', 'bankruptcy', 'insolvency',
                'financial leverage', 'capital requirements'
            ],
            'regulatory_risk': [
                'regulatory change', 'compliance', 'government regulation',
                'tax law', 'environmental regulation', 'safety regulation',
                'licensing', 'permit', 'legal proceeding', 'litigation'
            ],
            'strategic_risk': [
                'strategic initiative', 'merger', 'acquisition', 'divestiture',
                'market expansion', 'product development', 'innovation',
                'competitive position', 'brand reputation', 'customer concentration'
            ]
        }
        self.mongo_client = None
        self.mongo_db = None
        mongo_uri = os.getenv('ATLAS_URI')
        mongo_db_name = os.getenv('MONGODB_DB', 'test')

        if not mongo_uri:
            print("ATLAS_URI not set. Create a .env file and add it in")
        try:
            # SRV needs dnspython: pip install "pymongo[srv]"
            self.mongo_client = MongoClient(
            mongo_uri,
            tlsCAFile=certifi.where(),          # <-- keep this
            serverSelectionTimeoutMS=20000
            )
            # Force a connection attempt + show where we connected
            self.mongo_client.admin.command("ping")
            info = self.mongo_client.server_info()
            print(f"[SEC-Risk] Connected to MongoDB {info.get('version','?')}")
            self.mongo_db = self.mongo_client[mongo_db_name]
            print(f"[SEC-Risk] Using database: {self.mongo_db.name}")
        except Exception as e:
            print(f"[SEC-Risk] Mongo connection failed: {e}")
            self.mongo_client = None
            self.mongo_db = None
        
    def extract_risk_factors(self, text: str) -> Dict[str, List[str]]:
        """Extract risk factors from text using keyword matching"""
        text_lower = text.lower()
        identified_risks = {}
        
        for risk_category, keywords in self.risk_keywords.items():
            found_risks = []
            for keyword in keywords:
                if keyword in text_lower:
                    # Extract sentence containing the keyword
                    sentences = re.split(r'[.!?]+', text)
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            found_risks.append(sentence.strip())
                            break
            identified_risks[risk_category] = found_risks
            
        return identified_risks
    
    def calculate_risk_score(self, risk_factors: Dict[str, List[str]], 
                           financial_metrics: Dict = None) -> Dict[str, float]:
        """Calculate risk scores based on identified factors and financial metrics"""
        risk_scores = {}
        
        # Base risk calculation from text analysis
        for category, factors in risk_factors.items():
            base_score = min(len(factors) * 0.1, 1.0)  # Cap at 1.0
            risk_scores[category] = base_score
            
        # Adjust scores based on financial metrics if available
        if financial_metrics:
            risk_scores = self._adjust_for_financial_metrics(risk_scores, financial_metrics)
            
        # Calculate overall risk score
        category_weights = {
            'market_risk': 0.25,
            'operational_risk': 0.20,
            'financial_risk': 0.30,
            'regulatory_risk': 0.15,
            'strategic_risk': 0.10
        }
        
        overall_risk = sum(
            risk_scores.get(category, 0) * weight 
            for category, weight in category_weights.items()
        )
        
        risk_scores['overall'] = min(overall_risk, 1.0)
        return risk_scores
    
    def _adjust_for_financial_metrics(self, risk_scores: Dict[str, float], 
                                    metrics: Dict) -> Dict[str, float]:
        """Adjust risk scores based on financial health indicators"""
        adjusted_scores = risk_scores.copy()
        
        # Debt-to-equity ratio adjustment
        if 'debt_to_equity' in metrics:
            debt_ratio = metrics['debt_to_equity']
            if debt_ratio > 2.0:  # High leverage
                adjusted_scores['financial_risk'] = min(adjusted_scores.get('financial_risk', 0) + 0.3, 1.0)
            elif debt_ratio > 1.0:
                adjusted_scores['financial_risk'] = min(adjusted_scores.get('financial_risk', 0) + 0.1, 1.0)
                
        # Current ratio adjustment
        if 'current_ratio' in metrics:
            current_ratio = metrics['current_ratio']
            if current_ratio < 1.0:  # Liquidity concerns
                adjusted_scores['financial_risk'] = min(adjusted_scores.get('financial_risk', 0) + 0.2, 1.0)
                
        if 'profit_margin' in metrics:
            margin = metrics['profit_margin']
            if margin < 0:  # Negative margins
                adjusted_scores['operational_risk'] = min(adjusted_scores.get('operational_risk', 0) + 0.2, 1.0)
                
        return adjusted_scores

    def _get_financial_metrics_from_db(self, ticker: str) -> dict:
        if self.mongo_db is None:
            print("[SEC-Risk] No DB handle (connect failed).")
            return {}

        try:
            col = self.mongo_db['basic_financials']
            doc = col.find_one({'ticker': ticker}) or col.find_one({'_id': ticker})
            if not doc:
                print(f"[SEC-Risk] No basic_financials doc for {ticker}.")
                return {}

            metric = (doc.get('metric') or {})
            if not metric:
                print(f"[SEC-Risk] Doc found for {ticker} but 'metric' is empty.")
                return {}

            # Simple percent→ratio normalization: if abs(val) > 5, treat as %.
            debt_to_equity = (
                metric.get('totalDebt/totalEquityQuarterly')
                if metric.get('totalDebt/totalEquityQuarterly') is not None
                else metric.get('totalDebt/totalEquityAnnual')
            )
            if debt_to_equity is None:
                debt_to_equity = (
                    metric.get('longTermDebt/equityQuarterly')
                    if metric.get('longTermDebt/equityQuarterly') is not None
                    else metric.get('longTermDebt/equityAnnual')
                )
            if debt_to_equity is not None and abs(debt_to_equity) > 5:
                debt_to_equity = debt_to_equity / 100.0

            current_ratio = (
                metric.get('currentRatioQuarterly')
                if metric.get('currentRatioQuarterly') is not None
                else metric.get('currentRatioAnnual')
            )

            profit_margin = (
                metric.get('netProfitMarginTTM')
                if metric.get('netProfitMarginTTM') is not None
                else metric.get('netProfitMarginAnnual')
            )
            if profit_margin is not None and abs(profit_margin) > 5:
                profit_margin = profit_margin / 100.0

            beta = metric.get('beta')

            pe_ratio = (
                metric.get('peTTM')
                or metric.get('peInclExtraTTM')
                or metric.get('peExclExtraTTM')
                or metric.get('peAnnual')
                or metric.get('trailingPE')
                or metric.get('peBasicExclExtraTTM')
            )

            out = {
                'debt_to_equity': debt_to_equity,
                'current_ratio': current_ratio,
                'profit_margin': profit_margin,
                'beta': beta,
                'pe_ratio': pe_ratio
            }

            # return only populated keys
            return {k: v for k, v in out.items() if v is not None}

        except Exception as e:
            print(f"[SEC-Risk] Query error: {e}")
            return {}
        
    def get_financial_metrics(self, ticker: str) -> Dict:
        """Fetch basic financial metrics, preferring MongoDB then falling back to yfinance."""

        # 1) Try MongoDB first
        metrics = self._get_financial_metrics_from_db(ticker)
        if metrics:
            return metrics
        print("Mongodb doesnt work")
        
    
        # 2) Fallback to yfinance with retry/backoff and safe fallbacks
        delays = [1, 3, 6]
        last_error = None
        for attempt, delay in enumerate(delays, start=1):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                yf_metrics = {
                    'debt_to_equity': (info.get('debtToEquity', 0) / 100) if info.get('debtToEquity') else None,
                    'current_ratio': info.get('currentRatio', None),
                    'profit_margin': info.get('profitMargins', None),
                    'beta': info.get('beta', None),
                    'pe_ratio': info.get('trailingPE', None)
                }
                cleaned = { k: v for k, v in yf_metrics.items() if v is not None }
                if cleaned:
                    return cleaned
            except Exception as e:
                last_error = e
                message = str(e)
                if '429' in message or 'Too Many Requests' in message:
                    time.sleep(delay * 2)
                else:
                    time.sleep(delay)
        try:
            stock = yf.Ticker(ticker)
            fast = getattr(stock, 'fast_info', None)
            if fast:
                yf_fast_metrics = {
                    'beta': getattr(fast, 'beta', None),
                    'pe_ratio': getattr(fast, 'trailing_pe', None)
                }
                cleaned = { k: v for k, v in yf_fast_metrics.items() if v is not None }
                if cleaned:
                    return cleaned
        except Exception as e:
            last_error = e
        print(f"Error fetching financial metrics for {ticker}: {last_error}")
        return {}
    
    def analyze_comprehensive_risk(self, text: str, ticker: str = None) -> Dict:
        """Perform comprehensive risk analysis combining text and financial data"""
        # Extract risk factors from text
        risk_factors = self.extract_risk_factors(text)
        
        # Get financial metrics if ticker provided
        financial_metrics = {}
        if ticker:
            financial_metrics = self.get_financial_metrics(ticker)
            
        # Calculate risk scores
        risk_scores = self.calculate_risk_score(risk_factors, financial_metrics)
        
        return {
            'risk_factors': risk_factors,
            'risk_scores': risk_scores,
            'financial_metrics': financial_metrics
        }


if __name__ == "__main__":
    # Basic manual test for SECRiskAnalyzer with MongoDB preference
    ticker = "AAPJ"
    sample_text = (
        "Apple reported strong revenue growth but highlighted supply chain risks and regulatory scrutiny. "
        "Management noted rising input costs and potential demand fluctuations in key markets."
    )

    print("Initializing SECRiskAnalyzer...")
    analyzer = SECRiskAnalyzer()

    atlas_uri = os.getenv('ATLAS_URI', 'mongodb://localhost:27017')
    mongo_db_name = os.getenv('MONGODB_DB', 'test')
    # print(f"ATLAS_URI: {atlas_uri}")
    print(f"MONGODB_DB: {mongo_db_name}")

    # Check DB metrics availability
    try:
        db_metrics = analyzer._get_financial_metrics_from_db(ticker)
        source = "MongoDB" if db_metrics else "yfinance/fallback"
        print(f"Financial metrics source (intended): {source}")
        if db_metrics:
            print(f"Sample DB metrics: { {k: db_metrics[k] for k in list(db_metrics)[:4]} }")
    except Exception as e:
        print(f"DB metrics check failed: {e}")

    print("\nRunning comprehensive risk analysis...\n")
    result = analyzer.analyze_comprehensive_risk(sample_text, ticker)
    scores = result.get('risk_scores', {})
    factors = result.get('risk_factors', {})
    metrics = result.get('financial_metrics', {})

    print("Risk Scores:")
    for k in sorted(scores.keys()):
        print(f"- {k}: {scores[k]:.3f}")

    print("\nRisk Factors (counts):")
    for k in sorted(factors.keys()):
        print(f"- {k}: {len(factors[k])}")

    print("\nFinancial Metrics used:")
    if metrics:
        for k in sorted(metrics.keys()):
            print(f"- {k}: {metrics[k]}")
    else:
        print("- None")

    print("\nDone.")