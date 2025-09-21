import os
import re
from typing import Dict, List
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi

class SECRiskAnalyzer:

    def __init__(self):
        load_dotenv()
        self.risk_keywords = {
            'market_risk': [
                'market volatility', 'economic downturn', 'recession', 'inflation',
                'interest rate', 'currency', 'commodity price', 'competition',
                'demand', 'supply chain'
            ],
            'operational_risk': [
                'operational failure', 'system failure', 'cybersecurity',
                'data breach', 'key personnel', 'technology disruption',
                'manufacturing', 'quality control', 'supply chain'
            ],
            'financial_risk': [
                'liquidity', 'credit risk', 'debt', 'cash flow',
                'working capital', 'covenant', 'bankruptcy', 'insolvency',
                'leverage', 'capital'
            ],
            'regulatory_risk': [
                'regulatory', 'compliance', 'government regulation', 'law',
                'environmental regulation', 'safety regulation', 'licensing',
                'permit', 'legal', 'litigation', 'scrutiny'
            ],
            'strategic_risk': [
                'strategic', 'merger', 'acquisition', 'divestiture',
                'market expansion', 'product development', 'innovation',
                'competitive position', 'reputation', 'customer concentration'
            ]
        }
        self.mongo_client = None
        self.mongo_db = None
        mongo_uri = os.getenv('ATLAS_URI')
        mongo_db_name = os.getenv('MONGODB_DB', 'test')

        if not mongo_uri:
            print("ATLAS_URI not set. Create a .env file and add it in")
        try:
            self.mongo_client = MongoClient(
            mongo_uri,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=20000
            )
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
        text_lower = text.lower()
        identified_risks = {}
        
        for risk_category, keywords in self.risk_keywords.items():
            found_risks = set()
            for keyword in keywords:
                if keyword in text_lower:
                    sentences = re.split(r'[.!?]+', text)
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            found_risks.add(sentence.strip())
            identified_risks[risk_category] = list(found_risks)
            
        return identified_risks
    
    def calculate_risk_score(self, risk_factors: Dict[str, List[str]], 
                           financial_metrics: Dict = None) -> Dict[str, float]:
        risk_scores = {}
        
        for category, factors in risk_factors.items():
            if len(factors) > 0:
                score = 0.15 + (len(factors) - 1) * 0.05
                risk_scores[category] = min(score, 1.0)
            else:
                risk_scores[category] = 0.0

        if financial_metrics:
            risk_scores = self._adjust_for_financial_metrics(risk_scores, financial_metrics)
            
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
        adjusted_scores = risk_scores.copy()
        financial_multiplier = 1.0
        operational_multiplier = 1.0

        if metrics.get('debt_to_equity', 0) > 2.5:
            financial_multiplier += 0.5
        elif metrics.get('debt_to_equity', 0) > 1.5:
            financial_multiplier += 0.2

        if metrics.get('current_ratio', 2.0) < 1.0:
            financial_multiplier += 0.4
        elif metrics.get('current_ratio', 2.0) < 1.5:
            financial_multiplier += 0.15
        
        if metrics.get('profit_margin', 0.1) < 0:
            operational_multiplier += 0.5
        elif metrics.get('profit_margin', 0.1) < 0.05:
            operational_multiplier += 0.2

        if adjusted_scores.get('financial_risk', 0) > 0:
            adjusted_scores['financial_risk'] *= financial_multiplier
        else:
             adjusted_scores['financial_risk'] += (financial_multiplier - 1.0) * 0.2
        
        if adjusted_scores.get('operational_risk', 0) > 0:
            adjusted_scores['operational_risk'] *= operational_multiplier
        else:
            adjusted_scores['operational_risk'] += (operational_multiplier - 1.0) * 0.2

        for category in adjusted_scores:
            adjusted_scores[category] = min(adjusted_scores[category], 1.0)
                
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

            return {k: v for k, v in out.items() if v is not None}

        except Exception as e:
            print(f"[SEC-Risk] Query error: {e}")
            return {}
        
    def get_financial_metrics(self, ticker: str) -> Dict:
        metrics = self._get_financial_metrics_from_db(ticker)
        if metrics:
            return metrics
        
        print(f"No financial metrics found in the database for {ticker}. Returning empty.")
        return {}
    
    def analyze_comprehensive_risk(self, text: str, ticker: str = None) -> Dict:
        risk_factors = self.extract_risk_factors(text)
        
        financial_metrics = {}
        if ticker:
            financial_metrics = self.get_financial_metrics(ticker)
            
        risk_scores = self.calculate_risk_score(risk_factors, financial_metrics)
        
        return {
            'risk_factors': risk_factors,
            'risk_scores': risk_scores,
            'financial_metrics': financial_metrics
        }


if __name__ == "__main__":
    ticker = "ORSHF"
    sample_text = (
        "ORSHF reported strong revenue growth but highlighted supply chain risks and regulatory scrutiny. "
        "Management noted rising input costs and potential demand fluctuations in key markets."
    )

    print("Initializing SECRiskAnalyzer...")
    analyzer = SECRiskAnalyzer()

    atlas_uri = os.getenv('ATLAS_URI', 'mongodb://localhost:27017')
    mongo_db_name = os.getenv('MONGODB_DB', 'test')
    print(f"MONGODB_DB: {mongo_db_name}")

    try:
        db_metrics = analyzer._get_financial_metrics_from_db(ticker)
        source = "MongoDB" if db_metrics else "Not Found"
        print(f"Financial metrics source: {source}")
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
