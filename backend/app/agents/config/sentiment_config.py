import os
from typing import Dict


class SentimentConfig:
    """Configuration class for sentiment analysis components"""

    FINBERT_MODEL_NAME = "ProsusAI/finbert"
    FINBERT_MAX_LENGTH = 512
    FINBERT_BATCH_SIZE = 1

    SEC_RISK_CATEGORIES = {
        'market_risk': {
            'weight': 0.25,
            'keywords': [
                'market volatility', 'economic downturn', 'recession', 'inflation',
                'interest rate', 'currency fluctuation', 'commodity price',
                'market competition', 'demand fluctuation', 'supply chain'
            ]
        },
        'operational_risk': {
            'weight': 0.20,
            'keywords': [
                'operational failure', 'system failure', 'cybersecurity',
                'data breach', 'regulatory compliance', 'key personnel',
                'technology disruption', 'manufacturing', 'quality control',
                'supply chain disruption'
            ]
        },
        'financial_risk': {
            'weight': 0.30,
            'keywords': [
                'liquidity', 'credit risk', 'debt', 'cash flow',
                'working capital', 'covenant', 'bankruptcy', 'insolvency',
                'financial leverage', 'capital requirements'
            ]
        },
        'regulatory_risk': {
            'weight': 0.15,
            'keywords': [
                'regulatory change', 'compliance', 'government regulation',
                'tax law', 'environmental regulation', 'safety regulation',
                'licensing', 'permit', 'legal proceeding', 'litigation'
            ]
        },
        'strategic_risk': {
            'weight': 0.10,
            'keywords': [
                'strategic initiative', 'merger', 'acquisition', 'divestiture',
                'market expansion', 'product development', 'innovation',
                'competitive position', 'brand reputation', 'customer concentration'
            ]
        }
    }

    RISK_THRESHOLDS = {
        'low': 0.3,
        'moderate': 0.6,
        'high': 1.0
    }

    # Sentiment Score Thresholds
    SENTIMENT_THRESHOLDS = {
        'strong_negative': -0.5,
        'negative': -0.1,
        'neutral': 0.1,
        'positive': 0.5,
        'strong_positive': 1.0
    }

    # Financial Metrics Thresholds for Risk Adjustment
    FINANCIAL_THRESHOLDS = {
        'debt_to_equity': {
            'high_risk': 2.0,
            'moderate_risk': 1.0
        },
        'current_ratio': {
            'low_liquidity': 1.0
        },
        'profit_margin': {
            'negative': 0.0
        }
    }

    LLM_CONFIG = {
        'model': "gpt-5-nano",
        'temperature': 0.0,
        'streaming': True,
        'timeout': 300
    }

    OUTPUT_FORMAT = {
        'include_raw_scores': True,
        'include_risk_breakdown': True,
        'include_financial_metrics': True,
        'decimal_precision': 3
    }

    @classmethod
    def get_risk_category_weights(cls) -> Dict[str, float]:
        """Get risk category weights for overall score calculation"""
        return {category: config['weight'] for category, config in cls.SEC_RISK_CATEGORIES.items()}

    @classmethod
    def get_all_risk_keywords(cls) -> Dict[str, list]:
        """Get all risk keywords organized by category"""
        return {category: config['keywords'] for category, config in cls.SEC_RISK_CATEGORIES.items()}

    @classmethod
    def classify_risk_level(cls, risk_score: float) -> str:
        """Classify risk score into categorical level"""
        if risk_score <= cls.RISK_THRESHOLDS['low']:
            return 'Low'
        elif risk_score <= cls.RISK_THRESHOLDS['moderate']:
            return 'Moderate'
        else:
            return 'High'

    @classmethod
    def classify_sentiment(cls, sentiment_score: float) -> str:
        """Classify sentiment score into categorical level"""
        if sentiment_score <= cls.SENTIMENT_THRESHOLDS['strong_negative']:
            return 'Strong Negative'
        elif sentiment_score <= cls.SENTIMENT_THRESHOLDS['negative']:
            return 'Negative'
        elif sentiment_score <= cls.SENTIMENT_THRESHOLDS['neutral']:
            return 'Neutral'
        elif sentiment_score <= cls.SENTIMENT_THRESHOLDS['positive']:
            return 'Positive'
        else:
            return 'Strong Positive'

    @classmethod
    def get_investment_signal(cls, sentiment: str, confidence: float, risk_level: str) -> Dict[str, str]:
        """Generate investment signal based on sentiment and risk analysis"""

        signals = {
            'strong_positive_low_risk': {
                'signal': 'STRONG BUY',
                'description': 'Strong positive sentiment with low risk factors'
            },
            'positive_low_risk': {
                'signal': 'BUY',
                'description': 'Positive sentiment with manageable risk'
            },
            'neutral_low_risk': {
                'signal': 'HOLD',
                'description': 'Neutral sentiment but low risk environment'
            },
            'negative_any_risk': {
                'signal': 'SELL',
                'description': 'Negative sentiment indicates caution'
            },
            'any_high_risk': {
                'signal': 'AVOID',
                'description': 'High risk factors present'
            },
            'low_confidence': {
                'signal': 'MONITOR',
                'description': 'Low confidence in analysis, seek more data'
            }
        }

        # Decision logic
        if confidence < 0.5:
            return signals['low_confidence']
        elif risk_level == 'High':
            return signals['any_high_risk']
        elif 'Negative' in sentiment:
            return signals['negative_any_risk']
        elif sentiment == 'Strong Positive' and risk_level == 'Low':
            return signals['strong_positive_low_risk']
        elif sentiment == 'Positive' and risk_level in ['Low', 'Moderate']:
            return signals['positive_low_risk']
        else:
            return signals['neutral_low_risk']

# Environment-specific configurations


class DevelopmentConfig(SentimentConfig):
    """Development environment configuration"""
    DEBUG = True
    VERBOSE_LOGGING = True


class ProductionConfig(SentimentConfig):
    """Production environment configuration"""
    DEBUG = False
    VERBOSE_LOGGING = False

    # More conservative thresholds for production
    RISK_THRESHOLDS = {
        'low': 0.25,
        'moderate': 0.55,
        'high': 1.0
    }

# Configuration factory


def get_config() -> SentimentConfig:
    """Get configuration based on environment"""
    env = os.getenv('ENVIRONMENT', 'development').lower()

    if env == 'production':
        return ProductionConfig()
    else:
        return DevelopmentConfig()
