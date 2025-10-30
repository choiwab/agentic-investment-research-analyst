from langchain.output_parsers import ResponseSchema
from pydantic import BaseModel, Field, field_validator
from typing import ClassVar, List, Optional



class NewsModel():
    """Model Schema for News Scraper"""
    qualitative_summary : str = ResponseSchema(name = "qualitative_summary", description = "Textual company insights and details")
    quantitative_summary : str = ResponseSchema(name = "quantitative_summary", description = "Numerical/Financial results and figures")
    insight_outlook : str = ResponseSchema(name = "insight_outlook", description = "Insights or Outlook of the company based on summary")
    response_schema = [qualitative_summary, quantitative_summary, insight_outlook]

class PreprocessModel(BaseModel):
    """Validated output model for preprocessing agent."""

    query: Optional[str] = Field(default=None)
    intent: Optional[str] = Field(default=None)
    ticker: Optional[str] = Field(default=None)
    peers: Optional[List[str]] = Field(default=None)  
    timeframe: Optional[str] = Field(default=None)
    metrics: Optional[List[str]] = Field(default=None)  
    url: Optional[str] = Field(default=None)
    output_from_websearch: Optional[str] = Field(default=None)
    answer: Optional[str] = Field(default=None)

    @field_validator('peers', 'metrics', mode='before')
    @classmethod
    def convert_string_to_list(cls, v):
        """Auto-convert string inputs to lists"""
        if v is None:
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            if v.strip() == '' or v.strip().lower() in ['null', 'none']:
                return None
            items = [item.strip() for item in v.split(',') if item.strip()]
            return items if items else None
        return v

    @field_validator('output_from_websearch', mode='before')
    @classmethod
    def convert_list_to_string(cls, v):
        """Convert list to string for websearch output"""
        if v is None:
            return None
        if isinstance(v, list):
            if v and isinstance(v[0], dict):
                result = []
                for item in v:
                    if 'text' in item and item['text']:
                        result.append(item['text'][:500])  
                return '\n\n'.join(result) if result else None
            return str(v)
        return str(v) if v else None

    class Config:
        extra = "ignore"
class SentimentModel:
    """Model Schema for Sentiment Analysis with FinBERT and SEC Risk Assessment"""
    positive_sentiment : str = ResponseSchema(name = "positive_sentiment", description = "Positive sentiment score and analysis from FinBERT")
    negative_sentiment : str = ResponseSchema(name = "negative_sentiment", description = "Negative sentiment score and analysis from FinBERT")
    confidence_score : str = ResponseSchema(name = "confidence_score", description = "Overall confidence score of the sentiment analysis (0-1)")
    risk_parameter : str = ResponseSchema(name = "risk_parameter", description = "Comprehensive risk assessment score based on SEC risk factors")
    insight_summary : str = ResponseSchema(name = "insight_summary", description = "Combined insights from sentiment and risk analysis")
    response_schema = [positive_sentiment, negative_sentiment, confidence_score, risk_parameter, insight_summary]


class ResearchCompilerModel():
    """Model Schema for Research Compiler"""
    executive_summary : str = ResponseSchema(
        name = "executive_summary",
        description = "2-3 sentence executive summary of key findings and investment thesis"
    )
    financial_analysis : str = ResponseSchema(
        name = "financial_analysis",
        description = "Analysis of financial metrics, trends, and performance indicators"
    )
    news_sentiment_analysis : str = ResponseSchema(
        name = "news_sentiment_analysis",
        description = "Summary of news sentiment and key themes from qualitative analysis"
    )
    investment_outlook : str = ResponseSchema(
        name = "investment_outlook",
        description = "Forward-looking analysis including risks, opportunities, and catalysts"
    )
    recommendation : str = ResponseSchema(
        name = "recommendation",
        description = "Clear investment recommendation (BUY/HOLD/SELL) with supporting rationale"
    )
    price_target : str = ResponseSchema(
        name = "price_target",
        description = "12-month price target if applicable, or 'N/A' if not determined",
        required = False
    )

    response_schema = [
        executive_summary,
        financial_analysis,
        news_sentiment_analysis,
        investment_outlook,
        recommendation,
        price_target
    ]
    intent_schema: ClassVar = ResponseSchema(name="intent", description="Classified category of query.")
    ticker_schema: ClassVar = ResponseSchema(name="ticker", description="Company ticker symbol if mentioned.")
    peers_schema: ClassVar = ResponseSchema(name="peers", description="Peer tickers if comparison requested.")
    timeframe_schema: ClassVar = ResponseSchema(name="timeframe", description="Time period extracted or defaulted.")
    metrics_schema: ClassVar = ResponseSchema(name="metrics", description="Financial metrics extracted.")
    url_schema: ClassVar = ResponseSchema(name="url", description="Company data URL.")
    output_schema: ClassVar = ResponseSchema(name="output_from_websearch", description="Web search result text.")
    answer_schema: ClassVar = ResponseSchema(name="answer", description="Conceptual or educational answer.")

    response_schema: ClassVar[List[ResponseSchema]] = [
        intent_schema,
        ticker_schema,
        peers_schema,
        timeframe_schema,
        metrics_schema,
        url_schema,
        output_schema,
        answer_schema
    ]

