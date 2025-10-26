from langchain.output_parsers import ResponseSchema


class NewsModel():
    """Model Schema for News Scraper"""
    qualitative_summary : str = ResponseSchema(name = "qualitative_summary", description = "Textual company insights and details")
    quantitative_summary : str = ResponseSchema(name = "quantitative_summary", description = "Numerical/Financial results and figures")
    insight_outlook : str = ResponseSchema(name = "insight_outlook", description = "Insights or Outlook of the company based on summary")
    response_schema = [qualitative_summary, quantitative_summary, insight_outlook]

class SentimentModel:
    """Model Schema for Sentiment Analysis with FinBERT and SEC Risk Assessment"""
    positive_sentiment : str = ResponseSchema(name = "positive_sentiment", description = "Positive sentiment score and analysis from FinBERT")
    negative_sentiment : str = ResponseSchema(name = "negative_sentiment", description = "Negative sentiment score and analysis from FinBERT")
    confidence_score : str = ResponseSchema(name = "confidence_score", description = "Overall confidence score of the sentiment analysis (0-1)")
    risk_parameter : str = ResponseSchema(name = "risk_parameter", description = "Comprehensive risk assessment score based on SEC risk factors")
    insight_summary : str = ResponseSchema(name = "insight_summary", description = "Combined insights from sentiment and risk analysis")
    response_schema = [positive_sentiment, negative_sentiment, confidence_score, risk_parameter, insight_summary]




    