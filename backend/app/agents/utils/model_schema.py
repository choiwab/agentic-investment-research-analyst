from langchain.output_parsers import ResponseSchema

class NewsModel():
    """Model Schema for News Scraper"""
    qualitative_summary : str = ResponseSchema(name = "qualitative_summary", description = "Textual company insights and details")
    quantitative_summary : str = ResponseSchema(name = "quantitative_summary", description = "Numerical/Financial results and figures")
    insight_outlook : str = ResponseSchema(name = "insight_outlook", description = "Insights or Outlook of the company based on summary")
    response_schema = [qualitative_summary, quantitative_summary, insight_outlook]


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