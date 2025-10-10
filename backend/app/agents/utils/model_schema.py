from langchain.output_parsers import ResponseSchema

class NewsModel():
    """Model Schema for News Scraper"""
    qualitative_summary : str = ResponseSchema(name = "qualitative_summary", description = "Textual company insights and details")
    quantitative_summary : str = ResponseSchema(name = "quantitative_summary", description = "Numerical/Financial results and figures")
    insight_outlook : str = ResponseSchema(name = "insight_outlook", description = "Insights or Outlook of the company based on summary")
    response_schema = [qualitative_summary, quantitative_summary, insight_outlook]


class MetricModel():
    """Model Schema for Metric Extractor Agent"""
    ticker : str = ResponseSchema(
        name = "ticker",
        description = "Stock ticker symbol"
    )
    timeframe : str = ResponseSchema(
        name = "timeframe",
        description = "Time period analyzed (e.g., 'Q1 2024', 'Q2 2025')"
    )
    financials : str = ResponseSchema(
        name = "financials",
        description = "JSON object containing revenue (actual, estimate, growthYoY), eps (actual, estimate, surprisePercent), and margins (operatingMargin, netMargin)"
    )
    valuation : str = ResponseSchema(
        name = "valuation",
        description = "JSON object containing valuation ratios: peRatio, pbRatio, psRatio"
    )
    market : str = ResponseSchema(
        name = "market",
        description = "JSON object containing market data: price, 52WeekHigh, 52WeekLow, volatility"
    )
    insider_sentiment : str = ResponseSchema(
        name = "insider_sentiment",
        description = "JSON object containing insider sentiment: change (net insider buys/sells), mspr (monthly share purchase ratio)"
    )
    quantitative_news : str = ResponseSchema(
        name = "quantitative_news",
        description = "JSON array of news items with source, date, headline, summary extracted from news scraper agent output"
    )
    metric_evaluation : str = ResponseSchema(
        name = "metric_evaluation",
        description = "JSON object containing overallPerformance (Strong/Moderate/Weak), keyDrivers (array), negativeAnomalies (array), positiveAnomalies (array), risks (array)"
    )
    response_schema = [
        ticker, timeframe, financials, valuation, market,
        insider_sentiment, quantitative_news, metric_evaluation
    ]




    