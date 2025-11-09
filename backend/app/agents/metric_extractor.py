import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.output_parsers import StructuredOutputParser
from langchain.tools import BaseTool

# Agent Setup and Structuring Output
from langchain_openai import ChatOpenAI

# Import utils function and models
from utils.callback_handler import PrintCallbackHandler
from utils.conversation_buffer_safe import SafeConversationMemory
from utils.model_schema import MetricModel
from utils.tools import (
    detect_anomalies,
    fetch_ticker_data,
    fetch_ticker_summary,
    normalize_financial_data,
)

load_dotenv(override=True)


class MetricExtractorAgent:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.callback_handler = PrintCallbackHandler()
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,
            streaming=True,
            callbacks=[self.callback_handler]
        )
        self.memory = SafeConversationMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.parser = StructuredOutputParser.from_response_schemas(MetricModel.response_schema)
        self.agent = self.build_agent()

    def build_agent(self) -> AgentExecutor:
        """Builds a REACT Agent for metric extraction"""
        format_instructions: str = self.parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

        system_template: str = f"""
You are a senior quantitative analyst specializing in financial metrics extraction and evaluation.

Your task is to analyze comprehensive financial data for companies and produce structured investment metrics.

Given raw financial data, you must:

1. **Extract Key Financials**:
   - Revenue: actual, estimate, year-over-year growth
   - EPS: actual, estimate, surprise percentage
   - Margins: operating margin, net margin

2. **Extract Valuation Metrics**:
   - P/E Ratio (Price-to-Earnings)
   - P/B Ratio (Price-to-Book)
   - P/S Ratio (Price-to-Sales)

3. **Extract Market Data**:
   - Current price
   - 52-week high and low
   - Volatility (if available)

4. **Analyze Insider Sentiment**:
   - Net insider buying/selling (change)
   - Monthly share purchase ratio (MSPR)

5. **Synthesize Quantitative News**:
   - Extract key quantitative details from news summaries provided

6. **Provide Metric Evaluation**:
   - Overall Performance: Strong/Moderate/Weak based on metrics
   - Key Drivers: What's driving performance (e.g., "Revenue growth in Asia", "Margin expansion")
   - Positive Anomalies: Unusual positive signals (e.g., "EPS beat by 25%", "Strong insider buying")
   - Negative Anomalies: Concerning signals (e.g., "Revenue miss", "Elevated valuation")
   - Risks: Identified risk factors (e.g., "Insider selling", "Compressed margins")

**Important Guidelines**:
- Handle missing data gracefully by marking as null or "N/A"
- Use consistent units (revenue in millions, margins as decimals)
- Be objective and data-driven in your evaluation
- Flag significant anomalies (>20% surprises, unusual insider activity)
- Consider both quantitative metrics and qualitative context

Always output your final answer strictly as JSON in this format:
{format_instructions}

Use the available tools to fetch and process financial data.
"""

        return initialize_agent(
            tools=self.get_tools(),
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            agent_kwargs={"system_message": system_template},
            handle_parsing_errors=True
        )

    def get_tools(self) -> List[BaseTool]:
        """Returns the tools available to the agent"""
        return [
            fetch_ticker_data,
            fetch_ticker_summary
        ]

    def _parse_timeframe(self, timeframe: Optional[str]) -> tuple:
        """
        Parse timeframe string to extract year and quarter.

        Args:
            timeframe: String like "Q1 2024" or "2024"

        Returns:
            Tuple of (year, quarter)
        """
        if not timeframe:
            return None, None

        year = None
        quarter = None

        # Try to extract year (4-digit number)
        import re
        year_match = re.search(r'\b(20\d{2})\b', timeframe)
        if year_match:
            year = int(year_match.group(1))

        # Try to extract quarter (Q1, Q2, Q3, Q4)
        quarter_match = re.search(r'Q([1-4])', timeframe, re.IGNORECASE)
        if quarter_match:
            quarter = int(quarter_match.group(1))

        return year, quarter

    def _filter_by_timeframe(self, data: Dict[str, Any], year: int = None, quarter: int = None) -> Dict[str, Any]:
        """
        Internal method to filter financial data by specific timeframe.

        Args:
            data: Dictionary containing ticker financial data
            year: Target year (e.g., 2024)
            quarter: Target quarter (1-4)

        Returns:
            Filtered data matching the specified timeframe
        """
        filtered = {}

        # Filter earnings reports
        if data.get("earnings_reports"):
            filtered["earnings_reports"] = [
                e for e in data["earnings_reports"]
                if (year is None or e.get("year") == year) and
                   (quarter is None or e.get("quarter") == quarter)
            ]

        # Filter earnings surprises
        if data.get("earnings_surprises"):
            filtered["earnings_surprises"] = [
                e for e in data["earnings_surprises"]
                if (year is None or e.get("year") == year) and
                   (quarter is None or e.get("quarter") == quarter)
            ]

        # Filter financials reported
        if data.get("financials_reported"):
            filtered["financials_reported"] = [
                f for f in data["financials_reported"]
                if (year is None or f.get("year") == year) and
                   (quarter is None or f.get("quarter") == quarter)
            ]

        # Filter insider sentiment by year
        if data.get("insider_sentiment") and year:
            filtered["insider_sentiment"] = [
                i for i in data["insider_sentiment"]
                if i.get("year") == year
            ]

        # Keep non-time-series data as-is
        for key in ["company", "market_data", "basic_financials", "news", "sec_filings"]:
            if data.get(key):
                filtered[key] = data[key]

        return filtered

    def _process_single_ticker(
        self,
        ticker: str,
        timeframe: Optional[str] = None,
        news_scraper_output: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process a single ticker to extract metrics.

        Args:
            ticker: Stock ticker symbol
            timeframe: Optional timeframe (e.g., "Q1 2024")
            news_scraper_output: Optional output from news scraper agent

        Returns:
            Dictionary with extracted metrics
        """
        # Fetch comprehensive data
        raw_data = fetch_ticker_data.invoke({"tickers": [ticker]})
        ticker_data = raw_data.get(ticker, {})

        # Normalize data
        normalized_data = normalize_financial_data(ticker_data)

        # Filter by timeframe if provided
        year, quarter = self._parse_timeframe(timeframe)
        if year or quarter:
            normalized_data = self._filter_by_timeframe(normalized_data, year, quarter)

        # Detect anomalies
        anomalies = detect_anomalies(normalized_data)

        # Prepare context for agent
        context = {
            "ticker": ticker,
            "timeframe": timeframe or "Latest available",
            "data": normalized_data,
            "anomalies": anomalies,
            "news_summary": news_scraper_output or {}
        }

        # Build prompt for agent
        prompt = f"""
Analyze the following financial data for {ticker} and extract structured metrics.

Timeframe: {context['timeframe']}

Raw Financial Data:
{json.dumps(context['data'], indent=2, default=str)}

Detected Anomalies:
{json.dumps(context['anomalies'], indent=2)}

News Summary (if available):
{json.dumps(context['news_summary'], indent=2)}

Extract and structure all metrics according to the required JSON format.
"""

        # Run agent
        result = self.agent.invoke({"input": prompt})
        output = result['output']

        # Parse output
        if isinstance(output, dict):
            return output
        else:
            return self.parser.parse(output)

    def run(
        self,
        tickers: List[str],
        timeframe: Optional[str] = None,
        news_scraper_output: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Any]:
        """
        Extract metrics for one or more tickers.

        Args:
            tickers: List of ticker symbols
            timeframe: Optional timeframe string (e.g., "Q1 2024")
            news_scraper_output: Optional dict mapping tickers to news summaries

        Returns:
            Dictionary with ticker metrics (single ticker dict or nested dict for multiple)
        """
        news_scraper_output = news_scraper_output or {}

        if len(tickers) == 1:
            # Single ticker - return flat structure
            ticker = tickers[0]
            return self._process_single_ticker(
                ticker,
                timeframe,
                news_scraper_output.get(ticker)
            )
        else:
            # Multiple tickers - return nested structure
            results = {}
            for ticker in tickers:
                try:
                    results[ticker] = self._process_single_ticker(
                        ticker,
                        timeframe,
                        news_scraper_output.get(ticker)
                    )
                except Exception as e:
                    results[ticker] = {
                        "error": str(e),
                        "ticker": ticker,
                        "status": "failed"
                    }
            return results


if __name__ == "__main__":
    import os

    import requests

    # Check prerequisites
    fastapi_url = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")

    print("=" * 80)
    print("PREREQUISITE CHECKS")
    print("=" * 80)

    # Check FastAPI backend
    try:
        response = requests.get(f"{fastapi_url}/health", timeout=2)
        if response.status_code == 200:
            print(f"✓ FastAPI backend is running at {fastapi_url}")
        else:
            print(f"✗ FastAPI backend returned status {response.status_code}")
            print(f"\nPlease start the backend with:")
            print(f"  uvicorn backend.app.api_controller:app --reload")
            exit(1)
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to FastAPI backend at {fastapi_url}")
        print(f"\nPlease start the backend first with:")
        print(f"  uvicorn backend.app.api_controller:app --reload")
        exit(1)
    except Exception as e:
        print(f"✗ Error checking backend: {e}")
        exit(1)

    # Check OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print(f"OPENAI_API_KEY not found in environment variables")
        print(f"Please set OPENAI_API_KEY in your .env file or environment")
    else:
        print(f"✓ OpenAI API key is configured")

    # Initialize agent
    print("\n" + "=" * 80)
    print("INITIALIZING METRIC EXTRACTOR AGENT")
    print("=" * 80)
    agent = MetricExtractorAgent(model="gpt-4o-mini")
    print("✓ Agent initialized successfully\n")

    # Get available tickers from the database
    print("\nFetching available tickers from database...")
    try:
        response = requests.get(f"{fastapi_url}/tickers", timeout=5)
        available_tickers = response.json().get("tickers", [])
        print(f"Found {len(available_tickers)} tickers in database")

        # Filter for tickers that look like US stocks (simple heuristic)
        us_like_tickers = [t for t in available_tickers if len(t) <= 5 and t.replace(".", "").isalpha() and not ("." in t and t.split(".")[-1] in ["HK", "T", "SS"])]

        if us_like_tickers:
            test_ticker = us_like_tickers[0]
            print(f"Using ticker: {test_ticker}")
        elif available_tickers:
            test_ticker = available_tickers[0]
            print(f"Using ticker: {test_ticker} (non-US ticker)")
        else:
            print("No tickers found in database. Please run ETL first.")
            print("Run: python etl/etl_script.py")
            exit(1)
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        print("Using AAPL as fallback (may not have data)")
        test_ticker = "AAPL"

    # Test Case 1: Single ticker with timeframe
    print("\n" + "=" * 80)
    print(f"TEST 1: Single ticker ({test_ticker}) with timeframe")
    print("=" * 80)

    try:
        result1 = agent.run(
            tickers=[test_ticker],
            timeframe="Q1 2024"
        )
        print("\nRESULT:")
        print(json.dumps(result1, indent=2, default=str))
    except Exception as e:
        print(f"\n✗ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()

    # Test Case 2: Multiple tickers without timeframe (using first 2 available)
    if len(available_tickers) >= 2:
        test_tickers = available_tickers[:2]
        print("\n" + "=" * 80)
        print(f"TEST 2: Multiple tickers ({test_tickers}) - latest data")
        print("=" * 80)

        try:
            result2 = agent.run(tickers=test_tickers)
            print("\nRESULT:")
            print(json.dumps(result2, indent=2, default=str))
        except Exception as e:
            print(f"\n✗ Test 2 failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n" + "=" * 80)
        print("TEST 2: SKIPPED (need at least 2 tickers in database)")
        print("=" * 80)

    # Test Case 3: Single ticker with mock news scraper output
    print("\n" + "=" * 80)
    print(f"TEST 3: Single ticker ({test_ticker}) with news summary")
    print("=" * 80)

    mock_news = {
        test_ticker: {
            "qualitative_summary": "Company announced new product features and expansion plans",
            "quantitative_summary": "Revenue up 12% YoY, Operating margin at 22%",
            "insight_outlook": "Strong growth momentum with improving profitability"
        }
    }

    try:
        result3 = agent.run(
            tickers=[test_ticker],
            timeframe="Q2 2024",
            news_scraper_output=mock_news
        )
        print("\nRESULT:")
        print(json.dumps(result3, indent=2, default=str))
    except Exception as e:
        print(f"\n✗ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
