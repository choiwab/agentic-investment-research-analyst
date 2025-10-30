import os
import json
import re

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# Import utils function and models
from utils.callback_handler import PrintCallbackHandler
from utils.tools import fetch_peers, web_search
from utils.fetch_ticker_url import fetch_ticker_url
from utils.model_schema import PreprocessModel

load_dotenv()

class PreprocessAgent:
    """
    Simplified preprocessing agent optimized for Ollama/Llama models.
    Uses direct prompting instead of REACT agents for better compatibility.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.llm = ChatOllama(
            model=model,
            temperature=0
        )
        self.parser = StructuredOutputParser.from_response_schemas(PreprocessModel.response_schema)

    def _classify_intent(self, query: str) -> str:
        """Classify the intent of the query."""
        prompt = f"""You are a finance query classifier.

Classify this query into ONE of these intents:
- "finance-company" - Queries about specific companies/stocks (e.g., "Analyze Tesla", "Apple stock analysis")
- "finance-market" - Queries about general market/economy (e.g., "inflation trends", "S&P 500 outlook")
- "finance-education" - Queries asking for definitions/explanations (e.g., "What is P/E ratio?", "Explain EPS")
- "irrelevant" - Non-finance queries (e.g., "Tell me a joke", "What's the weather?")

Query: "{query}"

Respond with ONLY ONE of these exact strings: finance-company, finance-market, finance-education, or irrelevant"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        intent = response.content.strip().lower()

        # Validate intent
        valid_intents = ["finance-company", "finance-market", "finance-education", "irrelevant"]
        for valid in valid_intents:
            if valid in intent:
                return valid

        return "irrelevant"  # Default fallback

    def _extract_ticker(self, query: str) -> str:
        """Extract ticker symbol from query."""
        prompt = f"""Extract the stock ticker symbol from this query.

Common examples:
- "Tesla" or "Tesla stock" → TSLA
- "Apple" or "Apple Inc" → AAPL
- "Microsoft" or "MSFT" → MSFT
- "Amazon" → AMZN
- "Google" or "Alphabet" → GOOGL

Query: "{query}"

Respond with ONLY the ticker symbol (e.g., TSLA, AAPL) or "null" if no company is mentioned."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        ticker = response.content.strip().upper()

        if ticker == "NULL" or len(ticker) > 6:
            return None
        return ticker

    def _extract_timeframe(self, query: str) -> str:
        """Extract or default timeframe."""
        prompt = f"""Extract the time period from this query.

Examples:
- "Q1 2024" → Q1 2024
- "last quarter" → last quarter
- "past 6 months" → 6 months
- "2023" → 2023

Query: "{query}"

If no timeframe is mentioned, respond with "1 year" (default).
Respond with ONLY the timeframe string. DO NOT include any explanatory text."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        timeframe = response.content.strip()

        # Clean up common LLM responses
        if "default" in timeframe.lower() or "year" not in timeframe.lower() and len(timeframe) > 10:
            return "1 year"

        # Extract just the timeframe part if there's extra text
        if "\n" in timeframe:
            timeframe = timeframe.split("\n")[-1].strip()

        return timeframe if timeframe else "1 year"

    def _extract_metrics(self, query: str, intent: str) -> list:
        """Extract financial metrics from query."""
        if intent == "irrelevant":
            return None

        prompt = f"""Extract financial metric keywords from this query.

Valid examples: revenue, earnings, EPS, P/E ratio, profit margin, ROE, market cap, debt, cash flow, valuation

Query: "{query}"

Respond with a comma-separated list of metric keywords, or "general analysis" if the query asks for general/comprehensive analysis.
Examples:
- "revenue and profit margins" → revenue, profit margin
- "P/E ratio and earnings" → P/E ratio, earnings
- "comprehensive analysis" → general analysis"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        metrics_str = response.content.strip()

        if not metrics_str or "general analysis" in metrics_str.lower():
            return ["revenue", "earnings", "P/E ratio", "profit margin"]  # Default comprehensive metrics

        # Parse comma-separated list
        metrics = [m.strip() for m in metrics_str.split(',') if m.strip()]
        return metrics if metrics else None

    def _get_education_answer(self, query: str) -> str:
        """Get direct answer for finance-education queries."""
        prompt = f"""Provide a clear, concise answer to this financial education question.

Keep it to 2-3 sentences.

Question: "{query}"

Answer:"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def _call_web_search(self, search_query: str) -> str:
        """Call web search tool and return results."""
        try:
            result = web_search.invoke({"query": search_query})
            return str(result) if result else None
        except Exception as e:
            print(f"Web search error: {e}")
            return None

    def _call_fetch_ticker_url(self, ticker: str) -> str:
        """Call fetch ticker URL tool."""
        try:
            result = fetch_ticker_url.invoke({"ticker": ticker})
            return str(result) if result else None
        except Exception as e:
            print(f"Fetch ticker URL error: {e}")
            return None

    def run(self, state: dict[str, str]) -> dict[str, str]:
        """Run the preprocessing pipeline and return structured output."""
        query = state.get("query", "")

        print(f"\n🔍 Processing query: {query}\n")

        # Step 1: Classify intent
        intent = self._classify_intent(query)
        print(f"✅ Intent: {intent}")

        # Initialize result
        result = {
            "query": query,
            "intent": intent,
            "ticker": None,
            "peers": None,
            "timeframe": None,
            "metrics": None,
            "url": None,
            "output_from_websearch": None,
            "answer": None
        }

        # Step 2: Handle based on intent
        if intent == "irrelevant":
            # Nothing else to do
            pass

        elif intent == "finance-company":
            # Extract ticker
            ticker = self._extract_ticker(query)
            result["ticker"] = ticker
            print(f"✅ Ticker: {ticker}")

            # Get finnhub URL if ticker found
            if ticker:
                url = self._call_fetch_ticker_url(ticker)
                result["url"] = url
                print(f"✅ URL: {url}")

            # Extract timeframe and metrics
            result["timeframe"] = self._extract_timeframe(query)
            result["metrics"] = self._extract_metrics(query, intent)
            print(f"✅ Timeframe: {result['timeframe']}")
            print(f"✅ Metrics: {result['metrics']}")

        elif intent == "finance-market":
            # Call web search for market data
            search_result = self._call_web_search(query)
            result["output_from_websearch"] = search_result
            result["timeframe"] = self._extract_timeframe(query)
            result["metrics"] = self._extract_metrics(query, intent)
            print(f"✅ Web search completed")
            print(f"✅ Timeframe: {result['timeframe']}")
            print(f"✅ Metrics: {result['metrics']}")

        elif intent == "finance-education":
            # Get direct answer
            answer = self._get_education_answer(query)
            result["answer"] = answer
            print(f"✅ Answer: {answer[:100]}...")

        # Validate and return
        model = PreprocessModel(**result)
        return model.model_dump()


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Simplified Preprocessing Agent with Ollama")
    print("=" * 80)

    agent = PreprocessAgent(model="llama3.1")

    # Test 1: Finance-company
    print("\n### Test 1: Finance-company ###")
    result1 = agent.run({"query": "Give me a comprehensive analysis on Tesla stock"})
    print(f"\nResult: {json.dumps(result1, indent=2)}")

    # Test 2: Irrelevant
    print("\n\n### Test 2: Irrelevant ###")
    result2 = agent.run({"query": "Tell me a joke about dogs"})
    print(f"\nResult: {json.dumps(result2, indent=2)}")

    # Test 3: Finance-education
    print("\n\n### Test 3: Finance-education ###")
    result3 = agent.run({"query": "What is P/E ratio?"})
    print(f"\nResult: {json.dumps(result3, indent=2)}")
