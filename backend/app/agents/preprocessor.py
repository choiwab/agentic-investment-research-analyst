import json
import os
import re

from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

# Import utils function and models
from utils.callback_handler import PrintCallbackHandler
from utils.fetch_ticker_url import fetch_ticker_url
from utils.model_schema import PreprocessModel
from utils.tools import fetch_peers, web_search

load_dotenv(override=True)

class PreprocessAgent:
    """
    Preprocessing agent using OpenAI models.
    Uses direct prompting for intent classification and entity extraction.
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model
        self.llm = ChatOpenAI(
            model=model,
            temperature=0
        )
        self.parser = StructuredOutputParser.from_response_schemas(PreprocessModel.response_schema)

    def _classify_intent(self, query: str) -> str:
        """Classify the intent of the query using OpenAI with structured output."""
        from typing import Literal

        from pydantic import BaseModel, Field

        class IntentClassification(BaseModel):
            """Intent classification result"""
            intent: Literal["finance-company", "finance-market", "finance-education", "irrelevant"] = Field(
                description="The classified intent category"
            )
            reasoning: str = Field(
                description="Brief explanation of why this intent was chosen"
            )

        print(f"[DEBUG] Query: {query}")

        # Create a structured output LLM
        structured_llm = self.llm.with_structured_output(IntentClassification)

        prompt = f"""You are an expert financial query classifier. Analyze the user's query and classify it into ONE of these four categories:

**1. finance-company**
   - Queries about SPECIFIC companies or stocks
   - Mentions company names, stock tickers, or requests company analysis
   - Examples:
     * "Analyze Tesla stock"
     * "What's Apple's performance?"
     * "MSFT earnings report"
     * "Tell me about Amazon's financials"

**2. finance-market**
   - Queries about GENERAL market trends, economy, or broad financial indicators
   - Asks about macroeconomic conditions, market indices, economic policies
   - Examples:
     * "What are inflation trends?"
     * "Current state of the economy"
     * "S&P 500 outlook"
     * "How are interest rates affecting markets?"
     * "What's happening with the Fed?"

**3. finance-education**
   - Queries asking for definitions, explanations, or learning about financial concepts
   - Contains words like "what is", "explain", "define", "how does", "meaning of"
   - Examples:
     * "What is P/E ratio?"
     * "Explain dividend yield"
     * "How does compound interest work?"
     * "Define market capitalization"

**4. irrelevant**
   - Non-finance related queries
   - Examples:
     * "Tell me a joke"
     * "What's the weather?"
     * "Recipe for pasta"

User Query: "{query}"

Classify this query and provide your reasoning."""

        try:
            result = structured_llm.invoke(prompt)
            intent = result.intent
            reasoning = result.reasoning

            print(f"[DEBUG] OpenAI Classification: {intent}")
            print(f"[DEBUG] Reasoning: {reasoning}")

            # Validate the intent is one of the expected values
            valid_intents = ["finance-company", "finance-market", "finance-education", "irrelevant"]
            if intent in valid_intents:
                return intent
            else:
                print(f"[WARNING] Unexpected intent '{intent}', defaulting to irrelevant")
                return "irrelevant"

        except Exception as e:
            print(f"[ERROR] Intent classification failed: {e}")
            print(f"[DEBUG] Falling back to irrelevant")
            return "irrelevant"

    def _extract_ticker_with_yahoo_finance(self, query: str, company_name: str = None) -> str:
        """Extract ticker symbol using Yahoo Finance search via web search and yfinance validation."""
        from typing import Optional

        import yfinance as yf
        from pydantic import BaseModel, Field

        class TickerExtraction(BaseModel):
            """Ticker extraction result"""
            ticker: Optional[str] = Field(
                description="The stock ticker symbol in uppercase (e.g., TSLA, AAPL), or null if no ticker found"
            )
            confidence: str = Field(
                description="Confidence level: 'high', 'medium', or 'low'"
            )

        # Build search query specifically for Yahoo Finance
        search_term = company_name if company_name else query
        search_query = f"site:finance.yahoo.com {search_term} stock ticker"
        
        print(f"[DEBUG] Searching Yahoo Finance for ticker: {search_query}")
        
        try:
            # Call web search to find Yahoo Finance pages
            search_result = web_search.invoke({"query": search_query})
            
            if not search_result or not isinstance(search_result, dict):
                print(f"[DEBUG] Yahoo Finance search failed or returned invalid result")
                return None
            
            if "error" in search_result:
                print(f"[DEBUG] Yahoo Finance search returned error: {search_result.get('error')}")
                return None
            
            # Extract text from search results, prioritizing Yahoo Finance URLs
            search_text = ""
            yahoo_urls = []
            
            if "results" in search_result:
                for result in search_result.get("results", []):
                    if isinstance(result, dict):
                        url = result.get("url", "")
                        text = result.get("text", "")
                        
                        # Prioritize Yahoo Finance URLs
                        if "finance.yahoo.com" in url or "yahoo.com/quote" in url:
                            yahoo_urls.append(url)
                            search_text = text + "\n\n" + search_text  # Prepend Yahoo Finance results
                        else:
                            search_text += text + "\n\n"
            
            if not search_text:
                print(f"[DEBUG] No text content found in Yahoo Finance search results")
                return None
            
            # Use LLM to extract ticker from search results
            structured_llm = self.llm.with_structured_output(TickerExtraction)
            print(search_text)
            # Limit search text to first 3000 chars
            limited_text = search_text[:3000]
            
            prompt = f"""Extract the stock ticker symbol from the following Yahoo Finance search results.

Search Query: "{search_term}"

Yahoo Finance Search Results:
{limited_text}

Extract the ticker symbol (e.g., TSLA, AAPL, MSFT). Look for patterns like:
- Yahoo Finance URLs: finance.yahoo.com/quote/XXX
- "Symbol: XXX" or "Ticker: XXX"
- "NYSE: XXX" or "NASDAQ: XXX"
- Stock symbol in parentheses: Company Name (XXX)
- Ticker codes in the text

Return the ticker in uppercase. If no clear ticker is found, return null."""

            result = structured_llm.invoke(prompt)
            ticker = result.ticker

            print(f"[DEBUG] Yahoo Finance ticker extraction: {ticker} (confidence: {result.confidence})")
            
            # Validate ticker format and existence using yfinance
            if ticker:
                ticker = ticker.upper().strip()
                # Valid tickers are 1-6 characters (some have dots like BRK.B)
                if len(ticker.replace(".", "")) > 0 and len(ticker) <= 6:
                    # Validate ticker exists using yfinance
                    try:
                        yf_ticker = yf.Ticker(ticker)
                        info = yf_ticker.info
                        # Check if we got valid info (has symbol or company name)
                        if info and (info.get("symbol") or info.get("longName") or info.get("shortName")):
                            print(f"[DEBUG] Ticker {ticker} validated with yfinance")
                            return ticker
                        else:
                            print(f"[DEBUG] Ticker {ticker} not found in yfinance, but returning anyway")
                            # Return the ticker even if yfinance doesn't have full info
                            # (might be a valid ticker that yfinance doesn't fully support)
                            return ticker
                    except Exception as yf_error:
                        print(f"[DEBUG] yfinance validation failed for {ticker}: {yf_error}")
                        # Still return the ticker if yfinance validation fails (might be valid but yfinance issue)
                        return ticker

            return None

        except Exception as e:
            print(f"[ERROR] Yahoo Finance ticker extraction failed: {e}")
            return None

    def _extract_ticker(self, query: str) -> str:
        """Extract ticker symbol from query using OpenAI with structured output, with Yahoo Finance fallback."""
        from typing import Optional

        from pydantic import BaseModel, Field

        # First, check if query already contains a ticker-like pattern (1-6 uppercase letters, possibly with dot)
        # Pattern matches: 1-5 letters, optionally followed by . and 1 letter (e.g., BRK.B)
        ticker_pattern = r'\b([A-Z]{1,5}(?:\.[A-Z])?)\b'
        potential_tickers = re.findall(ticker_pattern, query.upper())
        
        # Filter potential tickers (common words to exclude, and must be at least 2 characters)
        common_words = {'I', 'A', 'AN', 'THE', 'IS', 'IT', 'IN', 'ON', 'AT', 'TO', 'FOR', 'OF', 'AND', 'OR', 'BUT', 'AS', 'AM', 'BE', 'DO', 'GO', 'IF', 'MY', 'NO', 'SO', 'UP', 'WE'}
        potential_tickers = [t for t in potential_tickers if t not in common_words and len(t.replace('.', '')) >= 2]
        
        if potential_tickers:
            # Try to validate the first potential ticker using yfinance
            for potential_ticker in potential_tickers[:3]:  # Check up to 3 potential tickers
                try:
                    import yfinance as yf
                    yf_ticker = yf.Ticker(potential_ticker)
                    info = yf_ticker.info
                    if info and (info.get("symbol") or info.get("longName") or info.get("shortName")):
                        print(f"[DEBUG] Found ticker in query: {potential_ticker} (validated with yfinance)")
                        return potential_ticker.upper()
                except Exception:
                    continue

        class TickerExtraction(BaseModel):
            """Ticker extraction result"""
            ticker: Optional[str] = Field(
                description="The stock ticker symbol in uppercase (e.g., TSLA, AAPL), or null if no company mentioned"
            )
            company_name: Optional[str] = Field(
                description="The full company name if identified"
            )

        # Create a structured output LLM
        structured_llm = self.llm.with_structured_output(TickerExtraction)

        prompt = f"""Extract the stock ticker symbol from this query.

IMPORTANT: The query may already contain a ticker symbol (1-5 uppercase letters like TSLA, AAPL, IOSP, etc.). 
If you see a ticker-like pattern in the query, extract it directly.

Common company to ticker mappings:
- Tesla / Tesla Motors → TSLA
- Apple / Apple Inc → AAPL
- Microsoft → MSFT
- Amazon → AMZN
- Google / Alphabet → GOOGL
- Meta / Facebook → META
- Netflix → NFLX
- Nvidia → NVDA
- Berkshire Hathaway → BRK.B
- JPMorgan / JPMorgan Chase → JPM

Query: "{query}"

If a ticker symbol appears in the query (like "IOSP", "TSLA", etc.), extract it directly.
If a company name is mentioned, return its ticker symbol in uppercase.
If no company or ticker is mentioned, return null for ticker."""

        try:
            result = structured_llm.invoke(prompt)
            ticker = result.ticker
            company_name = result.company_name

            print(f"[DEBUG] Ticker extraction: {ticker}")
            if company_name:
                print(f"[DEBUG] Company identified: {company_name}")

            # Validate ticker format
            if ticker:
                ticker = ticker.upper().strip()
                # Valid tickers are 1-6 characters (some have dots like BRK.B)
                if len(ticker.replace(".", "")) > 0 and len(ticker) <= 6:
                    return ticker
            
            # If OpenAI didn't find a ticker but identified a company, try Yahoo Finance search
            if not ticker and company_name:
                print(f"[DEBUG] OpenAI didn't find ticker, trying Yahoo Finance search for: {company_name}")
                yahoo_ticker = self._extract_ticker_with_yahoo_finance(query, company_name)
                if yahoo_ticker:
                    return yahoo_ticker

            return None

        except Exception as e:
            print(f"[ERROR] Ticker extraction failed: {e}")
            # Try Yahoo Finance as fallback
            print(f"[DEBUG] Trying Yahoo Finance search as fallback")
            yahoo_ticker = self._extract_ticker_with_yahoo_finance(query)
            return yahoo_ticker if yahoo_ticker else None

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
    print("Testing Preprocessing Agent with OpenAI")
    print("=" * 80)

    agent = PreprocessAgent(model="gpt-4o")

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

    # Test 4: Finance-market (inflation trends)
    print("\n\n### Test 4: Finance-market ###")
    result4 = agent.run({"query": "What are inflation trends"})
    print(f"\nResult: {json.dumps(result4, indent=2)}")
    assert result4["intent"] == "finance-market", f"Expected finance-market, got {result4['intent']}"
    print("✅ Test passed!")

    # Test 5: Finance-market (S&P 500)
    print("\n\n### Test 5: Finance-market (S&P 500) ###")
    result5 = agent.run({"query": "What's the S&P 500 outlook?"})
    print(f"\nResult: {json.dumps(result5, indent=2)}")
    assert result5["intent"] == "finance-market", f"Expected finance-market, got {result5['intent']}"
    print("✅ Test passed!")
