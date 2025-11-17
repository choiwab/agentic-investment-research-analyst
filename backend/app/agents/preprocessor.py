import json
import os
import re
from difflib import SequenceMatcher
from typing import List, Optional

from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser
from langchain.schema import HumanMessage
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from pymongo import MongoClient

# Import utils function and models
from utils.callback_handler import PrintCallbackHandler
from utils.fetch_ticker_url import fetch_ticker_url
from utils.model_schema import PreprocessModel
from utils.tools import web_search

load_dotenv(override=True)

ATLAS_URI = os.getenv("ATLAS_URI")
MONGODB_DB = "financial_data"
_MONGO_CLIENT: Optional[MongoClient] = None
_COMPANIES_COLLECTION = None


def _get_companies_collection():
    """Lazily initialize and return the companies collection."""
    global _MONGO_CLIENT, _COMPANIES_COLLECTION

    if _COMPANIES_COLLECTION is not None:
        return _COMPANIES_COLLECTION

    if not ATLAS_URI:
        print("[DEBUG] ATLAS_URI not set; cannot connect to MongoDB for ticker lookup")
        return None

    try:
        _MONGO_CLIENT = MongoClient(ATLAS_URI, serverSelectionTimeoutMS=5000)
        db_name = MONGODB_DB or "test"
        _COMPANIES_COLLECTION = _MONGO_CLIENT[db_name]["companies"]
    except Exception as e:
        print(f"[ERROR] MongoDB connection for ticker lookup failed: {e}")
        _COMPANIES_COLLECTION = None

    return _COMPANIES_COLLECTION

# Regex to capture words/ticker-like tokens, including formats such as BRK.B or Inc.
TICKER_TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:\.[A-Za-z]+)?\.?")

# Very common short words that should not be considered as tickers
TICKER_STOPWORDS = {
    "I", "A", "AN", "THE", "IS", "IT", "IN", "ON", "AT", "TO", "FOR", "OF", "AND",
    "OR", "BUT", "AS", "AM", "BE", "DO", "GO", "IF", "MY", "NO", "SO", "UP", "WE",
    "BY", "WITH", "FROM", "THIS", "THAT", "WHAT", "WHEN", "YOUR", "YOU", "OUR",
    "THEY", "THEM", "THEIR", "ARE", "NOT", "JUST", "ONLY", "ABOUT", "STOCK", "SHARE",
    "PRICE", "TREND"
}

# Company suffixes that often appear after a company name (ignore unless fully uppercase)
CORPORATE_SUFFIXES = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "LLC", "LLP", "PLC",
    "LTD", "LIMITED", "COMPANY", "CO", "HOLDINGS", "HOLDING", "GROUP", "LP"
}

CLASS_DESIGNATORS = {"CLASS", "CL", "SHARE", "SHARES", "SERIES", "ADR", "ADS", "UNIT", "UNITS"}

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
        self._company_lookup_cache: dict[str, Optional[str]] = {}
        self._ticker_presence_cache: dict[str, Optional[bool]] = {}

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

    def _detect_explicit_ticker(self, query: str) -> Optional[str]:
        """Return a directly mentioned ticker if the query includes one."""
        seen_tokens = set()

        for match in TICKER_TOKEN_PATTERN.finditer(query):
            token = match.group(0)
            normalized = token.upper().rstrip(".")
            plain = normalized.replace(".", "").replace("-", "")

            if len(plain) < 1 or len(plain) > 6:
                continue
            if normalized in TICKER_STOPWORDS or plain in TICKER_STOPWORDS:
                continue
            if normalized in CORPORATE_SUFFIXES and not token.isupper():
                continue
            start_index = match.start()
            preceding_char = query[start_index - 1] if start_index > 0 else ""
            if not token.isupper() and preceding_char != "$":
                continue
            if normalized in seen_tokens:
                continue

            seen_tokens.add(normalized)
            ticker_candidate = normalized

            # Check if ticker exists in MongoDB
            db_has_ticker = self._ticker_exists_in_db(ticker_candidate)

            # Allow ticker even if not in MongoDB - let metric extractor handle Alpha Vantage fallback
            if db_has_ticker is False:
                print(f"[INFO] Ticker {ticker_candidate} not in MongoDB, will use Alpha Vantage fallback")

            print(f"[DEBUG] Found explicit ticker in query: {ticker_candidate}")
            return ticker_candidate

        return None

    def _ticker_exists_in_db(self, ticker: str) -> Optional[bool]:
        """Check if a ticker exists in the companies collection (cached)."""
        if ticker in self._ticker_presence_cache:
            return self._ticker_presence_cache[ticker]

        collection = _get_companies_collection()
        if collection is None:
            self._ticker_presence_cache[ticker] = None
            return None

        try:
            exists = collection.find_one({"_id": ticker}, {"_id": 1}) is not None
            self._ticker_presence_cache[ticker] = exists
            return exists
        except Exception as e:
            print(f"[ERROR] Failed to verify ticker '{ticker}' in MongoDB: {e}")
            self._ticker_presence_cache[ticker] = None
            return None

    def _extract_company_names(self, query: str) -> List[str]:
        """Use the LLM to extract company names mentioned in the query."""
        from pydantic import BaseModel, Field

        class CompanyExtraction(BaseModel):
            company_names: List[str] = Field(
                default_factory=list,
                description="Ordered list of up to three public company names mentioned in the query",
            )

        try:
            structured_llm = self.llm.with_structured_output(CompanyExtraction)
            prompt = f"""Identify up to three publicly traded company names mentioned in this query.

Focus only on actual companies (no ETFs, indices, sectors, or cryptocurrencies).
Return the formal parent company names (e.g., "Tesla Inc", "Microsoft Corporation").
If no company is mentioned, return an empty list.

Query: "{query}"
"""
            result = structured_llm.invoke(prompt)
            company_names = [
                name.strip() for name in (result.company_names or []) if name and name.strip()
            ]
            return company_names[:3]
        except Exception as e:
            print(f"[ERROR] Company name extraction failed: {e}")
            return []

    def _generate_company_name_variants(self, company_name: str) -> List[str]:
        """Generate cleaned variants of a company name for DB lookups."""
        variants: List[str] = []
        if not company_name:
            return variants

        cleaned = company_name.strip()
        if cleaned:
            variants.append(cleaned)

        simplified = self._simplify_company_name(cleaned)
        if simplified and simplified not in variants:
            variants.append(simplified)

        and_variant = cleaned.replace("&", "and")
        if and_variant and and_variant not in variants:
            variants.append(and_variant)

        return [variant for variant in variants if variant]

    def _simplify_company_name(self, name: str) -> str:
        """Remove suffixes, share class details, and parentheticals from a company name."""
        if not name:
            return ""

        no_parenthetical = re.sub(r"\(.*?\)", "", name)
        tokens = re.split(r"\s+", no_parenthetical.strip())

        simplified_tokens: List[str] = []
        for token in tokens:
            stripped = token.strip(",.")
            if not stripped:
                continue
            normalized = re.sub(r"[^A-Za-z0-9&]", "", stripped).upper()
            if normalized in CLASS_DESIGNATORS:
                break
            if normalized in CORPORATE_SUFFIXES:
                continue
            simplified_tokens.append(stripped)

        return " ".join(simplified_tokens).strip()

    def _lookup_ticker_by_company_name(self, company_name: str) -> Optional[str]:
        """Map a company name to its ticker using the companies collection."""
        if not company_name:
            return None

        cache_key = company_name.lower().strip()
        if cache_key in self._company_lookup_cache:
            return self._company_lookup_cache[cache_key]

        collection = _get_companies_collection()
        if collection is None:
            print("[DEBUG] MongoDB connection unavailable for company lookup")
            self._company_lookup_cache[cache_key] = None
            return None

        try:
            variants = self._generate_company_name_variants(company_name)
            for variant in variants:
                ticker = self._match_company_name_in_collection(collection, variant)
                if ticker:
                    print(f"[DEBUG] Matched company '{company_name}' to ticker '{ticker}' via variant '{variant}'")
                    self._company_lookup_cache[cache_key] = ticker
                    return ticker

            ticker = self._fuzzy_company_lookup(collection, company_name)
            if ticker:
                print(f"[DEBUG] Fuzzy matched company '{company_name}' to ticker '{ticker}'")
            else:
                print(f"[DEBUG] No ticker match found for company '{company_name}'")

            self._company_lookup_cache[cache_key] = ticker
            return ticker
        except Exception as e:
            print(f"[ERROR] Company lookup failed for '{company_name}': {e}")
            self._company_lookup_cache[cache_key] = None
            return None

    def _match_company_name_in_collection(self, collection, candidate: str) -> Optional[str]:
        """Perform direct regex matches for a candidate company name."""
        if not candidate:
            return None

        escaped = re.escape(candidate)
        regexes = [
            re.compile(rf"^{escaped}$", re.IGNORECASE),
            re.compile(rf"^{escaped}\b", re.IGNORECASE),
            re.compile(escaped, re.IGNORECASE),
        ]

        for regex in regexes:
            doc = collection.find_one({"name": regex}, {"_id": 1})
            if doc and doc.get("_id"):
                return doc["_id"]

        return None

    def _fuzzy_company_lookup(self, collection, company_name: str) -> Optional[str]:
        """Fallback fuzzy match when direct variants fail."""
        normalized_target = self._normalize_similarity_key(company_name)
        if not normalized_target:
            return None

        keywords = [token for token in re.split(r"[^A-Za-z0-9&]+", company_name) if token]
        regex = None
        for keyword in keywords:
            keyword_upper = keyword.upper()
            if keyword_upper in CORPORATE_SUFFIXES or keyword_upper in CLASS_DESIGNATORS:
                continue
            if len(keyword) >= 2:
                regex = re.compile(re.escape(keyword), re.IGNORECASE)
                break

        query = {"name": regex} if regex else {}
        candidates = list(collection.find(query, {"_id": 1, "name": 1}).limit(50))
        return self._pick_best_company_match(normalized_target, candidates)

    def _pick_best_company_match(self, normalized_target: str, candidates: List[dict]) -> Optional[str]:
        """Select the best ticker candidate using sequence similarity."""
        best_score = 0.0
        best_ticker: Optional[str] = None

        for doc in candidates:
            candidate_name = doc.get("name") or ""
            candidate_key = self._normalize_similarity_key(candidate_name)
            if not candidate_key:
                continue
            score = SequenceMatcher(None, normalized_target, candidate_key).ratio()
            if score > best_score:
                best_score = score
                best_ticker = doc.get("_id")

        if best_score >= 0.7:
            return best_ticker
        return None

    def _normalize_similarity_key(self, text: str) -> str:
        """Normalize text for similarity comparisons by stripping suffixes and punctuation."""
        if not text:
            return ""

        text = re.sub(r"\(.*?\)", "", text.lower())
        tokens = [
            token
            for token in re.split(r"[^a-z0-9&]+", text)
            if token and token.upper() not in CORPORATE_SUFFIXES and token.upper() not in CLASS_DESIGNATORS
        ]
        return " ".join(tokens).strip()

    def _extract_ticker(self, query: str) -> Optional[str]:
        """Extract ticker by checking explicit mentions, then company-to-ticker mapping via MongoDB."""
        explicit_ticker = self._detect_explicit_ticker(query)
        if explicit_ticker:
            return explicit_ticker

        company_names = self._extract_company_names(query)
        if not company_names:
            print("[DEBUG] No company names detected in query for ticker lookup")
            return None

        for company_name in company_names:
            ticker = self._lookup_ticker_by_company_name(company_name)
            if ticker:
                return ticker

        return None

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

                if url is None:
                    print("No URL found - invoking LLM chain with web_search tool")
                    search_tool = StructuredTool.from_function(
                        func = self._call_web_search,
                        name = "web_search",
                        description = "Searches the web for recent and relevant news about a company. Input should be a plain text query, not code."
                    )
                    prompt = PromptTemplate.from_template("""
                    You are a senior financial assistant with access to the 'web_search' tool.
                    **Objective:**
                    Find the **most recent, detailed, and parsable financial news article or report** about the company that can be safely fetched and analyzed.  
                    Your goal is to provide a URL that contains substantive **text content** about the company’s **financial performance, earnings, or strategic updates.**
                    Prioritize for a website that contains the most details (metrics) and insights
                    
                    ---
                                                          
                    ### Steps:
                    1. Use the 'web_search' tool to find recent, relevant information about the company.
                        - Always call the tool using this exact structure:
                            Action: web_search
                            Action Input: <company name or ticker> recent financial news OR earnings report OR investor update OR press release
                        - Never use parentheses. 
                                             
                    2. **Select a page that meets ALL of the following criteria:**
                        - Published within the **past 6 months**
                        - Contains **textual article content** (news, press release, or report)
                        - Includes **metrics, numbers, or financial insights**
                        - Readable and fetchable using normal HTTP requests
                        - Not blocked by Cloudflare, CAPTCHA, login walls, or Akamai
                        - Comes from an **official or reliable source**, such as:
                            - Company investor relations (IR) page  
                            - SEC filings (edgar.gov)  
                            - Major financial outlets (Reuters, Bloomberg, CNBC, Yahoo Finance *news article pages*, MarketWatch *news section*)  
                    3. **Explicitly avoid:**
                        - Static stock profile or summary pages (like `/quote/TSLA/`)
                        - General “news listing” or “overview” pages
                        - PDF documents, images, or non-text content
                        - Pages that cannot be parsed into readable text
                    4. If all high-quality article URLs fail these conditions (e.g., unparseable, blocked, or PDF-only),  
                        → fallback to the **official investor relations page** (IR) for the company instead.
                    
                    ---

                    Return **strictly valid JSON** in this format:
                    {{
                        "url": "<url>"
                    }}

                    Ticker: {ticker}
                    """)
                    agent = initialize_agent(
                        tools = [search_tool],
                        llm = self.llm,
                        agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        verbose = True,
                        handle_parsing_errors = True
                    )
                    query_text = prompt.format(ticker = ticker)
                    try:
                        output = agent.invoke({"input" : query_text})
                        print(f"[DEBUG] Raw LLM Output: \n{output}\n")
                    except Exception as e:
                        print(f"Agent web search failed : {e}")
                        result['url'] = None
                        return result

                    raw_output = output.get("output", "") if isinstance(output, dict) else str(output)
                    clean_output = re.sub(r"^```(?:json)?|```$", "", raw_output.strip(), flags=re.MULTILINE).strip()
                    
                    try:
                        parsed = json.loads(clean_output)
                        url = parsed.get("url")
                    except:
                        print("Failed to parse JSON output")
                        urls = re.findall(r'https?://[^\s"\]]+', clean_output)
                        url = urls if urls else None

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

    # # Test 2: Irrelevant
    # print("\n\n### Test 2: Irrelevant ###")
    # result2 = agent.run({"query": "Tell me a joke about dogs"})
    # print(f"\nResult: {json.dumps(result2, indent=2)}")

    # # Test 3: Finance-education
    # print("\n\n### Test 3: Finance-education ###")
    # result3 = agent.run({"query": "What is P/E ratio?"})
    # print(f"\nResult: {json.dumps(result3, indent=2)}")

    # # Test 4: Finance-market (inflation trends)
    # print("\n\n### Test 4: Finance-market ###")
    # result4 = agent.run({"query": "What are inflation trends"})
    # print(f"\nResult: {json.dumps(result4, indent=2)}")
    # assert result4["intent"] == "finance-market", f"Expected finance-market, got {result4['intent']}"
    # print("✅ Test passed!")

    # # Test 5: Finance-market (S&P 500)
    # print("\n\n### Test 5: Finance-market (S&P 500) ###")
    # result5 = agent.run({"query": "What's the S&P 500 outlook?"})
    # print(f"\nResult: {json.dumps(result5, indent=2)}")
    # assert result5["intent"] == "finance-market", f"Expected finance-market, got {result5['intent']}"
    # print("✅ Test passed!")
