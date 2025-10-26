import os
import json
import re

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain import hub

# Agent Setup and Structuring Output
from langchain_openai import ChatOpenAI
from langchain.output_parsers import OutputFixingParser, StructuredOutputParser
from langchain.prompts import PromptTemplate

# Import utils function and models
from utils.callback_handler import PrintCallbackHandler
from utils.conversation_buffer_safe import SafeConversationMemory
from utils.tools import fetch_peers, web_search
from utils.fetch_ticker_url import fetch_ticker_url
from utils.model_schema import PreprocessModel

load_dotenv()

class PreprocessAgent:
    def __init__(self, model: str) -> None:
        self.callback_handler = PrintCallbackHandler()
        self.llm = ChatOpenAI(
            model = model, 
            temperature = 0, 
            streaming = True, 
            callbacks = [self.callback_handler],
            api_key = os.getenv("OPENAI_API_KEY")
        )
        self.parser = StructuredOutputParser.from_response_schemas(PreprocessModel.response_schema)
        self.format_instructions = self.parser.get_format_instructions()
        self.parsing_llm = ChatOpenAI(
            model = model,
            temperature = 0,
            streaming = False,
            api_key = os.getenv("OPENAI_API_KEY")
        )
        self.fixing_parser = OutputFixingParser.from_llm(parser = self.parser, llm = self.parsing_llm)
        self.memory = SafeConversationMemory(
            memory_key = "chat_history",
            return_messages = True 
        )
        self.agent = self.build_agent()
    
    def build_agent(self) -> AgentExecutor:
        """Builds a REACT Agent"""
        raw_system_template : str = """
        You are a preprocessing agent. 
        Your job: CLASSIFY intent and EXTRACT information using tools. DO NOT answer questions.
        
        CRITICAL RULES:
        1. ALWAYS use tools BEFORE returning Final Answer
        2. For finance-market queries: MUST call web_search tool
        3. For finance-company queries: MUST call fetch_ticker_data tool
        4. ONLY return JSON in the "Final Answer" field (after using tools)
        5. Follow the REACT pattern: Thought → Action → Observation → Final Answer

        === INTENT CLASSIFICATION ===

        Choose EXACTLY one intent (copy the exact string):

        1. "finance-company" - Queries about specific companies/stocks
        2. "finance-market" - Queries about general market/economy  
        3. "finance-education" - Queries asking for definitions/explanations
        4. "irrelevant" - Non-finance queries

        === FIELD RULES ===

        **query**: Exact original user input

        **intent**: One of the 4 strings above (NEVER make up new intent names)

        **ticker**: Company ticker symbol (e.g., "TSLA")

        **peers**: Array of peer tickers or null. Get via: web_search "X competitors" → fetch_peers

        **timeframe**: Extract from query, else default "1 year". NEVER "N/A"

        **metrics**: Array of financial metric KEYWORDS:
        - If specific metrics mentioned → extract those (e.g., ["P/E ratio", "earnings"])
        - If vague (e.g., "analysis") → web_search "key financial metrics for [topic]" → extract keywords
        - Valid examples: "revenue", "earnings", "P/E ratio", "profit margin", "ROE", "market cap"
        - NEVER generic terms like "analysis" or "N/A"

        **url**: ONLY for finance-company. MUST be a finnhub.io URL like "https://finnhub.io/api/news?id=xxxx". Use fetch_ticker_url(ticker) tool → it returns the finnhub.io URL directly

        **output_from_websearch**: 
        - finance-market: REQUIRED, from web_search
        - finance-education: Optional, from web_search if needed
        - finance-company: Usually null (unless used for peers/metrics discovery)

        **answer**: 
        - finance-education: Use ONLY if answering directly without tools
        - All other intents: MUST be null

        === WORKFLOWS ===

        **finance-company:**
        1. Extract ticker symbol
        2. ALWAYS call fetch_ticker_url(ticker) to get the finnhub.io URL
        3. Extract the URL from the observation and put it in the url field
        4. If metrics unclear → web_search "key financial metrics for [ticker] stock analysis"
        5. If peers needed → web_search "X competitors" → fetch_peers
        6. Extract/default timeframe to "1 year"
        7. answer = null

        **finance-market:**
        1. ALWAYS call web_search with query details to get current market data
        2. Store the search results in output_from_websearch field
        3. If metrics unclear → web_search "key indicators for [topic]"
        4. Extract/default timeframe to "1 year"
        5. Populate metrics with extracted keywords
        6. answer = null

        **finance-education:**
        1. Can answer directly? → put in "answer", output_from_websearch = null
        2. Need to search? → web_search → put in output_from_websearch, answer = null

        **irrelevant:**
        1. ONLY fill "query" and "intent" fields with "irrelevant"
        2. Set ALL other fields to null
        3. DO NOT use any tools for irrelevant queries
        4. Return immediately with Final Answer JSON
        5. DO NOT iterate or try different approaches


        === EXAMPLES ===

        User: "Give me an analysis on Tesla Stock"

        Action: web_search
        Action Input: key financial metrics for stock analysis
        Observation: "...P/E ratio, EPS, revenue, profit margin..."

        Action: fetch_ticker_url
        Action Input: TSLA
        Observation: "https://finnhub.io/api/news?id=abc123def"

        Final Answer:
        {
            "query": "Give me an analysis on Tesla Stock",
            "intent": "finance-company",
            "ticker": "TSLA",
            "peers": null,
            "timeframe": "1 year",
            "metrics": ["P/E ratio", "EPS", "revenue", "profit margin"],
            "url": "https://finnhub.io/api/news?id=abc123def",
            "output_from_websearch": null,
            "answer": null
        }

        ---

        User: "What are current inflation trends?"

        Action: web_search
        Action Input: current inflation trends October 2025
        Observation: "Inflation at 3.2%..."

        Final Answer:
        {
            "query": "What are current inflation trends?",
            "intent": "finance-market",
            "ticker": null,
            "peers": null,
            "timeframe": "1 year",
            "metrics": ["inflation"],
            "url": null,
            "output_from_websearch": "Inflation at 3.2%...",
            "answer": null
        }

        ---

        User: "What is P/E ratio?"

        Final Answer:
        {
            "query": "What is P/E ratio?",
            "intent": "finance-education",
            "ticker": null,
            "peers": null,
            "timeframe": null,
            "metrics": null,
            "url": null,
            "output_from_websearch": null,
            "answer": "P/E ratio (Price-to-Earnings) divides stock price by earnings per share to measure valuation."
        }

        ---

         User: "Tell me a joke about dogs. "

        Final Answer:
        {
            "query": "Tell me a joke about dogs.",
            "intent": "irrelevant",
            "ticker": null,
            "peers": null,
            "timeframe": null,
            "metrics": null,
            "url": null,
            "output_from_websearch": null,
            "answer": null
        }

        ---

        === KEY REMINDERS ===
        - Intent: Use exact strings, never invent new ones
        - Metrics: Array of keywords, use web_search if unclear
        - Timeframe: Default "1 year", never "N/A"
        - URL: Just the string, nothing else
        - Answer: Only for finance-education direct answers
        - You're a PREPROCESSOR, not an answering agent
        """

        react_prompt = hub.pull("hwchase17/react")
        escaped_template = raw_system_template.replace('{', '{{').replace('}', '}}')
        
        custom_prompt = PromptTemplate.from_template(
            escaped_template + "\n\n" + react_prompt.template
        )
        tools = self.get_tools()
        agent = create_react_agent(self.llm, tools, custom_prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            memory=self.memory,
            max_iterations=5,
            return_intermediate_steps=True
        )

    
    def get_tools(self) -> list[BaseTool]:
        """List of callable tools."""
        return [fetch_ticker_url, fetch_peers, web_search]
    
    def _extract_json_from_result(self, result: dict, query: str) -> dict:
        """Extract and parse JSON from agent result dict."""
        raw_output = result.get("output", "") if isinstance(result, dict) else str(result)
        
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = raw_output
        
        try:
            parsed = self.parser.parse(json_str)
        except Exception:
            try:
                parsed = self.fixing_parser.parse(json_str)
            except Exception as e:
                parsed = {
                    "query": query,
                    "intent": None,
                    "ticker": None,
                    "peers": None,
                    "timeframe": None,
                    "metrics": None,
                    "url": None,
                    "output_from_websearch": None,
                    "answer": None,
                }
        
        # Cleanups
        parsed['query'] = query
        
        # If intent is irrelevant, set all other fields to null
        if parsed['intent'] == "irrelevant":
            parsed["ticker"] = None
            parsed["peers"] = None
            parsed["timeframe"] = None
            parsed["metrics"] = None
            parsed["url"] = None
            parsed["output_from_websearch"] = None
            parsed["answer"] = None
        
        return parsed

    def run(self, state: dict[str, str]) -> dict[str, str]:
        """Run the preprocessing pipeline and return validated structured output."""
        self.memory.clear()
        
        result = self.agent.invoke({"input" : state['query']})
        parsed = self._extract_json_from_result(result, state.get("query"))

        model = PreprocessModel(**parsed)
        return model.model_dump()

    
if __name__ == "__main__":
    agent = PreprocessAgent(model = "gpt-4o-mini")
    state = {"query": "Should I invest in Pulse Biosciences Inc?"}
    results = agent.run(state)
    print(f"\nFinal Result: {results}")
