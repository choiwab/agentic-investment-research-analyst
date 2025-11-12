"""
Equity Research LangGraph Pipeline

This module orchestrates the entire equity research workflow using LangGraph,
with conditional routing based on intent classification.

Pipeline routes:
1. Irrelevant intent: Direct to END (no processing)
2. finance-company: preprocessing → news_scraper → metric_extractor → sentiment_extractor → research_compiler
3. finance-market: preprocessing → news_scraper → research_compiler
4. finance-education: preprocessing → research_compiler (uses answer from preprocessing)
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

# Add paths so agents can import utils
backend_app_path = Path(__file__).parent.parent
backend_agents_path = Path(__file__).parent

# Add both paths
for path in [str(backend_app_path), str(backend_agents_path)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from agents.metric_extractor import MetricExtractorAgent
from agents.news_scraper import NewsScraperAgent

# Import agents (now that path is set)
from agents.preprocessor import PreprocessAgent
from agents.research_compiler import ResearchCompilerAgent
from agents.sentiment_extractor import SentimentAnalysisAgent
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# STATE SCHEMA - Preserves all data throughout the pipeline
# ============================================================================

class EquityResearchState(TypedDict, total=False):
    """
    Comprehensive state schema that preserves all data across pipeline nodes.

    Fields are accumulated as the workflow progresses, ensuring no data loss
    between agents.
    """
    # ===== User Input =====
    query: str  # Original user query

    # ===== Preprocessing Output =====
    intent: Optional[str]  # "finance-company" | "finance-market" | "finance-education" | "irrelevant"
    ticker: Optional[str]  # Company ticker symbol
    peers: Optional[List[str]]  # Peer company tickers
    timeframe: Optional[str]  # Time period for analysis
    metrics: Optional[List[str]]  # Requested financial metrics
    url: Optional[str]  # Finnhub news URL
    output_from_websearch: Optional[str]  # Web search results (market/education)
    answer: Optional[str]  # Direct answer for education queries

    # ===== News Scraper Output =====
    qualitative_summary: Optional[str]  # Textual insights from news
    quantitative_summary: Optional[str]  # Numerical data from news
    insight_outlook: Optional[str]  # News-based outlook

    # ===== Metric Extractor Output =====
    financials: Optional[Dict]  # Revenue, EPS, margins
    valuation: Optional[Dict]  # P/E, P/B, P/S ratios
    market: Optional[Dict]  # Price, 52-week high/low
    insider_sentiment: Optional[Dict]  # Insider trading signals
    quantitative_news: Optional[List[Dict]]  # Structured news metrics
    metric_evaluation: Optional[Dict]  # Performance evaluation

    # ===== Sentiment Extractor Output =====
    sentiment_label: Optional[str]  # "positive" | "neutral" | "negative"
    sentiment_confidence: Optional[float]  # Confidence score
    sentiment_probs: Optional[Dict]  # Probability distribution
    sentiment_stability: Optional[Dict]  # Variance and entropy metrics
    sentiment_evidence: Optional[Dict]  # Top positive/negative sentences

    # ===== Research Compiler Output =====
    executive_summary: Optional[str]  # Final executive summary
    financial_analysis: Optional[str]  # Compiled financial analysis
    news_sentiment_analysis: Optional[str]  # Combined news + sentiment
    investment_outlook: Optional[str]  # Forward-looking analysis
    recommendation: Optional[str]  # BUY/HOLD/SELL
    price_target: Optional[str]  # 12-month target

    # ===== Output Formats =====
    pdf_path: Optional[str]  # Path to generated PDF
    csv_path: Optional[str]  # Path to CSV export
    graphs: Optional[List[str]]  # Paths to generated graphs
    table: Optional[str]  # Markdown table
    text_summary: Optional[str]  # Text summary

    # ===== Metadata =====
    current_node: Optional[str]  # Track current processing node
    errors: Optional[List[str]]  # Error messages
    warnings: Optional[List[str]]  # Warning messages
    processing_time: Optional[float]  # Total processing time
    timestamp: Optional[str]  # Generation timestamp


# ============================================================================
# NODE FUNCTIONS - Wrapper functions for each agent
# ============================================================================

def preprocess_node(state: EquityResearchState) -> EquityResearchState:
    """
    Preprocessing node: Classifies intent and extracts metadata

    Updates state with: intent, ticker, peers, timeframe, metrics, url,
                        output_from_websearch, answer
    """
    logger.info("🔍 Preprocessing node - Classifying intent and extracting metadata")

    try:
        agent = PreprocessAgent(model="gpt-4o-mini")
        result = agent.run({"query": state["query"]})

        # Update state with preprocessing results
        state.update({
            "intent": result.get("intent"),
            "ticker": result.get("ticker"),
            "peers": result.get("peers"),
            "timeframe": result.get("timeframe"),
            "metrics": result.get("metrics"),
            "url": result.get("url"),
            "output_from_websearch": result.get("output_from_websearch"),
            "answer": result.get("answer"),
            "current_node": "preprocess"
        })

        logger.info(f"✅ Intent classified as: {result.get('intent')}")

    except Exception as e:
        logger.error(f"❌ Error in preprocessing: {str(e)}")
        state["errors"] = state.get("errors", []) + [f"Preprocessing error: {str(e)}"]
        state["intent"] = "irrelevant"  # Fail-safe to irrelevant

    return state


def news_scraper_node(state: EquityResearchState) -> EquityResearchState:
    """
    News Scraper node: Extracts qualitative and quantitative insights from news

    Updates state with: qualitative_summary, quantitative_summary, insight_outlook
    """
    logger.info("📰 News Scraper node - Analyzing news article")

    try:
        agent = NewsScraperAgent(model="gpt-4o-mini")

        # For finance-company, use the URL from preprocessing
        if state.get("intent") == "finance-company" and state.get("url"):
            result = agent.run({"url": state["url"]})

            # Update state with news analysis
            state.update({
                "qualitative_summary": result.get("qualitative_summary"),
                "quantitative_summary": result.get("quantitative_summary"),
                "insight_outlook": result.get("insight_outlook"),
                "current_node": "news_scraper"
            })

            logger.info("✅ News analysis completed")

        # For finance-market, we already have web search data, just acknowledge
        elif state.get("intent") == "finance-market":
            # News scraper can optionally enhance market analysis with web search
            # For now, pass through with web search data
            logger.info("✅ Market data from web search already available")
            state["current_node"] = "news_scraper"

    except Exception as e:
        logger.error(f"❌ Error in news scraper: {str(e)}")
        state["errors"] = state.get("errors", []) + [f"News scraper error: {str(e)}"]
        # Continue pipeline even if news scraping fails

    return state


def metric_extractor_node(state: EquityResearchState) -> EquityResearchState:
    """
    Metric Extractor node: Extracts and analyzes financial metrics

    Updates state with: financials, valuation, market, insider_sentiment,
                        quantitative_news, metric_evaluation
    """
    logger.info("📊 Metric Extractor node - Analyzing financial metrics")

    try:
        agent = MetricExtractorAgent(model="gpt-4o-mini")

        # Prepare news scraper output for context
        news_context = None
        if state.get("qualitative_summary") or state.get("quantitative_summary"):
            news_context = {
                state.get("ticker"): {
                    "qualitative_summary": state.get("qualitative_summary", ""),
                    "quantitative_summary": state.get("quantitative_summary", ""),
                    "insight_outlook": state.get("insight_outlook", "")
                }
            }

        # Run metric extraction
        result = agent.run(
            tickers=[state["ticker"]],
            timeframe=state.get("timeframe"),
            news_scraper_output=news_context
        )

        # Update state with metric results
        state.update({
            "financials": result.get("financials"),
            "valuation": result.get("valuation"),
            "market": result.get("market"),
            "insider_sentiment": result.get("insider_sentiment"),
            "quantitative_news": result.get("quantitative_news"),
            "metric_evaluation": result.get("metric_evaluation"),
            "current_node": "metric_extractor"
        })

        logger.info("✅ Metric extraction completed")

    except Exception as e:
        logger.error(f"❌ Error in metric extractor: {str(e)}")
        state["errors"] = state.get("errors", []) + [f"Metric extractor error: {str(e)}"]
        # Continue pipeline even if metric extraction fails

    return state


def sentiment_extractor_node(state: EquityResearchState) -> EquityResearchState:
    """
    Sentiment Extractor node: Analyzes sentiment using OpenAI

    Updates state with: sentiment_label, sentiment_confidence, sentiment_probs,
                        sentiment_stability, sentiment_evidence
    """
    logger.info("💭 Sentiment Extractor node - Analyzing sentiment with OpenAI")

    try:
        agent = SentimentAnalysisAgent(model_id="gpt-4o-mini")

        # Combine available text for sentiment analysis
        text_sources = []

        if state.get("qualitative_summary"):
            text_sources.append(state["qualitative_summary"])

        if state.get("insight_outlook"):
            text_sources.append(state["insight_outlook"])

        # If we have output from web search (shouldn't be the case for finance-company, but safe)
        if state.get("output_from_websearch"):
            text_sources.append(state["output_from_websearch"][:500])  # Limit length

        raw_text = "\n\n".join(text_sources)

        if raw_text:
            # Run sentiment analysis
            result = agent.run(
                raw_text=raw_text,
                meta={
                    "ticker": state.get("ticker"),
                    "timeframe": state.get("timeframe"),
                    "source": "equity_research_pipeline"
                },
                entities=[state.get("ticker")] if state.get("ticker") else None
            )

            # Update state with sentiment results
            if "sentiment" in result:
                state.update({
                    "sentiment_label": result["sentiment"].get("label"),
                    "sentiment_confidence": result["sentiment"].get("confidence"),
                    "sentiment_probs": result.get("probs"),
                    "sentiment_stability": result.get("stability"),
                    "sentiment_evidence": result.get("evidence"),
                    "current_node": "sentiment_extractor"
                })

                logger.info(f"✅ Sentiment analysis completed: {result['sentiment'].get('label')}")
        else:
            logger.warning("⚠️ No text available for sentiment analysis")
            state["warnings"] = state.get("warnings", []) + ["No text for sentiment analysis"]

    except Exception as e:
        logger.error(f"❌ Error in sentiment extractor: {str(e)}")
        state["errors"] = state.get("errors", []) + [f"Sentiment extractor error: {str(e)}"]
        # Continue pipeline even if sentiment analysis fails

    return state


def research_compiler_node(state: EquityResearchState) -> EquityResearchState:
    """
    Research Compiler node: Synthesizes all data into final report

    Updates state with: executive_summary, financial_analysis, news_sentiment_analysis,
                        investment_outlook, recommendation, price_target, output files
    """
    logger.info("📝 Research Compiler node - Generating final report")

    try:
        agent = ResearchCompilerAgent(model="gpt-4o-mini")

        # Prepare state for compiler based on intent
        compiler_input = {
            "intent": state.get("intent"),
            "user_query": state.get("query"),
            "ticker": state.get("ticker"),
            "timeframe": state.get("timeframe"),
            "output_format": ["text", "table", "pdf"]  # Request multiple formats
        }

        # Add intent-specific fields
        if state.get("intent") == "finance-company":
            # Company analysis - full suite of data
            compiler_input.update({
                "Peers": state.get("peers"),
                "Url": state.get("url"),
                "metric_extractor_result": {
                    "ticker": state.get("ticker"),
                    "timeframe": state.get("timeframe"),
                    "financials": state.get("financials"),
                    "valuation": state.get("valuation"),
                    "market": state.get("market"),
                    "insider_sentiment": state.get("insider_sentiment"),
                    "quantitative_news": state.get("quantitative_news"),
                    "metric_evaluation": state.get("metric_evaluation")
                },
                "news_scraper_result": {
                    "qualitative_summary": state.get("qualitative_summary"),
                    "quantitative_summary": state.get("quantitative_summary"),
                    "insight_outlook": state.get("insight_outlook")
                },
                "sentiment_extractor_result": {
                    "sentiment_label": state.get("sentiment_label"),
                    "sentiment_confidence": state.get("sentiment_confidence"),
                    "sentiment_probs": state.get("sentiment_probs"),
                    "sentiment_stability": state.get("sentiment_stability"),
                    "sentiment_evidence": state.get("sentiment_evidence")
                }
            })

        elif state.get("intent") == "finance-market":
            # Market analysis - web search + news context
            compiler_input.update({
                "outputFromWebSearch": state.get("output_from_websearch"),
                "metrics": state.get("metrics"),
                # Include news summaries if available from news scraper node
                "news_context": {
                    "qualitative_summary": state.get("qualitative_summary"),
                    "quantitative_summary": state.get("quantitative_summary"),
                    "insight_outlook": state.get("insight_outlook")
                } if state.get("qualitative_summary") else None
            })

        elif state.get("intent") == "finance-education":
            # Educational content
            compiler_input.update({
                "outputFromWebSearch": state.get("output_from_websearch"),
                "Result": state.get("answer"),
                "ticker": state.get("ticker")  # Optional context
            })

        # Run compiler
        result = agent.run(compiler_input)

        # Update state with compiler results
        state.update({
            "executive_summary": result.get("executive_summary"),
            "financial_analysis": result.get("financial_analysis"),
            "news_sentiment_analysis": result.get("news_sentiment_analysis"),
            "investment_outlook": result.get("investment_outlook"),
            "recommendation": result.get("recommendation"),
            "price_target": result.get("price_target"),
            "pdf_path": result.get("pdf_path"),
            "csv_path": result.get("csv_path"),
            "graphs": result.get("graphs"),
            "table": result.get("table"),
            "text_summary": result.get("text_summary"),
            "current_node": "research_compiler",
            "timestamp": datetime.now().isoformat()
        })

        logger.info("✅ Research compilation completed")
        logger.info(f"   Generated formats: {result.get('formats_generated', [])}")

    except Exception as e:
        logger.error(f"❌ Error in research compiler: {str(e)}")
        state["errors"] = state.get("errors", []) + [f"Research compiler error: {str(e)}"]

    return state


# ============================================================================
# ROUTING LOGIC - Conditional edges based on intent
# ============================================================================

def route_after_preprocess(state: EquityResearchState) -> Literal["irrelevant", "news_scraper", "research_compiler"]:
    """
    Routes the workflow after preprocessing based on intent.

    Routes:
    - irrelevant → END
    - finance-company → news_scraper
    - finance-market → news_scraper
    - finance-education → research_compiler (skip news/metrics)
    """
    intent = state.get("intent", "irrelevant")

    if intent == "irrelevant":
        logger.info("➡️ Routing to END (irrelevant intent)")
        return "irrelevant"

    elif intent in ["finance-company", "finance-market"]:
        logger.info(f"➡️ Routing to news_scraper ({intent})")
        return "news_scraper"

    elif intent == "finance-education":
        logger.info("➡️ Routing to research_compiler (education intent)")
        return "research_compiler"

    else:
        # Fallback to irrelevant for unknown intents
        logger.warning(f"⚠️ Unknown intent '{intent}', routing to END")
        return "irrelevant"


def route_after_news_scraper(state: EquityResearchState) -> Literal["metric_extractor", "research_compiler"]:
    """
    Routes the workflow after news scraping based on intent.

    Routes:
    - finance-company → metric_extractor (need financial metrics)
    - finance-market → research_compiler (skip metrics, use web search)
    """
    intent = state.get("intent", "irrelevant")

    if intent == "finance-company":
        logger.info("➡️ Routing to metric_extractor (company analysis)")
        return "metric_extractor"

    elif intent == "finance-market":
        logger.info("➡️ Routing to research_compiler (market analysis)")
        return "research_compiler"

    else:
        # Fallback (shouldn't happen if routing is correct)
        logger.warning(f"⚠️ Unexpected intent '{intent}' after news scraper, routing to compiler")
        return "research_compiler"


def route_after_metric_extractor(state: EquityResearchState) -> Literal["sentiment_extractor"]:
    """
    Routes the workflow after metric extraction.

    Always goes to sentiment_extractor for finance-company intent.
    """
    logger.info("➡️ Routing to sentiment_extractor")
    return "sentiment_extractor"


def route_after_sentiment_extractor(state: EquityResearchState) -> Literal["research_compiler"]:
    """
    Routes the workflow after sentiment extraction.

    Always goes to research_compiler to generate final report.
    """
    logger.info("➡️ Routing to research_compiler")
    return "research_compiler"


# ============================================================================
# GRAPH BUILDER - Construct the LangGraph workflow
# ============================================================================

def build_equity_research_graph() -> StateGraph:
    """
    Builds and returns the complete equity research LangGraph workflow.

    Workflow structure:

    START
      ↓
    [preprocess] ←── Intent classification
      ↓
      ├─ irrelevant → END
      ├─ finance-company → [news_scraper] → [metric_extractor] → [sentiment_extractor] → [research_compiler] → END
      ├─ finance-market → [news_scraper] → [research_compiler] → END
      └─ finance-education → [research_compiler] → END

    Returns:
        StateGraph: Compiled LangGraph workflow
    """
    logger.info("🏗️ Building Equity Research LangGraph pipeline")

    # Initialize graph with state schema
    workflow = StateGraph(EquityResearchState)

    # Add nodes
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("news_scraper", news_scraper_node)
    workflow.add_node("metric_extractor", metric_extractor_node)
    workflow.add_node("sentiment_extractor", sentiment_extractor_node)
    workflow.add_node("research_compiler", research_compiler_node)

    # Set entry point
    workflow.set_entry_point("preprocess")

    # Add conditional routing after preprocessing
    workflow.add_conditional_edges(
        "preprocess",
        route_after_preprocess,
        {
            "irrelevant": END,
            "news_scraper": "news_scraper",
            "research_compiler": "research_compiler"
        }
    )

    # Add conditional routing after news scraper
    workflow.add_conditional_edges(
        "news_scraper",
        route_after_news_scraper,
        {
            "metric_extractor": "metric_extractor",
            "research_compiler": "research_compiler"
        }
    )

    # Add edge from metric extractor to sentiment extractor
    workflow.add_conditional_edges(
        "metric_extractor",
        route_after_metric_extractor,
        {
            "sentiment_extractor": "sentiment_extractor"
        }
    )

    # Add edge from sentiment extractor to research compiler
    workflow.add_conditional_edges(
        "sentiment_extractor",
        route_after_sentiment_extractor,
        {
            "research_compiler": "research_compiler"
        }
    )

    # Add edge from research compiler to END
    workflow.add_edge("research_compiler", END)

    # Compile graph
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    logger.info("✅ Equity Research Graph built successfully")

    return graph


# ============================================================================
# MAIN EXECUTION INTERFACE
# ============================================================================

class EquityResearchOrchestrator:
    """
    High-level interface for running the equity research pipeline.

    Usage:
        orchestrator = EquityResearchOrchestrator()
        result = orchestrator.run("Analyze Tesla stock")
    """

    def __init__(self):
        """Initialize the orchestrator with the compiled graph"""
        self.graph = build_equity_research_graph()
        logger.info("🚀 Equity Research Orchestrator initialized")

    def run(self, query: str, config: Optional[Dict[str, Any]] = None) -> EquityResearchState:
        """
        Run the equity research pipeline for a given query.

        Args:
            query: User's research query
            config: Optional configuration for the graph execution

        Returns:
            EquityResearchState: Final state with all analysis results
        """
        start_time = datetime.now()
        logger.info(f"📊 Starting equity research for query: '{query}'")

        # Initialize state
        initial_state: EquityResearchState = {
            "query": query,
            "errors": [],
            "warnings": []
        }

        # Default config with thread for checkpointing
        if config is None:
            config = {"configurable": {"thread_id": f"thread_{datetime.now().timestamp()}"}}

        try:
            # Execute graph
            final_state = self.graph.invoke(initial_state, config)

            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            final_state["processing_time"] = processing_time

            logger.info(f"✅ Pipeline completed in {processing_time:.2f}s")
            logger.info(f"   Intent: {final_state.get('intent')}")
            logger.info(f"   Errors: {len(final_state.get('errors', []))}")
            logger.info(f"   Warnings: {len(final_state.get('warnings', []))}")

            return final_state

        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {str(e)}")
            raise

    def stream(self, query: str, config: Optional[Dict[str, Any]] = None):
        """
        Stream the equity research pipeline execution for real-time updates.

        Args:
            query: User's research query
            config: Optional configuration for the graph execution

        Yields:
            Tuple of (node_name, state_update)
        """
        logger.info(f"📊 Streaming equity research for query: '{query}'")

        # Initialize state
        initial_state: EquityResearchState = {
            "query": query,
            "errors": [],
            "warnings": []
        }

        # Default config with thread for checkpointing
        if config is None:
            config = {"configurable": {"thread_id": f"thread_{datetime.now().timestamp()}"}}

        try:
            # Stream graph execution
            for output in self.graph.stream(initial_state, config):
                yield output

        except Exception as e:
            logger.error(f"❌ Pipeline streaming failed: {str(e)}")
            raise


# ============================================================================
# TESTING / EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Test the equity research pipeline with different intent scenarios
    """

    print("="*80)
    print("EQUITY RESEARCH LANGGRAPH PIPELINE - TESTING")
    print("="*80)

    orchestrator = EquityResearchOrchestrator()

    # Test Case 1: Finance-Company Intent
    print("\n" + "="*80)
    print("TEST 1: Finance-Company Intent (Full Pipeline)")
    print("="*80)

    query1 = "Give me an analysis on Tesla stock for Q2 2024"

    try:
        result1 = orchestrator.run(query1)
        print(f"\n✅ Test 1 Results:")
        print(f"   Intent: {result1.get('intent')}")
        print(f"   Ticker: {result1.get('ticker')}")
        print(f"   Processing time: {result1.get('processing_time', 0):.2f}s")
        print(f"   Recommendation: {result1.get('recommendation', 'N/A')}")
        print(f"   PDF: {result1.get('pdf_path', 'Not generated')}")
        if result1.get('text_summary'):
            print(f"\n   Summary Preview:\n{result1['text_summary'][:300]}...")
    except Exception as e:
        print(f"❌ Test 1 failed: {str(e)}")

    # Test Case 2: Finance-Market Intent
    print("\n" + "="*80)
    print("TEST 2: Finance-Market Intent (News Scraper → Compiler)")
    print("="*80)

    query2 = "What are current inflation trends in the US economy?"

    try:
        result2 = orchestrator.run(query2)
        print(f"\n✅ Test 2 Results:")
        print(f"   Intent: {result2.get('intent')}")
        print(f"   Processing time: {result2.get('processing_time', 0):.2f}s")
        print(f"   Web search data available: {bool(result2.get('output_from_websearch'))}")
        if result2.get('text_summary'):
            print(f"\n   Summary Preview:\n{result2['text_summary'][:300]}...")
    except Exception as e:
        print(f"❌ Test 2 failed: {str(e)}")

    # Test Case 3: Finance-Education Intent
    print("\n" + "="*80)
    print("TEST 3: Finance-Education Intent (Direct to Compiler)")
    print("="*80)

    query3 = "What is P/E ratio and why is it important?"

    try:
        result3 = orchestrator.run(query3)
        print(f"\n✅ Test 3 Results:")
        print(f"   Intent: {result3.get('intent')}")
        print(f"   Processing time: {result3.get('processing_time', 0):.2f}s")
        print(f"   Answer available: {bool(result3.get('answer'))}")
        if result3.get('text_summary'):
            print(f"\n   Summary Preview:\n{result3['text_summary'][:300]}...")
    except Exception as e:
        print(f"❌ Test 3 failed: {str(e)}")

    # Test Case 4: Irrelevant Intent
    print("\n" + "="*80)
    print("TEST 4: Irrelevant Intent (Should End Immediately)")
    print("="*80)

    query4 = "Tell me a joke about cats"

    try:
        result4 = orchestrator.run(query4)
        print(f"\n✅ Test 4 Results:")
        print(f"   Intent: {result4.get('intent')}")
        print(f"   Processing time: {result4.get('processing_time', 0):.2f}s")
        print(f"   Pipeline terminated early: {result4.get('current_node') == 'preprocess'}")
    except Exception as e:
        print(f"❌ Test 4 failed: {str(e)}")

    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED")
    print("="*80)
