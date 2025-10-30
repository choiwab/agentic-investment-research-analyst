"""
Equity Research Chatbot - Streamlit Interface

A conversational interface for the equity research LangGraph pipeline.
Supports all intent types: finance-company, finance-market, finance-education, and irrelevant.
"""

import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any

# Add paths for imports (must be BEFORE importing agents)
project_root = Path(__file__).parent.parent
backend_app_path = project_root / "backend" / "app"
backend_agents_path = backend_app_path / "agents"

# Add both paths so agents can find 'utils' module
for path in [str(backend_app_path), str(backend_agents_path)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Change to project root
os.chdir(str(project_root))

# Import the orchestrator (after path is set)
from agents.equity_research_graph import EquityResearchOrchestrator

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Equity Research AI Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
<style>
    /* Chat messages - transparent backgrounds with visible text */
    .user-message {
        background-color: rgba(33, 150, 243, 0.1);
        color: #1976D2;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
    }

    .assistant-message {
        background-color: transparent;
        color: inherit;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }

    /* Intent badges */
    .intent-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin: 5px 0;
    }

    .intent-company {
        background-color: #4CAF50;
        color: white;
    }

    .intent-market {
        background-color: #2196F3;
        color: white;
    }

    .intent-education {
        background-color: #FF9800;
        color: white;
    }

    .intent-irrelevant {
        background-color: #9E9E9E;
        color: white;
    }

    /* Metrics cards - semi-transparent */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 10px 0;
        color: inherit;
    }

    .metric-card h4 {
        color: inherit;
        margin-top: 0;
    }

    .metric-card p {
        color: inherit;
    }

    /* Recommendations */
    .recommendation {
        font-size: 18px;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }

    .rec-buy {
        background-color: rgba(76, 175, 80, 0.2);
        color: #4CAF50;
        border: 2px solid #4CAF50;
    }

    .rec-hold {
        background-color: rgba(255, 152, 0, 0.2);
        color: #FF9800;
        border: 2px solid #FF9800;
    }

    .rec-sell {
        background-color: rgba(244, 67, 54, 0.2);
        color: #F44336;
        border: 2px solid #F44336;
    }

    /* Processing indicator */
    .processing {
        color: #2196F3;
        font-style: italic;
    }

    /* Error messages */
    .error-box {
        background-color: rgba(244, 67, 54, 0.1);
        border-left: 4px solid #F44336;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: inherit;
    }

    /* Success messages */
    .success-box {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: inherit;
    }

    /* Make all text visible */
    .assistant-message * {
        color: inherit !important;
    }

    .metric-card * {
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "orchestrator" not in st.session_state:
    with st.spinner("Initializing AI Research Assistant..."):
        st.session_state.orchestrator = EquityResearchOrchestrator()

if "processing" not in st.session_state:
    st.session_state.processing = False

# ============================================================================
# SIDEBAR - CONFIGURATION & INFO
# ============================================================================

with st.sidebar:
    st.title("📊 Research Assistant")

    st.markdown("---")

    st.subheader("ℹ️ About")
    st.markdown("""
    This AI-powered equity research assistant provides:

    **🏢 Company Analysis**
    - Financial metrics & ratios
    - News sentiment analysis
    - Investment recommendations

    **📈 Market Analysis**
    - Economic trends
    - Market indicators
    - Sector insights

    **📚 Finance Education**
    - Concept explanations
    - Definitions
    - Educational content
    """)

    st.markdown("---")

    st.subheader("💡 Example Queries")

    if st.button("🏢 Analyze Tesla stock"):
        st.session_state.example_query = "Give me a comprehensive analysis on Tesla stock"

    if st.button("📈 Inflation trends"):
        st.session_state.example_query = "What are the current inflation trends?"

    if st.button("📚 What is P/E ratio?"):
        st.session_state.example_query = "What is P/E ratio and why is it important?"

    st.markdown("---")

    st.subheader("📊 Pipeline Status")

    # Show current processing status
    if st.session_state.processing:
        st.warning("🔄 Processing query...")
    else:
        st.success("✅ Ready")

    # Show statistics
    st.metric("Total Queries", len(st.session_state.messages) // 2)

    st.markdown("---")

    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Powered by LangGraph & OpenAI")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_intent_badge(intent: str) -> str:
    """Generate HTML badge for intent type"""
    intent_map = {
        "finance-company": ("Company Analysis", "intent-company"),
        "finance-market": ("Market Analysis", "intent-market"),
        "finance-education": ("Education", "intent-education"),
        "irrelevant": ("Irrelevant", "intent-irrelevant")
    }

    label, css_class = intent_map.get(intent, ("Unknown", "intent-irrelevant"))
    return f'<span class="intent-badge {css_class}">{label}</span>'


def get_recommendation_class(recommendation: str) -> str:
    """Get CSS class for recommendation"""
    if not recommendation:
        return ""

    rec_lower = recommendation.lower()
    if "buy" in rec_lower:
        return "rec-buy"
    elif "sell" in rec_lower:
        return "rec-sell"
    else:
        return "rec-hold"


def format_company_response(result: Dict[str, Any]) -> str:
    """Format response for finance-company intent"""
    html_parts = []

    # Intent badge
    html_parts.append(get_intent_badge(result.get("intent", "")))

    # Ticker info
    if result.get("ticker"):
        html_parts.append(f"<h3>📌 {result['ticker']}</h3>")
        if result.get("timeframe"):
            html_parts.append(f"<p><strong>Timeframe:</strong> {result['timeframe']}</p>")

    # Executive summary
    if result.get("executive_summary"):
        html_parts.append(f"""
        <div class="success-box">
            <h4>Executive Summary</h4>
            <p>{result['executive_summary']}</p>
        </div>
        """)

    # Recommendation
    if result.get("recommendation"):
        rec_class = get_recommendation_class(result["recommendation"])
        html_parts.append(f"""
        <div class="recommendation {rec_class}">
            💼 Recommendation: {result['recommendation']}
        </div>
        """)

    # Financial metrics (if available)
    if result.get("financials"):
        html_parts.append("<div class='metric-card'><h4>📊 Financial Metrics</h4>")

        financials = result["financials"]
        if isinstance(financials, dict):
            if "revenue" in financials:
                rev = financials["revenue"]
                html_parts.append(f"<p><strong>Revenue:</strong> ${rev.get('actual', 'N/A')}M (YoY: {rev.get('yoy_change', 'N/A')}%)</p>")

            if "eps" in financials:
                eps = financials["eps"]
                html_parts.append(f"<p><strong>EPS:</strong> ${eps.get('actual', 'N/A')} (vs Est: {eps.get('vs_estimate', 'N/A')}%)</p>")

        html_parts.append("</div>")

    # Sentiment analysis
    if result.get("sentiment_label"):
        sentiment_emoji = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}
        emoji = sentiment_emoji.get(result["sentiment_label"], "⚪")

        html_parts.append(f"""
        <div class="metric-card">
            <h4>💭 Sentiment Analysis</h4>
            <p>{emoji} <strong>{result['sentiment_label'].upper()}</strong>
            (Confidence: {result.get('sentiment_confidence', 0):.2%})</p>
        </div>
        """)

    # Text summary
    if result.get("text_summary"):
        html_parts.append(f"""
        <div class="metric-card">
            <h4>📝 Detailed Analysis</h4>
            <p>{result['text_summary']}</p>
        </div>
        """)

    # Table
    if result.get("table"):
        html_parts.append(f"""
        <div class="metric-card">
            <h4>📋 Data Table</h4>
            <div style="overflow-x:auto;">
                {result['table']}
            </div>
        </div>
        """)

    # Output files
    if result.get("pdf_path"):
        html_parts.append(f"""
        <div class="success-box">
            <p>📄 PDF Report generated: <code>{os.path.basename(result['pdf_path'])}</code></p>
        </div>
        """)

    if result.get("graphs"):
        html_parts.append(f"""
        <div class="success-box">
            <p>📊 {len(result['graphs'])} graphs generated</p>
        </div>
        """)

    return "".join(html_parts)


def format_market_response(result: Dict[str, Any]) -> str:
    """Format response for finance-market intent"""
    html_parts = []

    # Intent badge
    html_parts.append(get_intent_badge(result.get("intent", "")))

    # Title
    html_parts.append("<h3>📈 Market Analysis</h3>")

    # Timeframe
    if result.get("timeframe"):
        html_parts.append(f"<p><strong>Timeframe:</strong> {result['timeframe']}</p>")

    # Executive summary
    if result.get("executive_summary"):
        html_parts.append(f"""
        <div class="success-box">
            <h4>Executive Summary</h4>
            <p>{result['executive_summary']}</p>
        </div>
        """)

    # Market analysis
    if result.get("financial_analysis"):
        html_parts.append(f"""
        <div class="metric-card">
            <h4>📊 Market Insights</h4>
            <p>{result['financial_analysis']}</p>
        </div>
        """)

    # Outlook
    if result.get("investment_outlook"):
        html_parts.append(f"""
        <div class="metric-card">
            <h4>🔮 Outlook</h4>
            <p>{result['investment_outlook']}</p>
        </div>
        """)

    # Recommendation
    if result.get("recommendation"):
        html_parts.append(f"""
        <div class="recommendation rec-hold">
            💡 Key Takeaway: {result['recommendation']}
        </div>
        """)

    # Table
    if result.get("table"):
        html_parts.append(f"""
        <div class="metric-card">
            <div style="overflow-x:auto;">
                {result['table']}
            </div>
        </div>
        """)

    return "".join(html_parts)


def format_education_response(result: Dict[str, Any]) -> str:
    """Format response for finance-education intent"""
    html_parts = []

    # Intent badge
    html_parts.append(get_intent_badge(result.get("intent", "")))

    # Title
    html_parts.append("<h3>📚 Educational Content</h3>")

    # Direct answer (from preprocessing)
    if result.get("answer"):
        html_parts.append(f"""
        <div class="success-box">
            <h4>Quick Answer</h4>
            <p>{result['answer']}</p>
        </div>
        """)

    # Executive summary
    if result.get("executive_summary"):
        html_parts.append(f"""
        <div class="metric-card">
            <h4>Overview</h4>
            <p>{result['executive_summary']}</p>
        </div>
        """)

    # Detailed explanation
    if result.get("financial_analysis"):
        html_parts.append(f"""
        <div class="metric-card">
            <h4>Detailed Explanation</h4>
            <p>{result['financial_analysis']}</p>
        </div>
        """)

    # Key takeaways
    if result.get("recommendation"):
        html_parts.append(f"""
        <div class="metric-card">
            <h4>💡 Key Takeaways</h4>
            <p>{result['recommendation']}</p>
        </div>
        """)

    # Text summary
    if result.get("text_summary"):
        html_parts.append(f"""
        <div class="metric-card">
            <p>{result['text_summary']}</p>
        </div>
        """)

    return "".join(html_parts)


def format_irrelevant_response(result: Dict[str, Any]) -> str:
    """Format response for irrelevant intent"""
    html_parts = []

    html_parts.append(get_intent_badge("irrelevant"))

    html_parts.append("""
    <div class="error-box">
        <h4>⚠️ Query Not Relevant to Finance</h4>
        <p>I'm an equity research assistant specialized in financial analysis, market insights, and finance education.</p>
        <p><strong>I can help you with:</strong></p>
        <ul>
            <li>📊 Company stock analysis</li>
            <li>📈 Market trends and indicators</li>
            <li>📚 Financial concepts and definitions</li>
        </ul>
        <p>Please ask a finance-related question!</p>
    </div>
    """)

    return "".join(html_parts)


def format_response(result: Dict[str, Any]) -> str:
    """Route to appropriate formatter based on intent"""
    intent = result.get("intent", "irrelevant")

    if intent == "finance-company":
        return format_company_response(result)
    elif intent == "finance-market":
        return format_market_response(result)
    elif intent == "finance-education":
        return format_education_response(result)
    else:
        return format_irrelevant_response(result)


def display_message(role: str, content: str, is_html: bool = False):
    """Display a chat message"""
    if role == "user":
        st.markdown(f'<div class="user-message">👤 <strong>You:</strong><br/>{content}</div>', unsafe_allow_html=True)
    else:
        if is_html:
            st.markdown(f'<div class="assistant-message">🤖 <strong>AI Assistant:</strong><br/>{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">🤖 <strong>AI Assistant:</strong><br/><p>{content}</p></div>', unsafe_allow_html=True)


def display_error_message(errors: list):
    """Display error messages"""
    if errors:
        error_html = '<div class="error-box"><h4>⚠️ Errors Occurred:</h4><ul>'
        for error in errors:
            error_html += f"<li>{error}</li>"
        error_html += "</ul></div>"
        st.markdown(error_html, unsafe_allow_html=True)


# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

# Title
st.title("🤖 Equity Research AI Assistant")
st.markdown("Ask me anything about stocks, markets, or finance concepts!")

st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    display_message(
        message["role"],
        message["content"],
        message.get("is_html", False)
    )

# Handle example query from sidebar
if "example_query" in st.session_state and st.session_state.example_query:
    user_input = st.session_state.example_query
    del st.session_state.example_query
else:
    user_input = None

# Chat input
if not st.session_state.processing:
    if prompt := st.chat_input("Ask about stocks, market trends, or finance concepts...", key="chat_input"):
        user_input = prompt

# Process user input
if user_input and not st.session_state.processing:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    display_message("user", user_input)

    # Set processing flag
    st.session_state.processing = True

    # Show processing indicator
    with st.spinner("🔄 Analyzing your query..."):
        try:
            # Run the pipeline
            result = st.session_state.orchestrator.run(user_input)

            # Format response based on intent
            formatted_response = format_response(result)

            # Add assistant message to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": formatted_response,
                "is_html": True,
                "raw_result": result  # Store for potential download
            })

            # Display assistant response
            display_message("assistant", formatted_response, is_html=True)

            # Display errors if any
            if result.get("errors"):
                display_error_message(result["errors"])

            # Display processing time
            if result.get("processing_time"):
                st.caption(f"⏱️ Processing time: {result['processing_time']:.2f}s")

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "is_html": False
            })
            st.error(error_msg)

        finally:
            # Reset processing flag
            st.session_state.processing = False
            st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    <p>Powered by LangGraph, OpenAI GPT-4, Llama 3.1, and FinBERT</p>
    <p>© 2025 Equity Research AI Assistant | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
