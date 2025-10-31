
import os
import sys
from pathlib import Path
from typing import Any, Dict

import streamlit as st

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
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: inherit;
    }

    /* Recommendation boxes */
    .recommendation {
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
        text-align: center;
    }

    .rec-buy {
        background-color: rgba(76, 175, 80, 0.2);
        border: 2px solid #4CAF50;
        color: #4CAF50;
    }

    .rec-sell {
        background-color: rgba(244, 67, 54, 0.2);
        border: 2px solid #f44336;
        color: #f44336;
    }

    .rec-hold {
        background-color: rgba(255, 152, 0, 0.2);
        border: 2px solid #FF9800;
        color: #FF9800;
    }

    /* Info boxes */
    .success-box {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: inherit;
    }

    .warning-box {
        background-color: rgba(255, 152, 0, 0.1);
        border-left: 4px solid #FF9800;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: inherit;
    }

    .error-box {
        background-color: rgba(244, 67, 54, 0.1);
        border-left: 4px solid #f44336;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: inherit;
    }

    .info-box {
        background-color: rgba(33, 150, 243, 0.1);
        border-left: 4px solid #2196F3;
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

    /* Stop button styling - small square button */
    .stop-button button {
        background-color: #666 !important;
        color: white !important;
        border: 1px solid #888 !important;
        font-weight: normal !important;
        padding: 8px !important;
        border-radius: 4px !important;
        width: 40px !important;
        height: 40px !important;
        font-size: 18px !important;
        min-height: 40px !important;
    }

    .stop-button button:hover {
        background-color: #555 !important;
        border-color: #666 !important;
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

if "stop_generation" not in st.session_state:
    st.session_state.stop_generation = False

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
    st.caption("Powered by LangGraph & Ollama")

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
        html_parts.append(f"<h3>📌 {result['ticker']} - Company Analysis</h3>")
        if result.get("timeframe"):
            html_parts.append(f"<p><strong>Timeframe:</strong> {result['timeframe']}</p>")

    # Recommendation (highlight at top)
    if result.get("recommendation"):
        rec_class = get_recommendation_class(result["recommendation"])
        price_target_text = f" | Target: {result['price_target']}" if result.get("price_target") else ""
        html_parts.append(f"""
        <div class="recommendation {rec_class}">
            💼 <strong>Recommendation:</strong> {result['recommendation']}{price_target_text}
        </div>
        """)

    # Build comprehensive text output combining all available content
    comprehensive_text = []

    if result.get("executive_summary"):
        comprehensive_text.append(f"<strong>Executive Summary:</strong><br>{result['executive_summary']}")

    # Add sentiment info inline if available
    if result.get("sentiment_label"):
        sentiment_emoji = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}
        emoji = sentiment_emoji.get(result["sentiment_label"], "⚪")
        sentiment_text = f"{emoji} <strong>Sentiment:</strong> {result['sentiment_label'].upper()} (Confidence: {result.get('sentiment_confidence', 0):.2%})"
        comprehensive_text.append(sentiment_text)

    if result.get("financial_analysis"):
        comprehensive_text.append(f"<strong>Financial Analysis:</strong><br>{result['financial_analysis']}")

    if result.get("news_sentiment_analysis"):
        comprehensive_text.append(f"<strong>News & Sentiment:</strong><br>{result['news_sentiment_analysis']}")

    if result.get("investment_outlook"):
        comprehensive_text.append(f"<strong>Investment Outlook:</strong><br>{result['investment_outlook']}")

    # Fallback to text_summary if nothing else available
    if not comprehensive_text and result.get("text_summary"):
        comprehensive_text.append(result['text_summary'])

    # Display as single comprehensive content box
    if comprehensive_text:
        content_html = ('<br><br>'.join(comprehensive_text)).replace('\n', '<br>')
        html_parts.append("""
        <div class="metric-card">
            {content}
        </div>
        """.format(content=content_html))

    return "".join(html_parts)


def format_market_response(result: Dict[str, Any]) -> str:
    """Format response for finance-market intent"""
    html_parts = []

    # Intent badge
    html_parts.append(get_intent_badge(result.get("intent", "")))

    # Title
    html_parts.append("<h3>📈 Market Analysis</h3>")

    # Timeframe info
    if result.get("timeframe"):
        metrics = result.get("metrics") or ["General market trends"]
        if isinstance(metrics, (list, tuple)):
            focus_areas = ", ".join([str(m) for m in metrics])
        else:
            focus_areas = str(metrics)
        html_parts.append(f"<p><strong>Timeframe:</strong> {result['timeframe']} | <strong>Focus Areas:</strong> {focus_areas}</p>")

    # Build comprehensive text output combining all available content
    comprehensive_text = []

    if result.get("executive_summary"):
        comprehensive_text.append(f"<strong>Executive Summary:</strong><br>{result['executive_summary']}")

    if result.get("financial_analysis"):
        comprehensive_text.append(f"<strong>Market Analysis:</strong><br>{result['financial_analysis']}")

    if result.get("investment_outlook"):
        comprehensive_text.append(f"<strong>Market Outlook:</strong><br>{result['investment_outlook']}")

    # Fallback to text_summary if nothing else available
    if not comprehensive_text and result.get("text_summary"):
        comprehensive_text.append(result['text_summary'])

    # Display as single comprehensive content box
    if comprehensive_text:
        content_html = ('<br><br>'.join(comprehensive_text)).replace('\n', '<br>')
        html_parts.append("""
        <div class="metric-card">
            {content}
        </div>
        """.format(content=content_html))

    return "".join(html_parts)


def format_education_response(result: Dict[str, Any]) -> str:
    """Format response for finance-education intent"""
    html_parts = []

    # Intent badge
    html_parts.append(get_intent_badge(result.get("intent", "")))

    # Title
    html_parts.append("<h3>📚 Finance Education</h3>")

    # Build comprehensive text output combining all available content
    comprehensive_text = []

    # Start with executive summary or direct answer
    if result.get("executive_summary"):
        comprehensive_text.append(f"<strong>Answer:</strong><br>{result['executive_summary']}")
    elif result.get("answer"):
        comprehensive_text.append(f"<strong>Answer:</strong><br>{result['answer']}")

    if result.get("financial_analysis"):
        comprehensive_text.append(f"<strong>Detailed Explanation:</strong><br>{result['financial_analysis']}")

    if result.get("investment_outlook"):
        comprehensive_text.append(f"<strong>Practical Application:</strong><br>{result['investment_outlook']}")

    if result.get("recommendation"):
        comprehensive_text.append(f"<strong>Key Takeaways:</strong><br>{result['recommendation']}")

    # Fallback to text_summary if nothing else available
    if not comprehensive_text and result.get("text_summary"):
        comprehensive_text.append(result['text_summary'])

    # Display as single comprehensive content box
    if comprehensive_text:
        content_html = ('<br><br>'.join(comprehensive_text)).replace('\n', '<br>')
        html_parts.append("""
        <div class="metric-card">
            {content}
        </div>
        """.format(content=content_html))

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

# Create input area with stop button
input_col, stop_col = st.columns([0.95, 0.05])

with input_col:
    # Chat input - always visible
    if prompt := st.chat_input("Ask about stocks, market trends, or finance concepts...", key="chat_input"):
        user_input = prompt

with stop_col:
    # Show stop button only when processing
    if st.session_state.processing:
        st.markdown('<div class="stop-button">', unsafe_allow_html=True)
        if st.button("⬛", key="stop_button", help="Stop generation"):
            st.session_state.stop_generation = True
            st.session_state.processing = False
            st.session_state.messages.append({
                "role": "assistant",
                "content": "⚠️ Analysis was stopped by user.",
                "is_html": False
            })
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Process user input
if user_input and not st.session_state.processing:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Defer rendering to history on next rerun

    # Set processing flag
    st.session_state.processing = True
    st.session_state.stop_generation = False
    st.rerun()

# Run pipeline if processing
if st.session_state.processing and not st.session_state.stop_generation:
    # Show processing indicator
    with st.spinner("🔄 Analyzing your query..."):
        try:
            # Get the last user message
            last_user_message = None
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "user":
                    last_user_message = msg["content"]
                    break

            if last_user_message:
                # Run the pipeline
                result = st.session_state.orchestrator.run(last_user_message)

                # Format response based on intent
                formatted_response = format_response(result)

                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": formatted_response,
                    "is_html": True,
                    "raw_result": result  # Store for potential download
                })

                # Defer rendering to history on next rerun

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
    <p>Powered by LangGraph, Llama 3.1, and FinBERT</p>
    <p>© 2025 Equity Research AI Assistant | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
