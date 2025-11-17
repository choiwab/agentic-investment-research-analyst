"""
Research Compiler Agent

This agent synthesizes outputs from various upstream agents and compiles them into
user-requested formats (PDF reports, CSV files, graphs, tables, or text summaries).

Handles multiple intents:
- finance-company: Full analysis with metrics, news, sentiment
- finance-market: Market analysis with web search results
- finance-education: Educational content compilation
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from io import BytesIO
import base64
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, tool, initialize_agent, AgentExecutor
from langchain.tools import BaseTool
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate

# PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor

# Import utils
from utils.callback_handler import PrintCallbackHandler
from utils.model_schema import ResearchCompilerModel
from utils.conversation_buffer_safe import SafeConversationMemory


class ResearchCompilerAgent:
    """Agent that compiles research from various sources into formatted reports"""

    # Define supported intents and their characteristics
    INTENT_CONFIGS = {
        "finance-company": {
            "requires_ticker": True,
            "expected_fields": ["metric_extractor_result", "news_scraper_result", "sentiment_extractor_result"],
            "optional_fields": ["Peers", "Url"],
            "output_types": ["pdf", "csv", "graph", "table", "text"],
            "synthesis_focus": "company_analysis"
        },
        "finance-market": {
            "requires_ticker": False,
            "expected_fields": ["outputFromWebSearch"],
            "optional_fields": ["metrics", "timeframe"],
            "output_types": ["text", "table", "pdf"],
            "synthesis_focus": "market_trends"
        },
        "finance-education": {
            "requires_ticker": False,
            "expected_fields": ["outputFromWebSearch", "Result"],
            "optional_fields": ["ticker"],
            "output_types": ["text", "pdf"],
            "synthesis_focus": "educational_content"
        }
    }

    def __init__(self, model: str = "gpt-4o") -> None:
        """
        Initialize the Research Compiler Agent

        Args:
            model: Model name (default: gpt-4o for OpenAI)
        """
        self.callback_handler = PrintCallbackHandler()

        # Use OpenAI LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.3,
            streaming=True,
            callbacks=[self.callback_handler]
        )

        self.memory = SafeConversationMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.parser = StructuredOutputParser.from_response_schemas(ResearchCompilerModel.response_schema)
        self.agent = self.build_agent()

        # Set up matplotlib style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('default')
        sns.set_palette("husl")

        # Create output directories
        self.output_dir = "research_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/pdf", exist_ok=True)
        os.makedirs(f"{self.output_dir}/csv", exist_ok=True)
        os.makedirs(f"{self.output_dir}/graphs", exist_ok=True)

    def validate_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize input state based on intent

        Args:
            state: Input state dictionary

        Returns:
            Validated and normalized state with metadata
        """
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "normalized_state": state.copy()
        }

        # Get intent, default to finance-company if not specified
        intent = state.get("intent", "finance-company").lower()

        # Validate intent
        if intent not in self.INTENT_CONFIGS:
            logger.warning(f"Unknown intent '{intent}', defaulting to 'finance-company'")
            validation_result["warnings"].append(f"Unknown intent '{intent}', using 'finance-company'")
            intent = "finance-company"
            validation_result["normalized_state"]["intent"] = intent

        config = self.INTENT_CONFIGS[intent]

        # Check required fields based on intent
        if config["requires_ticker"] and not state.get("ticker"):
            if intent == "finance-company":
                validation_result["errors"].append("Missing required 'ticker' for company analysis")
                validation_result["is_valid"] = False

        # Check for expected fields and warn if missing
        for field in config["expected_fields"]:
            if field not in state:
                validation_result["warnings"].append(f"Expected field '{field}' not found for intent '{intent}'")

        # Normalize output format
        if "output_format" not in state:
            # Use default formats based on intent
            validation_result["normalized_state"]["output_format"] = ["text"]
        elif not isinstance(state["output_format"], list):
            validation_result["normalized_state"]["output_format"] = [state["output_format"]]

        # Ensure Query field exists
        if "Query" not in state and "user_query" not in state:
            validation_result["warnings"].append("No query found, using default")
            validation_result["normalized_state"]["user_query"] = "Provide analysis based on available data"
        elif "Query" in state:
            validation_result["normalized_state"]["user_query"] = state["Query"]

        return validation_result

    def build_agent(self) -> AgentExecutor:
        """Build the REACT agent for research compilation"""

        format_instructions = self.parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

        system_template = f"""
        You are a senior research analyst capable of handling various types of financial analysis:
        1. Company Analysis - Detailed equity research on specific companies
        2. Market Analysis - Broad market trends and insights
        3. Educational Content - Finance concepts and educational material

        Adapt your analysis style based on the intent and available data.
        Always maintain objectivity and accuracy.

        Output your synthesis as JSON:
        {format_instructions}

        Key principles:
        - Use only available data, don't hallucinate metrics
        - Clearly indicate when data is missing
        - Adapt tone and depth based on the query type
        - Provide actionable insights when possible
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

    def get_tools(self) -> list[BaseTool]:
        """Define tools for the research compiler agent"""

        @tool
        def analyze_financial_metrics(metrics_data: str) -> str:
            """Analyze financial metrics and identify key trends and insights"""
            try:
                data = json.loads(metrics_data) if isinstance(metrics_data, str) else metrics_data

                insights = []
                if "financials" in data:
                    financials = data["financials"]
                    if "revenue" in financials:
                        rev = financials["revenue"]
                        insights.append(f"Revenue: ${rev.get('actual', 'N/A')}M (YoY: {rev.get('yoy_change', 'N/A')}%)")
                    if "eps" in financials:
                        eps = financials["eps"]
                        insights.append(f"EPS: ${eps.get('actual', 'N/A')} (vs estimate: {eps.get('vs_estimate', 'N/A')}%)")

                if "valuation" in data:
                    val = data["valuation"]
                    insights.append(f"P/E Ratio: {val.get('pe_ratio', 'N/A')}")
                    insights.append(f"Market Cap: ${val.get('market_cap', 'N/A')}B")

                return "\n".join(insights) if insights else "No financial metrics available"

            except Exception as e:
                return f"Error analyzing metrics: {str(e)}"

        @tool
        def synthesize_web_content(web_data: str) -> str:
            """Synthesize web search results and extract key information"""
            try:
                if not web_data:
                    return "No web search data available"

                # Process web search output
                summary_points = []

                # Extract key points (adapt based on actual web search format)
                if isinstance(web_data, str):
                    # Take first 500 chars and create summary
                    content = web_data[:500] if len(web_data) > 500 else web_data
                    summary_points.append(f"Web Research: {content}...")

                return "\n".join(summary_points) if summary_points else "Unable to process web content"

            except Exception as e:
                return f"Error synthesizing web content: {str(e)}"

        @tool
        def generate_educational_summary(content: str) -> str:
            """Generate educational summary from available content"""
            try:
                if not content:
                    return "No educational content available"

                # Create structured educational summary
                summary = "Educational Overview:\n"

                if isinstance(content, dict):
                    for key, value in content.items():
                        summary += f"- {key}: {str(value)[:100]}...\n"
                else:
                    summary += str(content)[:500] + "..."

                return summary

            except Exception as e:
                return f"Error generating educational summary: {str(e)}"

        @tool
        def analyze_peer_comparison(peers_data: str) -> str:
            """Analyze peer companies for comparison"""
            try:
                if not peers_data:
                    return "No peer comparison data available"

                peers = json.loads(peers_data) if isinstance(peers_data, str) else peers_data

                comparison = "Peer Comparison:\n"
                if isinstance(peers, list):
                    comparison += f"Comparing against: {', '.join(peers)}\n"
                    comparison += "Detailed peer analysis would require additional data fetching"

                return comparison

            except Exception as e:
                return f"Error analyzing peers: {str(e)}"

        return [analyze_financial_metrics, synthesize_web_content, generate_educational_summary, analyze_peer_comparison]

    def synthesize_by_intent(self, state: Dict[str, Any], intent: str) -> Dict[str, Any]:
        """
        Generate synthesis based on intent type

        Args:
            state: Normalized state dictionary
            intent: The intent type

        Returns:
            Synthesis dictionary
        """
        synthesis_prompts = {
            "finance-company": self._create_company_synthesis_prompt,
            "finance-market": self._create_market_synthesis_prompt,
            "finance-education": self._create_education_synthesis_prompt
        }

        # Get appropriate prompt creator
        prompt_creator = synthesis_prompts.get(intent, self._create_company_synthesis_prompt)
        synthesis_prompt = prompt_creator(state)

        # Get synthesis from agent
        try:
            agent_result = self.agent.invoke({"input": synthesis_prompt})

            # Gracefully handle missing or malformed outputs
            output_payload = None
            if isinstance(agent_result, dict):
                output_payload = agent_result.get('output')

            if isinstance(output_payload, dict):
                synthesis = output_payload
            elif isinstance(output_payload, str):
                # For market analysis, the agent returns plain text, so wrap it in a dict
                if intent == "finance-market":
                    synthesis = {
                        "executive_summary": output_payload,
                        "financial_analysis": "",
                        "news_sentiment_analysis": "",
                        "investment_outlook": "",
                        "recommendation": ""
                    }
                    logger.info(f"Using agent's text output for market analysis ({len(output_payload)} chars)")
                else:
                    try:
                        synthesis = self.parser.parse(output_payload)
                    except Exception:
                        synthesis = self._create_fallback_synthesis(state, intent)
            else:
                # No usable output; fall back
                synthesis = self._create_fallback_synthesis(state, intent)

        except Exception as e:
            logger.error(f"Error in synthesis: {str(e)}")
            synthesis = self._create_fallback_synthesis(state, intent)

        return synthesis

    def _create_company_synthesis_prompt(self, state: Dict[str, Any]) -> str:
        """Create synthesis prompt for company analysis"""
        ticker = state.get("ticker", "UNKNOWN")
        user_query = state.get("user_query", "Analyze the company")

        prompt = f"""
        Analyze the following investment data for {ticker} and provide a comprehensive synthesis:

        User Query: {user_query}

        Available Data:
        - Metrics: {json.dumps(state.get('metric_extractor_result', {}))[:500]}
        - News: {json.dumps(state.get('news_scraper_result', {}))[:500]}
        - Sentiment: {json.dumps(state.get('sentiment_extractor_result', {}))[:500]}
        """

        # Add peer comparison if available
        if "Peers" in state:
            prompt += f"\n- Peer Companies: {state['Peers']}"

        prompt += """

        Please provide:
        1. Executive summary (2-3 sentences)
        2. Financial analysis (key metrics and trends)
        3. News and sentiment analysis
        4. Investment outlook
        5. Clear recommendation (BUY/HOLD/SELL) with rationale
        """

        return prompt

    def _create_market_synthesis_prompt(self, state: Dict[str, Any]) -> str:
        """Create synthesis prompt for market analysis"""
        user_query = state.get("user_query", "Provide market analysis")

        prompt = f"""
        Provide a market analysis based on the following information:

        User Query: {user_query}
        Timeframe: {state.get('timeframe', 'Recent')}
        Metrics of Interest: {state.get('metrics', [])}

        Web Search Results: {str(state.get('outputFromWebSearch') or 'No web search data')[:1000]}

        Please provide:
        1. Executive summary of market conditions
        2. Key trends and insights
        3. Market outlook
        4. Relevant metrics analysis if available
        5. Recommendations or key takeaways
        """

        return prompt

    def _create_education_synthesis_prompt(self, state: Dict[str, Any]) -> str:
        """Create synthesis prompt for educational content"""
        user_query = state.get("user_query", "Explain the financial concept")

        prompt = f"""
        Create educational content based on the following:

        Query: {user_query}

        Web Search Information: {str(state.get('outputFromWebSearch') or '')[:800]}

        Additional Context: {str(state.get('Result') or '')[:500]}

        Please provide:
        1. Executive summary (clear explanation of the concept)
        2. Detailed explanation with examples
        3. Key principles or formulas if applicable
        4. Practical applications
        5. Summary and key takeaways
        """

        if state.get("ticker"):
            prompt += f"\n\nRelated to ticker: {state['ticker']} for context"

        return prompt

    def _create_fallback_synthesis(self, state: Dict[str, Any], intent: str) -> Dict[str, Any]:
        """Create fallback synthesis when agent fails"""

        fallback_templates = {
            "finance-company": {
                "executive_summary": f"Analysis for {state.get('ticker', 'company')} based on available data.",
                "financial_analysis": "Financial metrics require additional data.",
                "news_sentiment_analysis": "News and sentiment data pending.",
                "investment_outlook": "Further analysis needed for comprehensive outlook.",
                "recommendation": "HOLD - Insufficient data for strong recommendation"
            },
            "finance-market": {
                "executive_summary": "Market analysis based on current information.",
                "financial_analysis": "Market trends show mixed signals.",
                "news_sentiment_analysis": "Market sentiment varies across sectors.",
                "investment_outlook": "Monitor key indicators for clearer direction.",
                "recommendation": "NEUTRAL - Market conditions uncertain"
            },
            "finance-education": {
                "executive_summary": "Educational overview of the requested topic.",
                "financial_analysis": "Concept explanation provided.",
                "news_sentiment_analysis": "Not applicable for educational content.",
                "investment_outlook": "Understanding fundamental concepts is key.",
                "recommendation": "Continue learning with practical examples"
            }
        }

        return fallback_templates.get(intent, fallback_templates["finance-company"])

    def generate_pdf_report(self, state: Dict[str, Any], synthesis: Dict[str, Any]) -> str:
        """Generate a professional PDF research report adapted to intent"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intent = state.get("intent", "finance-company")

        # Determine filename based on intent
        if intent == "finance-company":
            identifier = state.get("ticker", "UNKNOWN")
            report_type = "company_research"
        elif intent == "finance-market":
            identifier = "market"
            report_type = "market_analysis"
        else:  # finance-education
            identifier = "education"
            report_type = "educational_content"

        pdf_path = f"{self.output_dir}/pdf/{identifier}_{report_type}_{timestamp}.pdf"

        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=HexColor('#1e3a8a'),
            spaceAfter=30,
            alignment=TA_CENTER
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=HexColor('#1e3a8a'),
            spaceAfter=12
        )

        # Title Page - Adapt based on intent
        if intent == "finance-company":
            story.append(Paragraph(f"Investment Research Report", title_style))
            story.append(Paragraph(f"{state.get('ticker', 'N/A')}", title_style))
        elif intent == "finance-market":
            story.append(Paragraph(f"Market Analysis Report", title_style))
            story.append(Paragraph(f"{state.get('timeframe', 'Current Period')}", styles['Normal']))
        else:
            story.append(Paragraph(f"Educational Finance Content", title_style))
            story.append(Paragraph(f"Topic: {state.get('user_query', 'Finance Concepts')[:50]}", styles['Normal']))

        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        story.append(PageBreak())

        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        exec_summary = synthesis.get("executive_summary", "No executive summary available.")
        story.append(Paragraph(exec_summary, styles['BodyText']))
        story.append(Spacer(1, 0.3*inch))

        # Content sections based on intent
        if intent == "finance-company" and "metric_extractor_result" in state:
            # Financial Analysis section for company reports
            story.append(Paragraph("Financial Analysis", heading_style))
            self._add_financial_tables_to_pdf(story, state, styles)
            financial_analysis = synthesis.get("financial_analysis", "No financial analysis available.")
            story.append(Paragraph(financial_analysis, styles['BodyText']))
            story.append(Spacer(1, 0.3*inch))

        # Common sections
        if synthesis.get("news_sentiment_analysis") and synthesis["news_sentiment_analysis"] != "Not applicable for educational content.":
            story.append(Paragraph("Analysis & Insights", heading_style))
            news_analysis = synthesis.get("news_sentiment_analysis", "No analysis available.")
            story.append(Paragraph(news_analysis, styles['BodyText']))
            story.append(Spacer(1, 0.3*inch))

        # Outlook/Conclusion
        story.append(Paragraph("Outlook & Recommendations", heading_style))
        outlook = synthesis.get("investment_outlook", "No outlook available.")
        story.append(Paragraph(outlook, styles['BodyText']))
        story.append(Spacer(1, 0.2*inch))

        recommendation = synthesis.get("recommendation", "No recommendation available.")
        story.append(Paragraph(f"<b>Recommendation:</b> {recommendation}", styles['BodyText']))

        # Add graphs if available and relevant
        if intent == "finance-company":
            graph_paths = self.generate_graphs(state)
            if graph_paths:
                story.append(PageBreak())
                story.append(Paragraph("Charts & Visualizations", heading_style))
                for graph_path in graph_paths:
                    if os.path.exists(graph_path):
                        try:
                            img = Image(graph_path, width=6*inch, height=4*inch)
                            story.append(img)
                            story.append(Spacer(1, 0.2*inch))
                        except Exception as e:
                            logger.warning(f"Could not add graph to PDF: {e}")

        # Build PDF
        doc.build(story)

        return pdf_path

    def _add_financial_tables_to_pdf(self, story: list, state: Dict[str, Any], styles: any) -> None:
        """Helper method to add financial tables to PDF"""

        if "metric_extractor_result" in state:
            metrics = state["metric_extractor_result"]
            if "financials" in metrics:
                fin_data = []
                fin_data.append(['Metric', 'Value', 'YoY Change'])

                if "revenue" in metrics["financials"]:
                    rev = metrics["financials"]["revenue"]
                    fin_data.append(['Revenue', f"${rev.get('actual', 'N/A')}M", f"{rev.get('yoy_change', 'N/A')}%"])

                if "eps" in metrics["financials"]:
                    eps = metrics["financials"]["eps"]
                    fin_data.append(['EPS', f"${eps.get('actual', 'N/A')}", f"{eps.get('vs_estimate', 'N/A')}% vs Est"])

                if len(fin_data) > 1:
                    table = Table(fin_data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 14),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(table)
                    story.append(Spacer(1, 0.3*inch))

    def generate_csv_export(self, state: Dict[str, Any]) -> Optional[str]:
        """Export data to CSV format (mainly for finance-company intent)"""

        intent = state.get("intent", "finance-company")

        # CSV export is most relevant for company analysis with metrics
        if intent != "finance-company" or "metric_extractor_result" not in state:
            logger.info(f"CSV export not applicable for intent: {intent}")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ticker = state.get("ticker", "UNKNOWN")
        csv_path = f"{self.output_dir}/csv/{ticker}_data_{timestamp}.csv"

        # Prepare data for CSV
        rows = []

        # Add basic info
        rows.append({
            "Category": "Basic Info",
            "Metric": "Ticker",
            "Value": ticker,
            "Change_YoY": "",
            "vs_Estimate": "",
            "Date": datetime.now().strftime("%Y-%m-%d")
        })

        # Add financial metrics if available
        if "metric_extractor_result" in state:
            metrics = state["metric_extractor_result"]

            if "financials" in metrics:
                for key, value in metrics["financials"].items():
                    if isinstance(value, dict):
                        rows.append({
                            "Category": "Financials",
                            "Metric": key.upper(),
                            "Value": value.get("actual", ""),
                            "Change_YoY": value.get("yoy_change", ""),
                            "vs_Estimate": value.get("vs_estimate", ""),
                            "Date": metrics.get("timeframe", "")
                        })

            if "valuation" in metrics:
                for key, value in metrics["valuation"].items():
                    rows.append({
                        "Category": "Valuation",
                        "Metric": key.upper(),
                        "Value": value,
                        "Change_YoY": "",
                        "vs_Estimate": "",
                        "Date": datetime.now().strftime("%Y-%m-%d")
                    })

        # Add peer information if available
        if "Peers" in state:
            for peer in state["Peers"]:
                rows.append({
                    "Category": "Peers",
                    "Metric": "Competitor",
                    "Value": peer,
                    "Change_YoY": "",
                    "vs_Estimate": "",
                    "Date": datetime.now().strftime("%Y-%m-%d")
                })

        # Create DataFrame and save to CSV
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        return csv_path

    def generate_graphs(self, state: Dict[str, Any]) -> List[str]:
        """Generate matplotlib graphs for visualization (mainly for company analysis)"""

        graph_paths = []
        intent = state.get("intent", "finance-company")

        # Graphs are most relevant for company analysis
        if intent != "finance-company" or "metric_extractor_result" not in state:
            return graph_paths

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ticker = state.get("ticker", "UNKNOWN")

        metrics = state["metric_extractor_result"]

        # 1. Financial Metrics Comparison Chart
        if "financials" in metrics:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))

                categories = []
                values = []
                colors_list = []

                for key, value in metrics["financials"].items():
                    if isinstance(value, dict) and "actual" in value:
                        categories.append(key.upper())
                        actual_val = value["actual"]
                        # Normalize values for comparison
                        if key == "revenue":
                            values.append(actual_val / 1000 if actual_val else 0)  # Show in billions
                        elif key == "eps":
                            values.append(actual_val * 100 if actual_val else 0)  # Scale up for visibility
                        else:
                            values.append(actual_val if actual_val else 0)

                        # Color based on performance
                        if value.get("yoy_change", 0) > 0:
                            colors_list.append('#10b981')  # Green for positive
                        else:
                            colors_list.append('#ef4444')  # Red for negative

                if categories:
                    bars = ax.bar(categories, values, color=colors_list)
                    ax.set_title(f'{ticker} - Financial Metrics Overview', fontsize=16, fontweight='bold')
                    ax.set_ylabel('Normalized Values', fontsize=12)
                    ax.set_xlabel('Metrics', fontsize=12)
                    ax.grid(axis='y', alpha=0.3)

                    # Add value labels on bars
                    for bar, val in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{val:.1f}', ha='center', va='bottom')

                    plt.tight_layout()
                    graph_path = f"{self.output_dir}/graphs/{ticker}_metrics_{timestamp}.png"
                    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    graph_paths.append(graph_path)
            except Exception as e:
                logger.error(f"Error generating metrics graph: {e}")

        # 2. Valuation Metrics Chart
        if "valuation" in metrics:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))

                val_metrics = metrics["valuation"]
                if val_metrics:
                    labels = []
                    sizes = []
                    colors_pie = []

                    # Create a pie chart for valuation breakdown
                    for key, value in val_metrics.items():
                        if value and isinstance(value, (int, float)):
                            labels.append(key.replace("_", " ").upper())
                            sizes.append(abs(value))  # Use absolute values for pie chart
                            # Generate color from palette
                            color_rgb = sns.color_palette("husl", len(val_metrics))[len(labels)-1][:3]
                            colors_pie.append('#' + ''.join([f'{int(i*255):02x}' for i in color_rgb]))

                    if labels:
                        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie, startangle=90)
                        ax.set_title(f'{ticker} - Valuation Metrics Distribution', fontsize=16, fontweight='bold')

                        plt.tight_layout()
                        graph_path = f"{self.output_dir}/graphs/{ticker}_valuation_{timestamp}.png"
                        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        graph_paths.append(graph_path)
            except Exception as e:
                logger.error(f"Error generating valuation graph: {e}")

        return graph_paths

    def generate_table(self, state: Dict[str, Any]) -> str:
        """Generate a formatted markdown table based on intent and available data"""

        intent = state.get("intent", "finance-company")
        table_lines = []

        if intent == "finance-company":
            ticker = state.get("ticker", "UNKNOWN")
            table_lines.append(f"## {ticker} - Financial Summary Table\n")
            table_lines.append("| Metric | Value | YoY Change | vs Estimate |")
            table_lines.append("|--------|-------|------------|-------------|")

            # Add data rows if available
            if "metric_extractor_result" in state:
                metrics = state["metric_extractor_result"]

                if "financials" in metrics:
                    for key, value in metrics["financials"].items():
                        if isinstance(value, dict):
                            metric_name = key.upper()
                            actual = value.get("actual", "N/A")
                            yoy = value.get("yoy_change", "N/A")
                            vs_est = value.get("vs_estimate", "N/A")

                            if key == "revenue":
                                actual = f"${actual}M" if actual != "N/A" else actual
                            elif key == "eps":
                                actual = f"${actual}" if actual != "N/A" else actual

                            yoy_str = f"{yoy}%" if yoy != "N/A" else yoy
                            vs_est_str = f"{vs_est}%" if vs_est != "N/A" else vs_est

                            # Add color indicators
                            if yoy != "N/A" and yoy > 0:
                                yoy_str = f"🟢 {yoy_str}"
                            elif yoy != "N/A" and yoy < 0:
                                yoy_str = f"🔴 {yoy_str}"

                            table_lines.append(f"| {metric_name} | {actual} | {yoy_str} | {vs_est_str} |")

                if "valuation" in metrics:
                    table_lines.append(f"| **Valuation** | | | |")
                    for key, value in metrics["valuation"].items():
                        metric_name = key.replace("_", " ").upper()
                        table_lines.append(f"| {metric_name} | {value} | - | - |")

        elif intent == "finance-market":
            table_lines.append(f"## Market Analysis Summary\n")
            table_lines.append("| Aspect | Details |")
            table_lines.append("|--------|---------|")

            table_lines.append(f"| Timeframe | {state.get('timeframe', 'Current')} |")

            if "metrics" in state:
                metrics_value = state.get("metrics")
                if isinstance(metrics_value, (list, tuple)):
                    metrics_str = ", ".join([str(m) for m in metrics_value])
                elif metrics_value:
                    metrics_str = str(metrics_value)
                else:
                    metrics_str = ""
                if metrics_str:
                    table_lines.append(f"| Metrics | {metrics_str} |")

            if "outputFromWebSearch" in state:
                web_data = state.get("outputFromWebSearch")
                if web_data:
                    # Try to parse and extract URLs
                    try:
                        import json
                        import ast
                        logger.info(f"[DEBUG] Web data type: {type(web_data)}, value: {str(web_data)[:200]}")

                        # If it's a dict, use it directly
                        if isinstance(web_data, dict):
                            results = web_data.get("results", [])
                            logger.info(f"[DEBUG] Dict path - results: {len(results)} items")
                        # If it's a string, try to parse it
                        elif isinstance(web_data, str):
                            # Try ast.literal_eval first (safer for Python dict strings)
                            try:
                                parsed = ast.literal_eval(web_data)
                                results = parsed.get("results", [])
                                logger.info(f"[DEBUG] AST path - parsed results: {len(results)} items")
                            except:
                                # Fallback to JSON parsing with quote replacement
                                web_data_json = web_data.replace("'", '"')
                                parsed = json.loads(web_data_json)
                                results = parsed.get("results", [])
                                logger.info(f"[DEBUG] JSON path - parsed results: {len(results)} items")
                        else:
                            results = []
                            logger.info(f"[DEBUG] Unknown type path")

                        # Extract URLs from results
                        if results and len(results) > 0:
                            urls = [item.get("url", "") for item in results if item.get("url")]
                            logger.info(f"[DEBUG] Extracted {len(urls)} URLs")
                            if urls:
                                # Format URLs cleanly
                                url_links = ", ".join([f"[Source {i+1}]({url})" for i, url in enumerate(urls[:5])])  # Show max 5 URLs
                                table_lines.append(f"| Web Research | {url_links} |")
                                logger.info(f"[DEBUG] Added URL links to table")
                            else:
                                table_lines.append(f"| Web Research | No URLs available |")
                        else:
                            table_lines.append(f"| Web Research | No results |")
                    except Exception as e:
                        # Fallback: show truncated version
                        logger.error(f"[DEBUG] Exception parsing web data: {e}")
                        summary = str(web_data)[:100] + "..."
                        table_lines.append(f"| Web Research | {summary} |")

        elif intent == "finance-education":
            table_lines.append(f"## Educational Content Summary\n")
            table_lines.append("| Topic | Information |")
            table_lines.append("|-------|-------------|")

            query = state.get("user_query", "Finance Topic")[:50]
            table_lines.append(f"| Query | {query} |")

            if "ticker" in state:
                table_lines.append(f"| Related Ticker | {state['ticker']} |")

            if "Result" in state:
                result_summary = state["Result"][:100] + "..."
                table_lines.append(f"| Key Points | {result_summary} |")

        return "\n".join(table_lines) if table_lines else "No table data available"

    def generate_text_summary(self, state: Dict[str, Any], synthesis: Dict[str, Any]) -> str:
        """Generate a concise text summary adapted to intent"""

        intent = state.get("intent", "finance-company")
        summary_parts = []

        if intent == "finance-company":
            ticker = state.get("ticker", "UNKNOWN")
            timeframe = state.get("timeframe", "recent period")

            summary_parts.append(f"**{ticker} Investment Analysis ({timeframe})**\n")

            # Key metrics
            if "metric_extractor_result" in state:
                metrics = state["metric_extractor_result"]
                if "financials" in metrics:
                    fin = metrics["financials"]
                    if "revenue" in fin:
                        rev = fin["revenue"]
                        summary_parts.append(f"• Revenue: ${rev.get('actual', 'N/A')}M (YoY: {rev.get('yoy_change', 'N/A')}%)")
                    if "eps" in fin:
                        eps = fin["eps"]
                        summary_parts.append(f"• EPS: ${eps.get('actual', 'N/A')} (Beat by {eps.get('vs_estimate', 'N/A')}%)")

        elif intent == "finance-market":
            summary_parts.append(f"**Market Analysis**\n")

            # Use actual synthesis data if available
            exec_summary = synthesis.get("executive_summary", "")
            financial_analysis = synthesis.get("financial_analysis", "")
            news_sentiment = synthesis.get("news_sentiment_analysis", "")
            investment_outlook = synthesis.get("investment_outlook", "")
            recommendation = synthesis.get("recommendation", "")

            # Build comprehensive market summary from synthesis
            if exec_summary:
                summary_parts.append(f"\n{exec_summary}")

            if financial_analysis:
                summary_parts.append(f"\n**Market Analysis:**\n{financial_analysis}")

            if news_sentiment:
                summary_parts.append(f"\n**Market Sentiment:**\n{news_sentiment}")

            if investment_outlook:
                summary_parts.append(f"\n**Outlook:**\n{investment_outlook}")

            if recommendation:
                summary_parts.append(f"\n**Recommendation:** {recommendation}")

        elif intent == "finance-education":
            summary_parts.append(f"**Educational Summary**\n")
            summary_parts.append(f"Topic: {state.get('user_query', 'Finance Concept')[:100]}")

        # Add executive summary
        exec_summary = synthesis.get("executive_summary", "")
        if exec_summary:
            summary_parts.append(f"\n{exec_summary[:300]}...")

        # Add recommendation
        recommendation = synthesis.get("recommendation", "")
        if recommendation:
            summary_parts.append(f"\n**Recommendation:** {recommendation}")

        # Add price target if available
        if "price_target" in synthesis:
            summary_parts.append(f"**Price Target:** ${synthesis['price_target']}")

        return "\n".join(summary_parts)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for the Research Compiler Agent

        Args:
            state: Dictionary containing various fields based on intent:
                Common fields:
                - intent: Type of analysis (finance-company/market/education)
                - Query or user_query: User's question
                - output_format: List of desired formats

                Intent-specific fields:
                - finance-company: ticker, metric_extractor_result, news_scraper_result, etc.
                - finance-market: outputFromWebSearch, metrics, timeframe
                - finance-education: outputFromWebSearch, Result, optional ticker

        Returns:
            Dictionary with paths to generated files and summaries
        """

        # Validate and normalize input
        validation = self.validate_input(state)

        if not validation["is_valid"]:
            return {
                "error": "Invalid input",
                "errors": validation["errors"],
                "warnings": validation["warnings"]
            }

        # Log warnings if any
        for warning in validation["warnings"]:
            logger.warning(warning)

        # Use normalized state
        state = validation["normalized_state"]
        intent = state.get("intent", "finance-company")

        # Extract common information
        user_query = state.get("user_query", state.get("Query", "Analyze based on available data"))
        output_formats = state.get("output_format", ["text"])

        logger.info(f"Processing request with intent: {intent}")
        logger.info(f"Requested formats: {output_formats}")

        # Generate synthesis based on intent
        synthesis = self.synthesize_by_intent(state, intent)

        # Prepare results
        results = {
            "intent": intent,
            "user_query": user_query,
            "formats_generated": [],
            "metadata": {
                "intent": intent,
                "generated_at": datetime.now().isoformat()
            }
        }

        # Add intent-specific metadata
        if intent == "finance-company":
            results["ticker"] = state.get("ticker", "N/A")
            results["metadata"]["tickers_analyzed"] = [state.get("ticker")] if state.get("ticker") else []
            results["metadata"]["timeframe"] = state.get("timeframe", "N/A")
            if "Peers" in state:
                results["metadata"]["peers"] = state["Peers"]
        elif intent == "finance-market":
            results["metadata"]["timeframe"] = state.get("timeframe", "Current")
            results["metadata"]["metrics"] = state.get("metrics", [])
        elif intent == "finance-education":
            if state.get("ticker"):
                results["metadata"]["related_ticker"] = state["ticker"]

        # Generate requested output formats
        config = self.INTENT_CONFIGS[intent]

        # Generate each requested format if supported by intent
        if "pdf" in output_formats and "pdf" in config["output_types"]:
            try:
                pdf_path = self.generate_pdf_report(state, synthesis)
                results["pdf_path"] = pdf_path
                results["formats_generated"].append("pdf")
                logger.info(f"PDF report generated: {pdf_path}")
            except Exception as e:
                logger.error(f"Error generating PDF: {str(e)}")
                results["errors"] = results.get("errors", []) + [f"PDF generation failed: {str(e)}"]

        if "csv" in output_formats and "csv" in config["output_types"]:
            try:
                csv_path = self.generate_csv_export(state)
                if csv_path:
                    results["csv_path"] = csv_path
                    results["formats_generated"].append("csv")
                    logger.info(f"CSV export generated: {csv_path}")
            except Exception as e:
                logger.error(f"Error generating CSV: {str(e)}")

        if ("graph" in output_formats or "graphs" in output_formats) and "graph" in config["output_types"]:
            try:
                graph_paths = self.generate_graphs(state)
                if graph_paths:
                    results["graphs"] = graph_paths
                    results["formats_generated"].append("graphs")
                    logger.info(f"{len(graph_paths)} graphs generated")
            except Exception as e:
                logger.error(f"Error generating graphs: {str(e)}")

        if "table" in output_formats and "table" in config["output_types"]:
            try:
                table = self.generate_table(state)
                results["table"] = table
                results["formats_generated"].append("table")
                logger.info("Table generated")
            except Exception as e:
                logger.error(f"Error generating table: {str(e)}")

        # Always generate text summary as it's supported by all intents
        if "text" in output_formats or not output_formats:
            try:
                text_summary = self.generate_text_summary(state, synthesis)
                results["text_summary"] = text_summary
                results["formats_generated"].append("text")
                logger.info("Text summary generated")
            except Exception as e:
                logger.error(f"Error generating text summary: {str(e)}")

        # Add warnings to results if any
        if validation["warnings"]:
            results["warnings"] = validation["warnings"]

        return results


# Test the agent with different intents
if __name__ == "__main__":
    # Initialize agent
    print("Initializing Research Compiler Agent...")
    agent = ResearchCompilerAgent(model="gpt-4o")

    # Test 1: Company Analysis (finance-company)
    print("\n" + "="*50)
    print("TEST 1: Company Analysis Intent")
    print("="*50)

    company_state = {
        "Query": "Analyze Apple's Q2 2024 performance and provide investment recommendation",
        "intent": "finance-company",
        "ticker": "AAPL",
        "Peers": ["MSFT", "GOOGL"],
        "timeframe": "Q2 2024",
        "output_format": ["text", "table", "pdf"],
        "metric_extractor_result": {
            "ticker": "AAPL",
            "timeframe": "Q2 2024",
            "financials": {
                "revenue": {"actual": 117000, "yoy_change": 5, "vs_estimate": 2},
                "eps": {"actual": 1.40, "yoy_change": 8, "vs_estimate": 9}
            },
            "valuation": {
                "pe_ratio": 28.5,
                "market_cap": 3200
            }
        },
        "news_scraper_result": {
            "qualitative_summary": "Apple reported strong iPhone sales in Asia.",
            "quantitative_summary": "Revenue up 5% YoY, Services hit record $22B.",
            "insight_outlook": "Positive outlook with AI initiatives."
        }
    }

    try:
        results = agent.run(company_state)
        print(f"✅ Company Analysis Results:")
        print(f"   Intent: {results.get('intent')}")
        print(f"   Formats generated: {results.get('formats_generated')}")
        if "text_summary" in results:
            print(f"   Summary preview: {results['text_summary'][:150]}...")
    except Exception as e:
        print(f"❌ Error in company analysis: {str(e)}")

    # Test 2: Market Analysis (finance-market)
    print("\n" + "="*50)
    print("TEST 2: Market Analysis Intent")
    print("="*50)

    market_state = {
        "intent": "finance-market",
        "Query": "What are the current market trends in tech sector?",
        "timeframe": "last 4 quarters",
        "metrics": ["earnings", "insider sentiment"],
        "outputFromWebSearch": "Tech sector shows strong growth with AI driving valuations. Major companies reporting 15-20% YoY revenue growth. Cloud services expanding rapidly.",
        "output_format": ["text", "table"],
        "nextNode": "compiler_agent"
    }

    try:
        results = agent.run(market_state)
        print(f"✅ Market Analysis Results:")
        print(f"   Intent: {results.get('intent')}")
        print(f"   Formats generated: {results.get('formats_generated')}")
        if "table" in results:
            print(f"   Table preview:\n{results['table'][:200]}")
    except Exception as e:
        print(f"❌ Error in market analysis: {str(e)}")

    # Test 3: Educational Content (finance-education)
    print("\n" + "="*50)
    print("TEST 3: Educational Content Intent")
    print("="*50)

    education_state = {
        "intent": "finance-education",
        "Query": "Explain P/E ratio and its importance",
        "ticker": "TSLA",  # Optional
        "outputFromWebSearch": "P/E ratio (Price-to-Earnings) is a valuation metric comparing share price to earnings per share.",
        "Result": "P/E helps investors assess if a stock is overvalued or undervalued relative to earnings.",
        "output_format": ["text"],
        "nextNode": "compiler_agent"
    }

    try:
        results = agent.run(education_state)
        print(f"✅ Educational Content Results:")
        print(f"   Intent: {results.get('intent')}")
        print(f"   Formats generated: {results.get('formats_generated')}")
        if "text_summary" in results:
            print(f"   Educational summary:\n{results['text_summary']}")
    except Exception as e:
        print(f"❌ Error in educational content: {str(e)}")

    # Test 4: Handling unexpected input
    print("\n" + "="*50)
    print("TEST 4: Handling Unexpected Input")
    print("="*50)

    unexpected_state = {
        "some_field": "unexpected data",
        "Query": "Random query without proper structure"
    }

    try:
        results = agent.run(unexpected_state)
        print(f"✅ Handled unexpected input:")
        print(f"   Intent (defaulted): {results.get('intent')}")
        print(f"   Warnings: {results.get('warnings', [])}")
        print(f"   Formats generated: {results.get('formats_generated')}")
    except Exception as e:
        print(f"❌ Error handling unexpected input: {str(e)}")

    print("\n" + "="*50)
    print("✅ All tests completed!")
    print("="*50)