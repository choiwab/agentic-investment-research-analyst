"""
Research Compiler Agent

This agent synthesizes outputs from all upstream agents and compiles them into
user-requested formats (PDF reports, CSV files, graphs, tables, or text summaries).
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from io import BytesIO
import base64

# LangChain imports
from langchain_ollama import ChatOllama
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

    def __init__(self, model: str = "llama3.1", use_openai: bool = False) -> None:
        """
        Initialize the Research Compiler Agent

        Args:
            model: Model name (default: llama3.1 for Ollama, or gpt-4 for OpenAI)
            use_openai: Whether to use OpenAI instead of Ollama
        """
        self.callback_handler = PrintCallbackHandler()

        # Choose LLM based on preference
        if use_openai:
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.3,
                streaming=True,
                callbacks=[self.callback_handler]
            )
        else:
            self.llm = ChatOllama(
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
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        # Create output directories
        self.output_dir = "research_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/pdf", exist_ok=True)
        os.makedirs(f"{self.output_dir}/csv", exist_ok=True)
        os.makedirs(f"{self.output_dir}/graphs", exist_ok=True)

    def build_agent(self) -> AgentExecutor:
        """Build the REACT agent for research compilation"""

        format_instructions = self.parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

        system_template = f"""
        You are a senior equity research analyst responsible for compiling comprehensive investment research reports.
        Your task is to synthesize outputs from multiple analysis agents and create professional reports.

        You will receive:
        1. User query and desired output formats
        2. Preprocessing results (query categorization)
        3. News scraper analysis (qualitative/quantitative summaries, outlook)
        4. Metric extractor results (financial metrics, valuations)
        5. Sentiment analysis (if available)

        Your responsibilities:
        - Generate executive summaries that synthesize all findings
        - Create narrative sections connecting metrics, news, and sentiment
        - Identify key investment insights and thesis
        - Draft clear recommendations based on data
        - Maintain objectivity and accuracy (no hallucinated metrics)

        Output your synthesis as JSON:
        {format_instructions}

        Focus on:
        - Clarity and professionalism in all outputs
        - Data accuracy and proper attribution
        - Balanced analysis considering both positives and negatives
        - Actionable insights for investors
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
        def synthesize_news_sentiment(news_data: str) -> str:
            """Synthesize news and sentiment data into key themes"""
            try:
                data = json.loads(news_data) if isinstance(news_data, str) else news_data

                themes = []
                if "qualitative_summary" in data:
                    themes.append(f"Key Developments: {data['qualitative_summary'][:200]}...")
                if "quantitative_summary" in data:
                    themes.append(f"Key Metrics: {data['quantitative_summary'][:200]}...")
                if "insight_outlook" in data:
                    themes.append(f"Outlook: {data['insight_outlook'][:200]}...")

                return "\n".join(themes) if themes else "No news analysis available"

            except Exception as e:
                return f"Error synthesizing news: {str(e)}"

        @tool
        def generate_investment_thesis(combined_data: str) -> str:
            """Generate investment thesis based on all available data"""
            try:
                data = json.loads(combined_data) if isinstance(combined_data, str) else combined_data

                thesis_points = []

                # Analyze financial strength
                if "metric_extractor_result" in data:
                    metrics = data["metric_extractor_result"]
                    if metrics.get("financials", {}).get("revenue", {}).get("yoy_change", 0) > 5:
                        thesis_points.append("Strong revenue growth momentum")
                    if metrics.get("valuation", {}).get("pe_ratio", 100) < 25:
                        thesis_points.append("Attractive valuation relative to growth")

                # Analyze sentiment
                if "news_scraper_result" in data:
                    news = data["news_scraper_result"]
                    if "positive" in news.get("insight_outlook", "").lower():
                        thesis_points.append("Positive market sentiment and outlook")

                # Generate recommendation
                if len(thesis_points) >= 2:
                    recommendation = "BUY - Multiple positive catalysts"
                elif len(thesis_points) == 1:
                    recommendation = "HOLD - Mixed signals"
                else:
                    recommendation = "HOLD - Insufficient positive catalysts"

                thesis = f"Investment Thesis:\n"
                thesis += "\n".join(f"• {point}" for point in thesis_points)
                thesis += f"\n\nRecommendation: {recommendation}"

                return thesis

            except Exception as e:
                return f"Error generating thesis: {str(e)}"

        return [analyze_financial_metrics, synthesize_news_sentiment, generate_investment_thesis]

    def generate_pdf_report(self, state: Dict[str, Any], synthesis: Dict[str, Any]) -> str:
        """Generate a professional PDF research report"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ticker = state.get("ticker", "UNKNOWN")
        pdf_path = f"{self.output_dir}/pdf/{ticker}_research_{timestamp}.pdf"

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

        # Title Page
        story.append(Paragraph(f"Investment Research Report", title_style))
        story.append(Paragraph(f"{ticker}", title_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        story.append(PageBreak())

        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        exec_summary = synthesis.get("executive_summary", "No executive summary available.")
        story.append(Paragraph(exec_summary, styles['BodyText']))
        story.append(Spacer(1, 0.3*inch))

        # Financial Analysis
        story.append(Paragraph("Financial Analysis", heading_style))

        # Create financial metrics table if available
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

        financial_analysis = synthesis.get("financial_analysis", "No financial analysis available.")
        story.append(Paragraph(financial_analysis, styles['BodyText']))
        story.append(Spacer(1, 0.3*inch))

        # News & Sentiment Analysis
        story.append(Paragraph("News & Sentiment Analysis", heading_style))
        news_analysis = synthesis.get("news_sentiment_analysis", "No news analysis available.")
        story.append(Paragraph(news_analysis, styles['BodyText']))
        story.append(Spacer(1, 0.3*inch))

        # Investment Outlook
        story.append(Paragraph("Investment Outlook", heading_style))
        outlook = synthesis.get("investment_outlook", "No outlook available.")
        story.append(Paragraph(outlook, styles['BodyText']))
        story.append(Spacer(1, 0.3*inch))

        # Recommendation
        story.append(Paragraph("Recommendation", heading_style))
        recommendation = synthesis.get("recommendation", "No recommendation available.")
        story.append(Paragraph(recommendation, styles['BodyText']))

        # Add graphs if they exist
        graph_paths = self.generate_graphs(state)
        if graph_paths:
            story.append(PageBreak())
            story.append(Paragraph("Charts & Visualizations", heading_style))
            for graph_path in graph_paths:
                if os.path.exists(graph_path):
                    img = Image(graph_path, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.2*inch))

        # Build PDF
        doc.build(story)

        return pdf_path

    def generate_csv_export(self, state: Dict[str, Any]) -> str:
        """Export financial data to CSV format"""

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

        # Create DataFrame and save to CSV
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        return csv_path

    def generate_graphs(self, state: Dict[str, Any]) -> List[str]:
        """Generate matplotlib graphs for visualization"""

        graph_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ticker = state.get("ticker", "UNKNOWN")

        # Check if we have metric data to plot
        if "metric_extractor_result" not in state:
            return graph_paths

        metrics = state["metric_extractor_result"]

        # 1. Financial Metrics Comparison Chart
        if "financials" in metrics:
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

        # 2. Valuation Metrics Chart
        if "valuation" in metrics:
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

        return graph_paths

    def generate_table(self, state: Dict[str, Any]) -> str:
        """Generate a formatted markdown table"""

        ticker = state.get("ticker", "UNKNOWN")
        table_lines = []

        # Header
        table_lines.append(f"## {ticker} - Financial Summary Table\n")
        table_lines.append("| Metric | Value | YoY Change | vs Estimate |")
        table_lines.append("|--------|-------|------------|-------------|")

        # Add data rows
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
                            yoy_str = f"=� {yoy_str}"
                        elif yoy != "N/A" and yoy < 0:
                            yoy_str = f"=4 {yoy_str}"

                        table_lines.append(f"| {metric_name} | {actual} | {yoy_str} | {vs_est_str} |")

            if "valuation" in metrics:
                table_lines.append(f"| **Valuation** | | | |")
                for key, value in metrics["valuation"].items():
                    metric_name = key.replace("_", " ").upper()
                    table_lines.append(f"| {metric_name} | {value} | - | - |")

        return "\n".join(table_lines)

    def generate_text_summary(self, state: Dict[str, Any], synthesis: Dict[str, Any]) -> str:
        """Generate a concise text summary for chatbot response"""

        ticker = state.get("ticker", "UNKNOWN")
        timeframe = state.get("timeframe", "recent period")

        summary_parts = []

        # Opening statement
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

        # Executive summary
        exec_summary = synthesis.get("executive_summary", "")
        if exec_summary:
            summary_parts.append(f"\n{exec_summary[:300]}...")

        # Recommendation
        recommendation = synthesis.get("recommendation", "HOLD")
        summary_parts.append(f"\n**Recommendation:** {recommendation}")

        # Price target if available
        if "price_target" in synthesis:
            summary_parts.append(f"**Price Target:** ${synthesis['price_target']}")

        return "\n".join(summary_parts)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for the Research Compiler Agent

        Args:
            state: Dictionary containing:
                - user_query: Original user request
                - output_format: List of desired formats ["pdf", "csv", "graph", "table", "text"]
                - preprocessing_result: Query categorization
                - news_scraper_result: News analysis
                - metric_extractor_result: Financial metrics
                - sentiment_extractor_result: Sentiment analysis (optional)
                - ticker: Stock ticker being analyzed
                - timeframe: Analysis timeframe

        Returns:
            Dictionary with paths to generated files and summaries
        """

        # Extract required information
        user_query = state.get("user_query", "Analyze the company")
        output_formats = state.get("output_format", ["text"])
        ticker = state.get("ticker", "UNKNOWN")

        # Prepare context for synthesis
        context = {
            "user_query": user_query,
            "ticker": ticker,
            "has_metrics": "metric_extractor_result" in state,
            "has_news": "news_scraper_result" in state,
            "has_sentiment": "sentiment_extractor_result" in state
        }

        # Use agent to synthesize information
        synthesis_prompt = f"""
        Analyze the following investment data for {ticker} and provide a comprehensive synthesis:

        User Query: {user_query}

        Available Data:
        - Metrics: {json.dumps(state.get('metric_extractor_result', {}))[:500]}
        - News: {json.dumps(state.get('news_scraper_result', {}))[:500]}
        - Sentiment: {json.dumps(state.get('sentiment_extractor_result', {}))[:500]}

        Please provide:
        1. Executive summary (2-3 sentences)
        2. Financial analysis (key metrics and trends)
        3. News and sentiment analysis
        4. Investment outlook
        5. Clear recommendation (BUY/HOLD/SELL) with rationale
        """

        # Get synthesis from agent
        agent_result = self.agent.invoke({"input": synthesis_prompt})

        # Parse the output
        if isinstance(agent_result['output'], dict):
            synthesis = agent_result['output']
        else:
            try:
                synthesis = self.parser.parse(agent_result['output'])
            except:
                # Fallback to basic synthesis
                synthesis = {
                    "executive_summary": "Analysis complete for " + ticker,
                    "financial_analysis": "See detailed metrics below.",
                    "news_sentiment_analysis": "Market sentiment analyzed.",
                    "investment_outlook": "Outlook based on available data.",
                    "recommendation": "HOLD - Awaiting further analysis"
                }

        # Generate requested output formats
        results = {
            "user_query": user_query,
            "ticker": ticker,
            "formats_generated": [],
            "metadata": {
                "tickers_analyzed": [ticker],
                "timeframe": state.get("timeframe", "N/A"),
                "generated_at": datetime.now().isoformat()
            }
        }

        # Generate each requested format
        if "pdf" in output_formats:
            try:
                pdf_path = self.generate_pdf_report(state, synthesis)
                results["pdf_path"] = pdf_path
                results["formats_generated"].append("pdf")
                print(f" PDF report generated: {pdf_path}")
            except Exception as e:
                print(f"L Error generating PDF: {str(e)}")

        if "csv" in output_formats:
            try:
                csv_path = self.generate_csv_export(state)
                results["csv_path"] = csv_path
                results["formats_generated"].append("csv")
                print(f" CSV export generated: {csv_path}")
            except Exception as e:
                print(f"L Error generating CSV: {str(e)}")

        if "graph" in output_formats or "graphs" in output_formats:
            try:
                graph_paths = self.generate_graphs(state)
                results["graphs"] = graph_paths
                results["formats_generated"].append("graphs")
                print(f" {len(graph_paths)} graphs generated")
            except Exception as e:
                print(f"L Error generating graphs: {str(e)}")

        if "table" in output_formats:
            try:
                table = self.generate_table(state)
                results["table"] = table
                results["formats_generated"].append("table")
                print(f" Table generated")
            except Exception as e:
                print(f"L Error generating table: {str(e)}")

        if "text" in output_formats or not output_formats:
            try:
                text_summary = self.generate_text_summary(state, synthesis)
                results["text_summary"] = text_summary
                results["formats_generated"].append("text")
                print(f" Text summary generated")
            except Exception as e:
                print(f"L Error generating text summary: {str(e)}")

        return results


# Test the agent
if __name__ == "__main__":
    # Initialize agent
    print("Initializing Research Compiler Agent...")
    agent = ResearchCompilerAgent(model="llama3.1", use_openai=False)

    # Create mock state with sample data
    mock_state = {
        "user_query": "Analyze Apple's Q2 2024 performance and provide investment recommendation",
        "output_format": ["pdf", "csv", "graph", "table", "text"],
        "ticker": "AAPL",
        "timeframe": "Q2 2024",
        "preprocessing_result": {
            "category": "finance-company",
            "confidence": 0.95
        },
        "news_scraper_result": {
            "qualitative_summary": "Apple reported strong iPhone sales in Asia and continued growth in Services revenue. The company announced new AI features for upcoming products.",
            "quantitative_summary": "Revenue reached $117B, up 5% YoY. iPhone revenue grew 7% to $51B. Services revenue hit record $22B.",
            "insight_outlook": "Positive outlook driven by AI initiatives and strong ecosystem lock-in. China market remains a concern but showing recovery signs."
        },
        "metric_extractor_result": {
            "ticker": "AAPL",
            "timeframe": "Q2 2024",
            "financials": {
                "revenue": {"actual": 117000, "yoy_change": 5, "vs_estimate": 2},
                "eps": {"actual": 1.40, "yoy_change": 8, "vs_estimate": 9},
                "operating_margin": {"actual": 0.31, "yoy_change": -1}
            },
            "valuation": {
                "pe_ratio": 28.5,
                "market_cap": 3200,
                "price_to_book": 49.2,
                "ev_to_ebitda": 23.1
            }
        },
        "sentiment_extractor_result": {
            "overall_sentiment": "positive",
            "sentiment_score": 0.72,
            "key_themes": ["AI innovation", "Services growth", "China recovery"]
        }
    }

    print("\nRunning Research Compiler Agent with mock data...")
    print(f"Analyzing: {mock_state['ticker']}")
    print(f"Requested formats: {mock_state['output_format']}")

    # Run the agent
    try:
        results = agent.run(mock_state)

        print("\n" + "="*50)
        print("RESULTS:")
        print("="*50)
        print(f" Formats generated: {results['formats_generated']}")

        if "pdf_path" in results:
            print(f"=� PDF Report: {results['pdf_path']}")

        if "csv_path" in results:
            print(f"=� CSV Export: {results['csv_path']}")

        if "graphs" in results:
            print(f"=� Graphs: {results['graphs']}")

        if "table" in results:
            print(f"\n=� Table Output:")
            print(results["table"])

        if "text_summary" in results:
            print(f"\n=� Text Summary:")
            print(results["text_summary"])

        print(f"\n=P Generated at: {results['metadata']['generated_at']}")

    except Exception as e:
        print(f"L Error running agent: {str(e)}")
        import traceback
        traceback.print_exc()