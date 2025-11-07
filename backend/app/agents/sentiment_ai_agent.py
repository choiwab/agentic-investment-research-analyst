import json
import re
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

import numpy as np
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from news_scraper import NewsScraperAgent
from utils.callback_handler import PrintCallbackHandler
from utils.model_schema import SentimentModel
from utils.sec_risk_analyzer import SECRiskAnalyzer

load_dotenv(override=True)

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class AnalysisStage(Enum):
    ENTITY_EXTRACTION = "entity_extraction"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    FINANCIAL_METRICS = "financial_metrics"
    SYNTHESIS = "synthesis"


class AgentState(TypedDict):
    text: str
    ticker: Optional[str]
    entities: Optional[List[str]]
    
    messages: Annotated[Sequence[BaseMessage], "append"]
    current_stage: str
    
    extracted_entities: Optional[Dict[str, Any]]
    sentiment_results: Optional[Dict[str, Any]]
    risk_results: Optional[Dict[str, Any]]
    financial_results: Optional[Dict[str, Any]]
    
    final_analysis: Optional[Dict[str, Any]]
    errors: List[str]
    metadata: Dict[str, Any]


class SentimentAgent:
    
    def __init__(self, model: str = "gpt-5-nano"):
        self.callback_handler = PrintCallbackHandler()
        self.llm = ChatOpenAI(
            model=model, 
            temperature=0, 
            streaming=True, 
            callbacks=[self.callback_handler]
        )
        try:
            _ = self.llm.invoke("ping")
            print("[SentimentAgent] OpenAI LLM reachable.")
        except Exception as e:
            print(f"[SentimentAgent] OpenAI not reachable: {e}")
        
        # FinBERT removed - using OpenAI for sentiment analysis
        self.risk_analyzer = SECRiskAnalyzer()
        
        self.parser = StructuredOutputParser.from_response_schemas(SentimentModel.response_schema)
        
        self.workflow = self._build_workflow()
        
        self.memory_saver = MemorySaver()
        
        self.app = self.workflow.compile(checkpointer=self.memory_saver)

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        workflow.add_node("entity_extraction", self._entity_extraction_node)
        workflow.add_node("sentiment_analysis", self._sentiment_analysis_node)
        workflow.add_node("risk_assessment", self._risk_assessment_node)
        workflow.add_node("financial_metrics", self._financial_metrics_node)
        workflow.add_node("synthesis", self._synthesis_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        workflow.set_entry_point("entity_extraction")
        
        workflow.add_conditional_edges(
            "entity_extraction",
            self._route_after_extraction,
            {
                "sentiment": "sentiment_analysis",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "sentiment_analysis",
            self._route_after_sentiment,
            {
                "risk": "risk_assessment",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "risk_assessment",
            self._route_after_risk,
            {
                "financial": "financial_metrics",
                "synthesis": "synthesis",
                "error": "error_handler"
            }
        )
        
        workflow.add_edge("financial_metrics", "synthesis")
        workflow.add_edge("synthesis", END)
        workflow.add_edge("error_handler", END)
        
        return workflow

    def _entity_extraction_node(self, state: AgentState) -> Dict[str, Any]:
        try:
            text = state["text"]
            
            if state.get("entities"):
                extracted = {
                    "provided_entities": state["entities"],
                    "tickers": [],
                    "companies": []
                }
            else:
                extracted = self._extract_entities(text)
            
            if state.get("ticker"):
                extracted["primary_ticker"] = state["ticker"]
            
            return {
                "extracted_entities": extracted,
                "current_stage": AnalysisStage.ENTITY_EXTRACTION.value,
                "messages": [AIMessage(content=f"Extracted entities: {json.dumps(extracted, indent=2)}")]
            }
            
        except Exception as e:
            return {
                "errors": [f"Entity extraction failed: {str(e)}"],
                "current_stage": "error"
            }

    def _sentiment_analysis_node(self, state: AgentState) -> Dict[str, Any]:
        try:
            text = state["text"]
            result = self._analyze_sentiment_with_openai(text)
            if not result:
                raise ValueError("Sentiment analysis returned no results")

            sentiment_data = {
                "label": result.get("sentiment_label", "neutral"),
                "confidence": float(result.get("confidence_score", 0.0)),
                "sentiment_score": float(result.get("sentiment_score", 0.0)),
                "probabilities": {
                    "positive": float(result.get("positive_prob", 0.0)),
                    "negative": float(result.get("negative_prob", 0.0)),
                    "neutral":  float(result.get("neutral_prob", 0.0)),
                },
                "dominant_sentiment": result.get("sentiment_label", "neutral"),
                "sentiment_context": self._analyze_sentiment_context(text, result),
            }

            # Prefer analyzer-provided evidence; otherwise fall back to sentence-level evidence
            if "evidence" in result and result["evidence"]:
                sentiment_data["evidence"] = result["evidence"]
            else:
                ev = self._collect_sentiment_evidence_openai(
                    text, sentiment_data["dominant_sentiment"], top_k=5
                )
                if ev:
                    sentiment_data["evidence"] = ev

            if "uncertainty_entropy" in result:
                sentiment_data["uncertainty_entropy"] = float(result["uncertainty_entropy"])

            return {
                "sentiment_results": convert_numpy_types(sentiment_data),
                "current_stage": AnalysisStage.SENTIMENT_ANALYSIS.value,
                "messages": [
                    AIMessage(
                        content=f"Sentiment analysis complete: "
                                f"{sentiment_data['label']} "
                                f"({sentiment_data['confidence']:.2f} confidence)"
                    )
                ],
            }

        except Exception as e:
            return {
                "errors": [f"Sentiment analysis failed: {str(e)}"],
                "current_stage": "error",
            }

    def _risk_assessment_node(self, state: AgentState) -> Dict[str, Any]:
        try:
            text = state["text"]
            ticker = state.get("ticker") or state.get("extracted_entities", {}).get("primary_ticker")
            risk_result = self.risk_analyzer.analyze_comprehensive_risk(text, ticker=ticker)
            
            risk_data = {
                "risk_scores": risk_result["risk_scores"],
                "risk_factors_count": {
                    category: len(factors) 
                    for category, factors in risk_result["risk_factors"].items()
                },
                "detailed_risk_factors": risk_result["risk_factors"],
                "overall_risk_level": self._categorize_risk_level(risk_result["risk_scores"].get("overall", 0)),
                "key_risk_categories": self._identify_key_risks(risk_result["risk_factors"]),
                "risk_mitigation_suggestions": self._generate_mitigation_suggestions(risk_result)
            }
            
            return {
                "risk_results": convert_numpy_types(risk_data),
                "current_stage": AnalysisStage.RISK_ASSESSMENT.value,
                "messages": [AIMessage(content=f"Risk assessment complete: {risk_data['overall_risk_level']}")]
            }
            
        except Exception as e:
            return {
                "errors": [f"Risk assessment failed: {str(e)}"],
                "current_stage": "error"
            }

    def _financial_metrics_node(self, state: AgentState) -> Dict[str, Any]:
        try:
            ticker = state.get("ticker")
            if not ticker and state.get("extracted_entities"):
                ticker = state["extracted_entities"].get("primary_ticker")
            
            if not ticker:
                return {
                    "financial_results": {"status": "No ticker available for financial analysis"},
                    "current_stage": AnalysisStage.FINANCIAL_METRICS.value
                }
            
            metrics = self.risk_analyzer.get_financial_metrics(ticker)
            
            if not metrics:
                return {
                    "financial_results": {"status": f"No financial data available for {ticker}"},
                    "current_stage": AnalysisStage.FINANCIAL_METRICS.value
                }
            
            financial_data = {
                "ticker": ticker,
                "raw_metrics": metrics,
                "interpretation": self._interpret_financial_metrics(metrics),
                "financial_health_score": self._calculate_financial_health_score(metrics),
                "recommendations": self._generate_financial_recommendations(metrics, state.get("sentiment_results"), state.get("risk_results"))
            }
            
            return {
                "financial_results": convert_numpy_types(financial_data),
                "current_stage": AnalysisStage.FINANCIAL_METRICS.value,
                "messages": [AIMessage(content=f"Financial metrics analyzed for {ticker}")]
            }
            
        except Exception as e:
            return {
                "errors": [f"Financial metrics retrieval failed: {str(e)}"],
                "current_stage": "error"
            }

    def _synthesis_node(self, state: AgentState) -> Dict[str, Any]:
        try:
            synthesis_prompt = self._create_synthesis_prompt(state)
            response = self.llm.invoke(synthesis_prompt)

            parsed_structured = {}
            if hasattr(self, "parser") and self.parser:
                try:
                    parsed_structured = self.parser.parse(response.content)
                except Exception:
                    try:
                        retry_msg = f"""Your previous output did not contain valid JSON matching the schema.
    Return ONLY a valid JSON object per these instructions:
    {self.parser.get_format_instructions()}"""
                        retry_resp = self.llm.invoke(retry_msg)
                        parsed_structured = self.parser.parse(retry_resp.content)
                        response = retry_resp
                    except Exception:
                        parsed_structured = {}

            final_analysis = {
                "executive_summary": self._generate_executive_summary(state),
                "sentiment_analysis": state.get("sentiment_results", {}),
                "risk_assessment": state.get("risk_results", {}),
                "financial_metrics": state.get("financial_results", {}),
                "entities": state.get("extracted_entities", {}),
                "key_insights": self._extract_key_insights(state),
                "recommendations": self._generate_comprehensive_recommendations(state),
                "confidence_metrics": self._calculate_confidence_metrics(state)
                # Uncomment if you want the full machine-readable content (for future work)
                # "llm_synthesis": response.content,
                # "llm_synthesis_structured": parsed_structured,
            }

            final_analysis["metadata"] = {
                "ticker": state.get("ticker"),
                "text_length": len(state["text"]),
                "entities_count": len(state.get("extracted_entities", {}).get("all_entities", [])),
                "analysis_timestamp": datetime.now().isoformat(),
                "stages_completed": [
                    "entity_extraction",
                    "sentiment_analysis",
                    "risk_assessment",
                    "financial_metrics" if state.get("financial_results") else None,
                    "synthesis"
                ]
            }

            return {
                "final_analysis": convert_numpy_types(final_analysis),
                "current_stage": AnalysisStage.SYNTHESIS.value,
                "messages": [AIMessage(content="Analysis synthesis complete (narrative + structured)")],
            }

        except Exception as e:
            return {
                "errors": [f"Synthesis failed: {str(e)}"],
                "current_stage": "error"
            }

    def _error_handler_node(self, state: AgentState) -> Dict[str, Any]:
        error_summary = {
            "status": "error",
            "errors": state.get("errors", ["Unknown error occurred"]),
            "partial_results": {
                "entities": state.get("extracted_entities"),
                "sentiment": state.get("sentiment_results"),
                "risk": state.get("risk_results"),
                "financial": state.get("financial_results")
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "last_successful_stage": state.get("current_stage", "unknown")
            }
        }
        
        return {"final_analysis": error_summary}

    def _route_after_extraction(self, state: AgentState) -> str:
        if state.get("errors"):
            return "error"
        return "sentiment"

    def _route_after_sentiment(self, state: AgentState) -> str:
        if state.get("errors"):
            return "error"
        return "risk"

    def _route_after_risk(self, state: AgentState) -> str:
        if state.get("errors"):
            return "error"
        if state.get("ticker") or (state.get("extracted_entities", {}).get("primary_ticker")):
            return "financial"
        return "synthesis"

    def _extract_entities(self, text: str) -> Dict[str, Any]:
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        company_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|Corporation|Company|Ltd|Limited|LLC|Co)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Technologies|Systems|Solutions|Group|Holdings)\b'
        ]
        
        tickers = list(set(re.findall(ticker_pattern, text)))
        companies = []
        
        for pattern in company_patterns:
            companies.extend(re.findall(pattern, text))
        
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD'}
        tickers = [t for t in tickers if t not in common_words and len(t) >= 2]
        
        return {
            "tickers": tickers[:10],
            "companies": list(set(companies))[:10],
            "all_entities": tickers[:10] + list(set(companies))[:10]
        }

    def _analyze_sentiment_context(self, text: str, sentiment_result: Dict) -> Dict[str, Any]:
        positive_indicators = ['growth', 'profit', 'success', 'increase', 'improve', 'strong', 'beat', 'exceed']
        negative_indicators = ['loss', 'decline', 'risk', 'challenge', 'concern', 'weak', 'miss', 'below']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_indicators if word in text_lower)
        negative_count = sum(1 for word in negative_indicators if word in text_lower)
        
        return {
            "positive_indicators_found": positive_count,
            "negative_indicators_found": negative_count,
            "sentiment_consistency": "consistent" if (
                (sentiment_result["sentiment_label"] == "positive" and positive_count > negative_count) or
                (sentiment_result["sentiment_label"] == "negative" and negative_count > positive_count) or
                (sentiment_result["sentiment_label"] == "neutral" and abs(positive_count - negative_count) <= 2)
            ) else "mixed"
        }

    def _categorize_risk_level(self, risk_score: float) -> str:
        if risk_score < 0.2:
            return "Low Risk"
        elif risk_score < 0.4:
            return "Moderate Risk"
        elif risk_score < 0.6:
            return "Medium-High Risk"
        elif risk_score < 0.8:
            return "High Risk"
        else:
            return "Very High Risk"

    def _identify_key_risks(self, risk_factors: Dict[str, List[str]]) -> List[str]:
        risk_counts = {category: len(factors) for category, factors in risk_factors.items()}
        sorted_risks = sorted(risk_counts.items(), key=lambda x: x[1], reverse=True)
        return [category for category, count in sorted_risks[:3] if count > 0]

    def _generate_mitigation_suggestions(self, risk_result: Dict) -> List[str]:
        suggestions = []
        risk_scores = risk_result.get("risk_scores", {})

        market = risk_scores.get("market_risk", 0.0)
        operational = risk_scores.get("operational_risk", 0.0)
        financial = risk_scores.get("financial_risk", 0.0)
        regulatory = risk_scores.get("regulatory_risk", 0.0)
        strategic = risk_scores.get("strategic_risk", 0.0)
        
        if market > 0.6:
            suggestions.append("Consider diversification strategies to reduce market risk exposure")
        if operational > 0.6:
            suggestions.append("Review and strengthen operational processes and contingency plans")
        if financial > 0.6:
            suggestions.append("Focus on improving financial metrics and liquidity management")
        if regulatory > 0.6:
            suggestions.append("Enhance regulatory compliance monitoring and reporting")
        if strategic > 0.6:
            suggestions.append("Stress-test strategic initiatives and address customer concentration")
        
        return suggestions

    def _interpret_financial_metrics(self, metrics: Dict) -> Dict[str, str]:
        interpretation = {}
        
        if 'debt_to_equity' in metrics:
            de_ratio = metrics['debt_to_equity']
            if de_ratio > 2.0:
                interpretation['debt_to_equity'] = "High leverage - significant financial risk"
            elif de_ratio > 1.0:
                interpretation['debt_to_equity'] = "Moderate leverage - manageable debt levels"
            else:
                interpretation['debt_to_equity'] = "Low leverage - conservative debt structure"
        
        if 'current_ratio' in metrics:
            cr = metrics['current_ratio']
            if cr < 1.0:
                interpretation['current_ratio'] = "Liquidity concerns - may struggle with short-term obligations"
            elif cr < 2.0:
                interpretation['current_ratio'] = "Adequate liquidity - reasonable short-term financial health"
            else:
                interpretation['current_ratio'] = "Strong liquidity - excellent short-term financial position"
        
        if 'profit_margin' in metrics:
            margin = metrics['profit_margin']
            if margin < 0:
                interpretation['profit_margin'] = "Negative margins - operating at a loss"
            elif margin < 0.05:
                interpretation['profit_margin'] = "Low margins - thin profitability"
            elif margin < 0.15:
                interpretation['profit_margin'] = "Moderate margins - reasonable profitability"
            else:
                interpretation['profit_margin'] = "Strong margins - excellent profitability"
        
        return interpretation

    def _calculate_financial_health_score(self, metrics: Dict) -> float:
        score = 0.5
        
        if metrics.get('current_ratio', 0) > 1.5:
            score += 0.1
        elif metrics.get('current_ratio', 0) < 1.0:
            score -= 0.15
        
        if metrics.get('debt_to_equity', float('inf')) < 1.0:
            score += 0.1
        elif metrics.get('debt_to_equity', 0) > 2.0:
            score -= 0.15
        
        if metrics.get('profit_margin', 0) > 0.1:
            score += 0.15
        elif metrics.get('profit_margin', 0) < 0:
            score -= 0.2
        
        if metrics.get('return_on_equity', 0) > 0.15:
            score += 0.15
        elif metrics.get('return_on_equity', 0) < 0.05:
            score -= 0.1
        
        return float(max(0, min(1, score)))

    def _generate_financial_recommendations(self, metrics: Dict, sentiment: Dict, risk: Dict) -> List[str]:
        recommendations = []
        
        if metrics.get('debt_to_equity', 0) > 1.5:
            recommendations.append("Consider debt reduction strategies to improve leverage ratios")
        
        if metrics.get('current_ratio', 0) < 1.2:
            recommendations.append("Focus on improving working capital management")
        
        if sentiment and sentiment.get('label') == 'negative' and metrics.get('profit_margin', 1) < 0.05:
            recommendations.append("Address profitability concerns highlighted by both sentiment and margins")
        
        if risk and risk.get('overall_risk_level') == 'High Risk' and metrics.get('debt_to_equity', 0) > 1.0:
            recommendations.append("High risk combined with leverage suggests need for conservative financial strategy")
        
        return recommendations

    def _create_synthesis_prompt(self, state: AgentState) -> str:
        prompt = f"""
        Based on the comprehensive analysis performed, provide a professional investment analysis summary:
        
        Sentiment Analysis Results:
        {json.dumps(state.get('sentiment_results', {}), indent=2)}
        
        Risk Assessment Results:
        {json.dumps(state.get('risk_results', {}), indent=2)}
        
        Financial Metrics Results:
        {json.dumps(state.get('financial_results', {}), indent=2)}
        
        Please provide:
        1. A cohesive narrative connecting sentiment, risk, and financial metrics
        2. Key investment implications
        3. Critical factors to monitor going forward
        4. Overall investment recommendation (Buy/Hold/Sell perspective)
        
        Keep the summary concise but comprehensive, focusing on actionable insights.
        """
        # Get machine readable JSON block for future work
        try: 
            format_instructions = self.parser.get_format_instructions()
            prompt += f"""

        ---
        Additionally, return a JSON object that follows these instructions.
        Do not include any prose before or after the JSON object.

        {format_instructions}
        """
        except Exception:
            pass

        return prompt

    def _generate_executive_summary(self, state: AgentState) -> str:
        sentiment = state.get('sentiment_results', {})
        risk = state.get('risk_results', {})
        financial = state.get('financial_results', {})
        
        summary_parts = []
        
        if sentiment:
            summary_parts.append(
                f"Sentiment analysis reveals {sentiment.get('label', 'unknown')} outlook "
                f"with {sentiment.get('confidence', 0):.1%} confidence."
            )
        
        if risk:
            summary_parts.append(
                f"Risk assessment indicates {risk.get('overall_risk_level', 'unknown')} "
                f"with key concerns in {', '.join(risk.get('key_risk_categories', [])[:2]) or 'multiple areas'}."
            )
        
        if financial and financial.get('raw_metrics'):
            health_score = financial.get('financial_health_score', 0.5)
            summary_parts.append(
                f"Financial health score of {health_score:.1%} suggests "
                f"{'strong' if health_score > 0.7 else 'moderate' if health_score > 0.4 else 'weak'} financial position."
            )
        
        return " ".join(summary_parts) if summary_parts else "Analysis complete with mixed signals requiring further investigation."

    def _extract_key_insights(self, state: AgentState) -> List[str]:
        insights = []
        
        sentiment = state.get('sentiment_results', {})
        risk = state.get('risk_results', {})
        financial = state.get('financial_results', {})
        
        if sentiment and sentiment.get('confidence', 0) > 0.7:
            insights.append(
                f"Strong {sentiment.get('label')} sentiment signal with high confidence"
            )
        
        if risk and risk.get('risk_scores', {}).get('overall', 0) > 0.7:
            insights.append(
                "Elevated risk levels require careful monitoring"
            )
        
        if financial and financial.get('raw_metrics'):
            metrics = financial.get('raw_metrics', {})
            if metrics.get('profit_margin', 0) > 0.15:
                insights.append("Strong profitability margins indicate operational efficiency")
            if metrics.get('current_ratio', 0) < 1.0:
                insights.append("Liquidity concerns may impact short-term operations")
        
        if (sentiment and sentiment.get('label') == 'positive' and 
            risk and risk.get('overall_risk_level') in ['Low Risk', 'Moderate Risk']):
            insights.append("Positive sentiment aligned with manageable risk profile")
        
        return insights[:5]

    def _generate_comprehensive_recommendations(self, state: AgentState) -> List[str]:
        recommendations = []
        
        sentiment = state.get('sentiment_results', {})
        risk = state.get('risk_results', {})
        financial = state.get('financial_results', {})
        
        sentiment_score = sentiment.get('sentiment_score', 0) if sentiment else 0
        risk_score = risk.get('risk_scores', {}).get('overall', 0.5) if risk else 0.5
        financial_health = financial.get('financial_health_score', 0.5) if financial else 0.5
        
        composite_score = (sentiment_score * 0.3 + (1 - risk_score) * 0.3 + financial_health * 0.4)
        
        if composite_score > 0.65:
            recommendations.append("Consider increasing position based on favorable analysis")
        elif composite_score > 0.35:
            recommendations.append("Maintain current position with continued monitoring")
        else:
            recommendations.append("Consider reducing exposure given current risk-reward profile")
        
        if risk and risk.get('risk_mitigation_suggestions'):
            recommendations.extend(risk['risk_mitigation_suggestions'][:2])
        
        if financial and financial.get('recommendations'):
            recommendations.extend(financial['recommendations'][:2])
        
        return recommendations[:5]

    def _calculate_confidence_metrics(self, state: AgentState) -> Dict[str, float]:
        metrics = {}
        
        completeness_score = 0
        if state.get('sentiment_results'):
            completeness_score += 0.25
        if state.get('risk_results'):
            completeness_score += 0.25
        if state.get('financial_results'):
            completeness_score += 0.25
        if state.get('extracted_entities'):
            completeness_score += 0.25
        
        metrics['data_completeness'] = completeness_score
        
        if state.get('sentiment_results'):
            metrics['sentiment_confidence'] = state['sentiment_results'].get('confidence', 0)
        
        confidence_components = []
        if 'sentiment_confidence' in metrics:
            confidence_components.append(metrics['sentiment_confidence'])
        confidence_components.append(completeness_score)
        
        metrics['overall_confidence'] = float(np.mean(confidence_components)) if confidence_components else 0.0
        
        return convert_numpy_types(metrics)
    
    def _build_text_from_news(self, news: Dict[str, str]) -> str:
        #For taking in output from NewsScraperAgent
        
        sections = []
        if news.get("qualitative_summary"):
            sections.append(f"Qualitative Summary:\n{news['qualitative_summary']}")
        if news.get("quantitative_summary"):
            sections.append(f"Quantitative Summary:\n{news['quantitative_summary']}")
        if news.get("insight_outlook"):
            sections.append(f"Insight/Outlook:\n{news['insight_outlook']}")
        return "\n\n".join(sections)
    

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r'(?<=[\.\!\?])\s+', text.strip())
        return [p for p in parts if p]

    def _analyze_sentiment_with_openai(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using OpenAI with structured output for financial text"""
        from pydantic import BaseModel, Field
        
        class SentimentAnalysis(BaseModel):
            sentiment_label: str = Field(description="Overall sentiment: 'positive', 'negative', or 'neutral'")
            confidence_score: float = Field(description="Confidence score between 0.0 and 1.0")
            positive_prob: float = Field(description="Probability of positive sentiment (0.0 to 1.0)")
            negative_prob: float = Field(description="Probability of negative sentiment (0.0 to 1.0)")
            neutral_prob: float = Field(description="Probability of neutral sentiment (0.0 to 1.0)")
            reasoning: str = Field(description="Brief explanation of the sentiment analysis")
        
        prompt = f"""Analyze the sentiment of the following financial text. Consider financial terminology, market context, and investment implications.

Text:
{text}

Provide a detailed sentiment analysis with probabilities that sum to 1.0. Focus on financial sentiment indicators like:
- Positive: growth, profit, gains, beat expectations, strong performance
- Negative: losses, decline, miss expectations, weak performance, concerns
- Neutral: stable, unchanged, mixed signals, balanced

Return structured sentiment analysis with probabilities."""

        try:
            structured_llm = self.llm.with_structured_output(SentimentAnalysis)
            result = structured_llm.invoke(prompt)
            
            sentiment_score = result.positive_prob - result.negative_prob
            
            return {
                "sentiment_label": result.sentiment_label.lower(),
                "confidence_score": result.confidence_score,
                "sentiment_score": sentiment_score,
                "positive_prob": result.positive_prob,
                "negative_prob": result.negative_prob,
                "neutral_prob": result.neutral_prob,
                "reasoning": result.reasoning,
                "uncertainty_entropy": self._calculate_entropy([
                    result.positive_prob, result.neutral_prob, result.negative_prob
                ])
            }
        except Exception as e:
            print(f"Error in OpenAI sentiment analysis: {e}")
            # Fallback to simple keyword-based analysis
            return self._fallback_sentiment_analysis(text)
    
    def _calculate_entropy(self, probs: List[float]) -> float:
        """Calculate entropy for uncertainty measurement"""
        import numpy as np
        probs = np.clip(probs, 1e-9, 1.0)
        return float(-np.sum(probs * np.log(probs)))
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Simple keyword-based fallback sentiment analysis"""
        positive_keywords = ['growth', 'profit', 'gain', 'strong', 'beat', 'exceed', 'increase', 'improve', 'positive', 'bullish']
        negative_keywords = ['loss', 'decline', 'weak', 'miss', 'concern', 'risk', 'decrease', 'negative', 'bearish', 'fall']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_keywords if word in text_lower)
        neg_count = sum(1 for word in negative_keywords if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return {
                "sentiment_label": "neutral",
                "confidence_score": 0.5,
                "sentiment_score": 0.0,
                "positive_prob": 0.33,
                "negative_prob": 0.33,
                "neutral_prob": 0.34,
                "uncertainty_entropy": 1.0
            }
        
        pos_prob = pos_count / total if total > 0 else 0.33
        neg_prob = neg_count / total if total > 0 else 0.33
        neu_prob = 1.0 - pos_prob - neg_prob
        
        if pos_prob > neg_prob:
            label = "positive"
            confidence = pos_prob
        elif neg_prob > pos_prob:
            label = "negative"
            confidence = neg_prob
        else:
            label = "neutral"
            confidence = neu_prob
        
        return {
            "sentiment_label": label,
            "confidence_score": confidence,
            "sentiment_score": pos_prob - neg_prob,
            "positive_prob": pos_prob,
            "negative_prob": neg_prob,
            "neutral_prob": neu_prob,
            "uncertainty_entropy": self._calculate_entropy([pos_prob, neu_prob, neg_prob])
        }
    
    def _collect_sentiment_evidence_openai(self, text: str, dominant_label: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Collect evidence sentences using OpenAI"""
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        
        # Use OpenAI to score sentences for the dominant sentiment
        evidence_prompt = f"""Given the following sentences from financial text, identify the top {top_k} sentences that most strongly support {dominant_label} sentiment.

Sentences:
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(sentences))}

Return a JSON array with the sentence numbers (1-indexed) that best support {dominant_label} sentiment, ordered by strength."""

        try:
            from pydantic import BaseModel, Field
            
            class EvidenceResponse(BaseModel):
                sentence_numbers: List[int] = Field(description=f"Top {top_k} sentence numbers (1-indexed) supporting {dominant_label} sentiment")
            
            structured_llm = self.llm.with_structured_output(EvidenceResponse)
            result = structured_llm.invoke(evidence_prompt)
            
            evidence = []
            for num in result.sentence_numbers[:top_k]:
                if 1 <= num <= len(sentences):
                    evidence.append({
                        "sentence": sentences[num - 1],
                        "label": dominant_label,
                        "confidence": 0.8  # Default confidence
                    })
            
            return evidence
        except Exception as e:
            print(f"Error collecting evidence with OpenAI: {e}")
            # Fallback: return first few sentences
            return [{"sentence": s, "label": dominant_label, "confidence": 0.5} 
                   for s in sentences[:top_k]]

    def run(self, text: str, ticker: Optional[str] = None, entities: Optional[List[str]] = None, 
            thread_id: Optional[str] = None) -> Dict[str, Any]:
        initial_state = {
            "text": text,
            "ticker": ticker,
            "entities": entities,
            "messages": [SystemMessage(content="Starting comprehensive financial analysis")],
            "errors": [],
            "metadata": {}
        }
        
        config = {"configurable": {"thread_id": thread_id or "default"}}
        
        result = self.app.invoke(initial_state, config)
        
        return result.get("final_analysis", {})
    
    def run_from_news(self, news: Dict[str, str], ticker: Optional[str] = None) -> Dict[str, Any]:

        # Run sentiment + risk + synthesis pipeline using the NewsScraperAgent output as input.
    
        text = self._build_text_from_news(news)
        return self.run(text=text, ticker=ticker)
    

    def run_batch(self, analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        
        for i, analysis in enumerate(analyses):
            thread_id = f"batch_{i}"
            result = self.run(
                text=analysis.get('text', ''),
                ticker=analysis.get('ticker'),
                entities=analysis.get('entities'),
                thread_id=thread_id
            )
            results.append(result)
        
        return results

    def get_analysis_history(self, thread_id: str) -> List[BaseMessage]:
        config = {"configurable": {"thread_id": thread_id}}
        state = self.app.get_state(config)
        return state.values.get("messages", [])


if __name__ == "__main__":
    agent = SentimentAgent(model="gpt-5-nano")
    
    # sample_text = """
    # Australia's top supermarket chains Woolworths and Coles said on Monday they could incur millions in additional remediation costs following the federal court's decision on historical underpayments to staff.
    # """
    
    # results = agent.run(text=sample_text, ticker="WOLWF", thread_id="analysis_thread")
    
    # print(json.dumps(results, indent=2))
    
    # print("\n--- Analysis History ---\n")
    # history = agent.get_analysis_history(thread_id="analysis_thread")
    # for msg in history:
    #     print(f"[{msg.type.upper()}]: {msg.content}\n")

    ### Possible news scraper output
    
    # First, scrape
    
    scraper = NewsScraperAgent(model="gpt-5-nano")
    news_output = scraper.run({"url": "https://finnhub.io/api/news?id=bec745cd1ffd8d5793fcd33ceb7c795378e49a5"})

    # Then analyze
    agent = SentimentAgent(model="gpt-5-nano")
    results = agent.run_from_news(news_output, ticker="")
    print(json.dumps(results, indent=2))
    
    # print("============== Probable News Scraper Output ===============")
    # scraper_output = {
    # "qualitative_summary": "Tesla announced the expansion of its Gigafactory operations in Berlin, emphasizing a stronger push into the European EV market. Management highlighted continued improvements in production efficiency and battery technology. The company also stressed its focus on sustainability and reducing reliance on rare earth materials. Additionally, executives noted progress in software-driven features such as Full Self-Driving, which remains a long-term strategic priority.",
    #  "quantitative_summary": "Q2 revenue: $25.3 billion. Net income: $3.4 billion. Automotive gross margin: 18.1%. Energy segment revenue: $1.5 billion. Deliveries: 466,000 vehicles. Operating cash flow: $2.6 billion. Capex: $2.1 billion.",
    #  "insight_outlook": "Tesla’s near-term outlook appears stable, supported by strong delivery volumes and expanding production capacity. However, margin compression remains a concern due to pricing pressures and rising raw material costs. The long-term outlook hinges on successful scaling of energy storage and autonomy features. Overall, while growth prospects are strong, investors should monitor margins and regulatory developments in key markets."
    #  }
    
    # r = agent.run(scraper_output["qualitative_summary"])
    # print(json.dumps(r, indent=2))
    # results_from_news = agent.run_from_news(scraper_output, ticker="TSLA")
    # print("\n--- Analysis from News Output ---\n")
    # print(json.dumps(results_from_news, indent=2))