import json
import re
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

import numpy as np
from langchain.output_parsers import StructuredOutputParser
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from utils.callback_handler import PrintCallbackHandler
from utils.finbert_analyzer import FinBERTAnalyzer
from utils.model_schema import SentimentModel
from utils.sec_risk_analyzer import SECRiskAnalyzer


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
    
    def __init__(self, model: str = "llama3.1", finbert_model: str = "ProsusAI/finbert"):
        self.callback_handler = PrintCallbackHandler()
        self.llm = ChatOllama(
            model=model, 
            temperature=0, 
            streaming=True, 
            callbacks=[self.callback_handler]
        )
        
        self.finbert_analyzer = FinBERTAnalyzer(model_name=finbert_model)
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
            
            result = self.finbert_analyzer.analyze_sentiment(text)
            
            if not result:
                raise ValueError("Sentiment analysis returned no results")
            
            sentiment_data = {
                "label": result["sentiment_label"],
                "confidence": result["confidence_score"],
                "sentiment_score": result["sentiment_score"],
                "probabilities": {
                    "positive": result["positive_prob"],
                    "negative": result["negative_prob"],
                    "neutral": result["neutral_prob"]
                },
                "dominant_sentiment": result["dominant_sentiment"],
                "sentiment_context": self._analyze_sentiment_context(text, result)
            }
            
            return {
                "sentiment_results": convert_numpy_types(sentiment_data),
                "current_stage": AnalysisStage.SENTIMENT_ANALYSIS.value,
                "messages": [AIMessage(content=f"Sentiment analysis complete: {sentiment_data['label']} ({sentiment_data['confidence']:.2f} confidence)")]
            }
            
        except Exception as e:
            return {
                "errors": [f"Sentiment analysis failed: {str(e)}"],
                "current_stage": "error"
            }

    def _risk_assessment_node(self, state: AgentState) -> Dict[str, Any]:
        try:
            text = state["text"]
            
            risk_result = self.risk_analyzer.analyze_comprehensive_risk(text)
            
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
            
            final_analysis = {
                "executive_summary": self._generate_executive_summary(state),
                "sentiment_analysis": state.get("sentiment_results", {}),
                "risk_assessment": state.get("risk_results", {}),
                "financial_metrics": state.get("financial_results", {}),
                "entities": state.get("extracted_entities", {}),
                "key_insights": self._extract_key_insights(state),
                "recommendations": self._generate_comprehensive_recommendations(state),
                "confidence_metrics": self._calculate_confidence_metrics(state),
                "llm_synthesis": response.content
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
                "messages": [AIMessage(content="Analysis synthesis complete")]
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
        
        if risk_scores.get("market", 0) > 0.6:
            suggestions.append("Consider diversification strategies to reduce market risk exposure")
        if risk_scores.get("operational", 0) > 0.6:
            suggestions.append("Review and strengthen operational processes and contingency plans")
        if risk_scores.get("financial", 0) > 0.6:
            suggestions.append("Focus on improving financial metrics and liquidity management")
        if risk_scores.get("regulatory", 0) > 0.6:
            suggestions.append("Enhance regulatory compliance monitoring and reporting")
        
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
    agent = SentimentAgent(model="llama3.1")
    
    sample_text = """
    Apple Inc. reported strong quarterly earnings, beating analyst expectations with revenue growth of 8% year-over-year. 
    The company's iPhone sales showed resilience despite market headwinds, while services revenue continued to expand. 
    However, management highlighted ongoing supply chain challenges and regulatory scrutiny in key markets. 
    The company faces increasing competition in the smartphone space and potential regulatory changes that could impact 
    its App Store revenue model. Despite these challenges, Apple's strong balance sheet and cash position provide 
    significant flexibility for future investments and shareholder returns.
    """
    
    results = agent.run(text=sample_text, ticker="ORSHF", thread_id="apple_analysis_thread")
    
    print(json.dumps(results, indent=2))
    
    print("\n--- Analysis History ---\n")
    history = agent.get_analysis_history(thread_id="apple_analysis_thread")
    for msg in history:
        print(f"[{msg.type.upper()}]: {msg.content}\n")