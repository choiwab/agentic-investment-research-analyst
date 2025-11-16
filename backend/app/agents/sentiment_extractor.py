# sentiment_agent.py
import re
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv(override=True)

# for sentence splitting and text cleaning
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\?!])\s+(?=[A-Z(\[])")
_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"^\W+$")

class SentimentAnalysisAgent:
    """
    Stand-alone OpenAI-based sentiment agent for financial text.
    Input: raw_text (str), optional meta dict, optional entities list for light weighting.
    Output: dict with label, confidence, probs, stability, and evidence sentences.
    """
    def __init__(self, model_id: str = "gpt-4o", max_length: int = 256):
        self.model_id = model_id
        self.max_length = max_length
        self.llm = ChatOpenAI(model=model_id, temperature=0)

    @staticmethod
    def _clean_text(text: str) -> str:
        return _WS_RE.sub(" ", text or "").strip()

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        text = SentimentAnalysisAgent._clean_text(text)
        if not text:
            return []
        sents = _SENT_SPLIT_RE.split(text)
        # keep only reasonable sentences
        sents = [s.strip() for s in sents if 5 <= len(s) <= 800 and not _NON_WORD_RE.match(s)]
        # dedup near-duplicates (simple hash)
        seen = set()
        uniq = []
        for s in sents:
            key = hash(s.lower())
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        return uniq

    @staticmethod
    def _entropy_row(p: np.ndarray) -> float:
        p = np.clip(p, 1e-9, 1.0)
        return float(-(p * np.log(p)).sum())

    def _analyze_sentiment_with_openai(self, text: str, entities: List[str] = None) -> Dict[str, Any]:
        """Analyze sentiment using OpenAI with structured output"""
        class SentimentResult(BaseModel):
            sentiment_label: str = Field(description="Overall sentiment: 'positive', 'negative', or 'neutral'")
            confidence_score: float = Field(description="Confidence score between 0.0 and 1.0")
            positive_prob: float = Field(description="Probability of positive sentiment (0.0 to 1.0)")
            negative_prob: float = Field(description="Probability of negative sentiment (0.0 to 1.0)")
            neutral_prob: float = Field(description="Probability of neutral sentiment (0.0 to 1.0)")
            reasoning: str = Field(description="Brief explanation of the sentiment analysis")
        
        entities_context = ""
        if entities:
            entities_context = f"\n\nPay special attention to mentions of: {', '.join(entities)}"
        
        prompt = f"""Analyze the sentiment of the following financial text. Consider financial terminology, market context, and investment implications.{entities_context}

Text:
{text}

Provide a detailed sentiment analysis with probabilities that sum to 1.0. Focus on financial sentiment indicators like:
- Positive: growth, profit, gains, beat expectations, strong performance, bullish
- Negative: losses, decline, miss expectations, weak performance, concerns, bearish
- Neutral: stable, unchanged, mixed signals, balanced

Return structured sentiment analysis with probabilities."""

        try:
            structured_llm = self.llm.with_structured_output(SentimentResult)
            result = structured_llm.invoke(prompt)
            
            return {
                "sentiment_label": result.sentiment_label.lower(),
                "confidence_score": result.confidence_score,
                "positive_prob": result.positive_prob,
                "negative_prob": result.negative_prob,
                "neutral_prob": result.neutral_prob,
                "reasoning": result.reasoning
            }
        except Exception as e:
            print(f"Error in OpenAI sentiment analysis: {e}")
            # Fallback to keyword-based analysis
            return self._fallback_sentiment_analysis(text)

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
                "positive_prob": 0.33,
                "negative_prob": 0.33,
                "neutral_prob": 0.34
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
            "positive_prob": pos_prob,
            "negative_prob": neg_prob,
            "neutral_prob": neu_prob
        }

    def run(self, raw_text: str, meta: dict | None = None, entities: list[str] | None = None) -> dict:
        if not raw_text or len(raw_text) < 40:
            return {"error": "empty_or_short_text", "meta": meta or {}}

        # Analyze sentiment with OpenAI
        sentiment_result = self._analyze_sentiment_with_openai(raw_text, entities or [])
        
        if "error" in sentiment_result:
            return {"error": "sentiment_analysis_failed", "meta": meta or {}}

        # Extract probabilities
        p_neg_mean = sentiment_result.get("negative_prob", 0.33)
        p_neu_mean = sentiment_result.get("neutral_prob", 0.33)
        p_pos_mean = sentiment_result.get("positive_prob", 0.34)
        
        # Calculate entropy for stability
        probs_array = np.array([p_neg_mean, p_neu_mean, p_pos_mean])
        entropy_mean = self._entropy_row(probs_array)
        p_neg_var = 0.0  # Variance not calculated for single analysis

        # Get evidence sentences
        sentences = self._split_sentences(raw_text)
        k = min(5, len(sentences))
        
        # Use keyword matching for evidence (simplified)
        top_negative = []
        top_positive = []
        negative_keywords = ['loss', 'decline', 'weak', 'miss', 'concern', 'risk', 'decrease', 'negative', 'bearish']
        positive_keywords = ['growth', 'profit', 'gain', 'strong', 'beat', 'exceed', 'increase', 'improve', 'positive', 'bullish']
        
        for i, s in enumerate(sentences[:k]):
            s_lower = s.lower()
            neg_score = sum(1 for word in negative_keywords if word in s_lower)
            pos_score = sum(1 for word in positive_keywords if word in s_lower)
            
            if neg_score > 0:
                top_negative.append({"sentence": s, "p_neg": float(neg_score / max(len(s.split()), 1))})
            if pos_score > 0:
                top_positive.append({"sentence": s, "p_pos": float(pos_score / max(len(s.split()), 1))})

        return {
            "sentiment": {
                "label": sentiment_result.get("sentiment_label", "neutral"),
                "confidence": sentiment_result.get("confidence_score", 0.5),
                "model": self.model_id,
                "aggregation": "openai_structured"
            },
            "probs": {
                "p_neg_mean": p_neg_mean,
                "p_neu_mean": p_neu_mean,
                "p_pos_mean": p_pos_mean
            },
            "stability": {
                "p_neg_var": p_neg_var,
                "entropy_mean": entropy_mean
            },
            "evidence": {
                "top_negative": top_negative[:k],
                "top_positive": top_positive[:k]
            },
            "meta": meta or {}
        }

# --- quick usage example ---
article_text = "FixStop at Alafaya has earned a strong reputation in Orlando for delivering quick, reliable repairs combined with competitive pricing, making it a preferred destination for residents seeking expert electronic device servicing. FixStop at Alafaya has earned a strong reputation in Orlando for delivering quick, reliable repairs combined with competitive pricing, making it a preferred destination for residents seeking expert electronic device servicing. Located conveniently in the Shoppes at Eastwood Publix Plaza, this repair shop has become synonymous with quality and efficiency, offering a comprehensive range of services that cover everything from smartphones to computers and beyond. Fast and Expert Repair Services. One of the key strengths of FixStop at Alafaya is its commitment to fast turnaround times without compromising on quality. Customers frequently praise the shop for completing repairs within 24 hours, with some services such as iPhone screen replacements done in under an hour. This rapid service is complemented by the expertise of certified technicians who handle even the most complex repairs, including micro-soldering and board-level fixes under a microscope. This combination of speed and technical skill ensures that clients receive dependable repairs that restore their devices to optimal functionality quickly. In addition to repairs, FixStop at Alafaya also caters to customers interested in building or upgrading a custom PC. This service involves assembling personalized desktop computers tailored to specific needs, whether for gaming, professional work, or general use. The technicians assist with component selection, installation, and configuration to ensure optimal performance and reliability. This capability reflects the shop's comprehensive understanding of computer hardware and its commitment to meeting diverse customer requirements."


# Commented out to prevent loading model at import time
# agent = SentimentAnalysisAgent()
# out = agent.run(article_text, meta={"ticker":"AAPL","source":"Reuters"}, entities=["Apple","AAPL"])
# print(out["sentiment"], out["probs"])