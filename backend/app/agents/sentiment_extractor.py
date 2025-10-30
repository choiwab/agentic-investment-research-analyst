# sentiment_agent.py
import re
import math
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# for sentence splitting and text cleaning
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\?!])\s+(?=[A-Z(\[])")
_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"^\W+$")

class SentimentAnalysisAgent:
    """
    Stand-alone FinBERT-based sentiment agent.
    Input: raw_text (str), optional meta dict, optional entities list for light weighting.
    Output: dict with label, confidence, probs, stability, and evidence sentences.
    """
    def __init__(self, model_id: str = "yiyanghkust/finbert-tone", max_length: int = 256):
        self.model_id = model_id
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id).to(self.device).eval()

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

    @torch.inference_mode()
    def _score_batch(self, sentences: list[str], batch_size: int = 32) -> np.ndarray:
        probs = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            enc = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            p = torch.softmax(logits, dim=-1).detach().cpu().numpy()  # [N,3] -> [neg, neu, pos]
            probs.append(p)
        return np.vstack(probs) if probs else np.zeros((0, 3), dtype=np.float32)

    @staticmethod
    def _entropy_row(p: np.ndarray) -> float:
        p = np.clip(p, 1e-9, 1.0)
        return float(-(p * np.log(p)).sum())

    def run(self, raw_text: str, meta: dict | None = None, entities: list[str] | None = None) -> dict:
        if not raw_text or len(raw_text) < 40:
            return {"error": "empty_or_short_text", "meta": meta or {}}

        sentences = self._split_sentences(raw_text)
        if not sentences:
            return {"error": "no_sentences_parsed", "meta": meta or {}}

        # light weighting: boost headline + sentences that mention entities/tickers
        weights = np.ones(len(sentences), dtype=np.float32)
        weights[0] *= 1.5  # headline boost
        if entities:
            pat = re.compile(r"|".join([re.escape(e) for e in entities]), re.I)
            for i, s in enumerate(sentences):
                if pat.search(s):
                    weights[i] *= 1.2
        weights = weights / weights.sum()

        probs = self._score_batch(sentences)  # [N,3] -> [neg, neu, pos]
        p_neg, p_neu, p_pos = probs[:, 0], probs[:, 1], probs[:, 2]

        # weighted document-level means
        p_neg_mean = float(np.dot(p_neg, weights))
        p_neu_mean = float(np.dot(p_neu, weights))
        p_pos_mean = float(np.dot(p_pos, weights))

        # stability / uncertainty
        p_neg_var = float(np.dot((p_neg - p_neg_mean) ** 2, weights))
        entropies = np.array([self._entropy_row(p) for p in probs], dtype=np.float32)
        entropy_mean = float(np.dot(entropies, weights))

        # label + confidence from aggregated probs
        means = np.array([p_neg_mean, p_neu_mean, p_pos_mean])
        idx = int(np.argmax(means))
        label = ["negative", "neutral", "positive"][idx]
        confidence = float(means[idx])

        # evidence sentences (top-k extremes)
        k = min(5, len(sentences))
        top_negative_idx = np.argsort(-p_neg)[:k]
        top_positive_idx = np.argsort(-p_pos)[:k]
        top_negative = [{"sentence": sentences[i], "p_neg": float(p_neg[i])} for i in top_negative_idx]
        top_positive = [{"sentence": sentences[i], "p_pos": float(p_pos[i])} for i in top_positive_idx]

        return {
            "sentiment": {
                "label": label,
                "confidence": confidence,
                "model": self.model_id,
                "aggregation": "sentence_mean_weighted"
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
                "top_negative": top_negative,
                "top_positive": top_positive
            },
            "meta": meta or {}
        }

# --- quick usage example ---
article_text = "FixStop at Alafaya has earned a strong reputation in Orlando for delivering quick, reliable repairs combined with competitive pricing, making it a preferred destination for residents seeking expert electronic device servicing. FixStop at Alafaya has earned a strong reputation in Orlando for delivering quick, reliable repairs combined with competitive pricing, making it a preferred destination for residents seeking expert electronic device servicing. Located conveniently in the Shoppes at Eastwood Publix Plaza, this repair shop has become synonymous with quality and efficiency, offering a comprehensive range of services that cover everything from smartphones to computers and beyond. Fast and Expert Repair Services. One of the key strengths of FixStop at Alafaya is its commitment to fast turnaround times without compromising on quality. Customers frequently praise the shop for completing repairs within 24 hours, with some services such as iPhone screen replacements done in under an hour. This rapid service is complemented by the expertise of certified technicians who handle even the most complex repairs, including micro-soldering and board-level fixes under a microscope. This combination of speed and technical skill ensures that clients receive dependable repairs that restore their devices to optimal functionality quickly. In addition to repairs, FixStop at Alafaya also caters to customers interested in building or upgrading a custom PC. This service involves assembling personalized desktop computers tailored to specific needs, whether for gaming, professional work, or general use. The technicians assist with component selection, installation, and configuration to ensure optimal performance and reliability. This capability reflects the shop's comprehensive understanding of computer hardware and its commitment to meeting diverse customer requirements."


agent = SentimentAnalysisAgent()
out = agent.run(article_text, meta={"ticker":"AAPL","source":"Reuters"}, entities=["Apple","AAPL"])
print(out["sentiment"], out["probs"])