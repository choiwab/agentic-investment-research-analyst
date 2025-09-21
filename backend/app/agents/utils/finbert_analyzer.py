import re
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class FinBERTAnalyzer:
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_mapping = None
        self._load_model()
        
    def _load_model(self):
        """Load FinBERT model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            if hasattr(self.model.config, 'id2label'):
                self.label_mapping = self.model.config.id2label
            else:
                self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
            
            print(f"FinBERT model loaded successfully on {self.device}")
            print(f"Label mapping: {self.label_mapping}")
        except Exception as e:
            print(f"Error loading FinBERT model: {e}")
            self.model = None
            self.tokenizer = None
    

    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis"""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters but keep financial symbols
        text = re.sub(r'[^\w\s\$\%\.\,\-\+]', '', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks that fit model's max input length"""
        if not self.tokenizer:
            return [text]
            
        # Tokenize to check length
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        if len(tokens) <= max_length:
            return [text]
        
        # Use sliding window approach for better context preservation
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            sentence_length = len(sentence_tokens)
            
            if current_length + sentence_length > max_length - 2:  # Account for special tokens
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Keep last few sentences for overlap
                    overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk[-1:]
                    current_chunk = overlap_sentences + [sentence]
                    current_length = sum(len(self.tokenizer.encode(s, add_special_tokens=False)) 
                                       for s in current_chunk)
                else:
                    # Single sentence too long, truncate it
                    current_chunk = [sentence[:max_length * 3]]  # Rough character estimate
                    current_length = max_length - 2
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using FinBERT model"""

        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            

            # Split into chunks if necessary
            chunks = self.chunk_text(processed_text)
            
            all_probabilities = []
            
            for chunk in chunks:
                # Tokenize
                inputs = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    
                # Convert to numpy
                predictions_np = predictions.cpu().numpy()[0]
                all_probabilities.append(predictions_np)
            
            avg_probabilities = np.mean(all_probabilities, axis=0)
            # Convert to native Python types for serialization
            avg_probabilities = avg_probabilities.astype(float)
            
            sentiment_probs = {}
            for idx, label in self.label_mapping.items():
                label_lower = label.lower()
                if 'pos' in label_lower:
                    sentiment_probs['positive'] = float(avg_probabilities[idx])
                elif 'neg' in label_lower:
                    sentiment_probs['negative'] = float(avg_probabilities[idx])
                elif 'neu' in label_lower:
                    sentiment_probs['neutral'] = float(avg_probabilities[idx])
            
            # Ensure we have all three sentiments
            sentiment_probs.setdefault('positive', 0.0)
            sentiment_probs.setdefault('negative', 0.0)
            sentiment_probs.setdefault('neutral', 0.0)
            
            # Calculate sentiment score (-1 to 1)
            sentiment_score = sentiment_probs['positive'] - sentiment_probs['negative']
            
            dominant_sentiment = max(sentiment_probs, key=sentiment_probs.get)
            confidence = sentiment_probs[dominant_sentiment]
            
            return {
                "sentiment_score": float(sentiment_score),
                "confidence_score": float(confidence),
                "sentiment_label": dominant_sentiment,
                "positive_prob": sentiment_probs['positive'],
                "negative_prob": sentiment_probs['negative'],
                "neutral_prob": sentiment_probs['neutral'],
                "dominant_sentiment": dominant_sentiment
            }
            
        except Exception as e:
            print(f"Error in FinBERT analysis: {e}")
    
    def batch_analyze(self, texts: List[str], batch_size: int = 8) -> List[Dict[str, float]]:
        """Analyze sentiment for multiple texts with batching for efficiency"""

        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Process each text individually to handle varying lengths
            for text in batch_texts:
                results.append(self.analyze_sentiment(text))
        
        return results
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model"""
        if not self.model:
            return {"status": "Model not loaded", "device": str(self.device)}
        
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "label_mapping": self.label_mapping,
            "num_labels": self.model.config.num_labels,
            "max_position_embeddings": self.model.config.max_position_embeddings if hasattr(self.model.config, 'max_position_embeddings') else 512
        }



if __name__ == "__main__":
    analyzer = FinBERTAnalyzer()
    
    sample_texts = [
        "The company reported strong earnings growth and exceeded analyst expectations.",
        "Market volatility increased due to economic uncertainty and declining profits.",
        "The quarterly report shows stable performance with moderate revenue growth."
    ]
    
    for text in sample_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"\nText: {text[:50]}...")
        print(f"Sentiment: {result['sentiment_label']} (score: {result['sentiment_score']:.3f})")
        print(f"Confidence: {result['confidence_score']:.3f}")
        print(f"Probabilities - Pos: {result['positive_prob']:.3f}, "
              f"Neg: {result['negative_prob']:.3f}, Neu: {result['neutral_prob']:.3f}")