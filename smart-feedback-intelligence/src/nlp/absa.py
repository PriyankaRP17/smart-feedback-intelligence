import re
import numpy as np
import pandas as pd
from typing import List, Dict
from src.utils.config import ASPECTS
import logging

logger = logging.getLogger(__name__)


class ABSAnalyzer:
    """
    Aspect-Based Sentiment Analysis.
    Detects which aspects are mentioned in a review and classifies
    the sentiment for each aspect independently.

    Example:
        Input:  "Delivery was late but the product quality is great!"
        Output: {"delivery": "negative", "product": "positive"}
    """

    def __init__(self):
        self.aspects = ASPECTS
        self.positive_words = {
            "great", "excellent", "amazing", "wonderful", "fantastic",
            "good", "best", "love", "perfect", "happy", "quick",
            "fast", "helpful", "easy", "smooth", "resolved", "satisfied"
        }
        self.negative_words = {
            "terrible", "awful", "bad", "worst", "poor", "horrible",
            "broken", "damaged", "late", "slow", "rude", "unhelpful",
            "difficult", "failed", "useless", "disappointed", "frustrating"
        }
        self.negation_words = {"not", "no", "never", "didn't", "wasn't",
                               "isn't", "aren't", "couldn't", "wouldn't"}

    def _extract_aspect_sentences(self, text: str, aspect_keywords: List[str]) -> List[str]:
        """Extract sentences that mention a given aspect."""
        sentences = re.split(r"[.!?,;]", text.lower())
        relevant = []
        for sentence in sentences:
            if any(kw in sentence for kw in aspect_keywords):
                relevant.append(sentence.strip())
        return relevant

    def _classify_sentence_sentiment(self, sentence: str) -> str:
        """Rule-based + lexicon sentiment for a single sentence."""
        tokens = sentence.lower().split()
        pos_score = 0
        neg_score = 0
        i = 0
        while i < len(tokens):
            token = re.sub(r"[^\w]", "", tokens[i])
            if token in self.negation_words and i + 1 < len(tokens):
                # Flip next sentiment word
                next_token = re.sub(r"[^\w]", "", tokens[i + 1])
                if next_token in self.positive_words:
                    neg_score += 1.5
                elif next_token in self.negative_words:
                    pos_score += 1.5
                i += 2
                continue
            if token in self.positive_words:
                pos_score += 1
            elif token in self.negative_words:
                neg_score += 1
            i += 1

        if pos_score > neg_score:
            return "positive"
        elif neg_score > pos_score:
            return "negative"
        return "neutral"

    def analyze(self, text: str) -> Dict[str, str]:
        """
        Run ABSA on input text.
        Returns dict of aspect → sentiment for detected aspects only.
        """
        results = {}
        for aspect, keywords in self.aspects.items():
            sentences = self._extract_aspect_sentences(text, keywords)
            if not sentences:
                continue
            sentiments = [self._classify_sentence_sentiment(s) for s in sentences]
            # Aggregate: majority vote
            from collections import Counter
            most_common = Counter(sentiments).most_common(1)[0][0]
            results[aspect] = most_common
        return results

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, str]]:
        """Run ABSA on a list of texts."""
        return [self.analyze(t) for t in texts]

    def to_dataframe(self, texts: List[str]) -> pd.DataFrame:
        """Return ABSA results as a DataFrame with one column per aspect."""
        results = self.analyze_batch(texts)
        df = pd.DataFrame(results).fillna("not_mentioned")
        df.index = range(len(texts))
        return df

    def get_aspect_summary(self, texts: List[str]) -> pd.DataFrame:
        """
        Aggregate ABSA results across all texts.
        Returns a summary of sentiment distribution per aspect.
        """
        results = self.analyze_batch(texts)
        rows = []
        for aspect in self.aspects.keys():
            sentiments = [r.get(aspect, "not_mentioned") for r in results]
            mentioned = [s for s in sentiments if s != "not_mentioned"]
            if not mentioned:
                continue
            from collections import Counter
            counts = Counter(mentioned)
            total = len(mentioned)
            rows.append({
                "aspect": aspect,
                "total_mentions": total,
                "positive": counts.get("positive", 0),
                "neutral": counts.get("neutral", 0),
                "negative": counts.get("negative", 0),
                "positive_pct": round(counts.get("positive", 0) / total * 100, 1),
                "negative_pct": round(counts.get("negative", 0) / total * 100, 1),
            })
        return pd.DataFrame(rows).sort_values("total_mentions", ascending=False)
