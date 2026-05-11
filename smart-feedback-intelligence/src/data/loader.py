import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset
from src.utils.config import RAW_DIR, PROCESSED_DIR
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads Amazon Reviews or Twitter Customer Support dataset.
    Falls back to synthetic sample data for development.
    """

    def load_amazon_reviews(self, split: str = "train", max_samples: int = 50000) -> pd.DataFrame:
        """Load Amazon Polarity dataset from HuggingFace."""
        logger.info(f"Loading Amazon Reviews — {max_samples} samples")
        dataset = load_dataset("amazon_polarity", split=f"{split}[:{max_samples}]")
        df = dataset.to_pandas()
        df = df.rename(columns={"content": "text", "label": "sentiment_label"})
        df["sentiment"] = df["sentiment_label"].map({0: "negative", 1: "positive"})
        df["source"] = "amazon"
        return df[["text", "sentiment", "source"]]

    def load_twitter_support(self, max_samples: int = 20000) -> pd.DataFrame:
        """Load Twitter Customer Support dataset."""
        logger.info("Loading Twitter Customer Support dataset")
        try:
            dataset = load_dataset("twcs", split=f"train[:{max_samples}]")
            df = dataset.to_pandas()
            df = df.rename(columns={"text": "text"})
            df["source"] = "twitter"
            df["sentiment"] = "neutral"  # label via pipeline
            return df[["text", "sentiment", "source"]]
        except Exception as e:
            logger.warning(f"Twitter dataset unavailable: {e}. Using synthetic data.")
            return self._generate_synthetic_data(max_samples)

    def _generate_synthetic_data(self, n: int = 5000) -> pd.DataFrame:
        """Generate realistic synthetic feedback data for development."""
        np.random.seed(42)
        templates = {
            "negative": [
                "The product arrived damaged and support refused to help me.",
                "Terrible delivery experience, package came two weeks late.",
                "I was charged twice and still haven't received my refund.",
                "Customer service was extremely rude and unhelpful.",
                "Product quality is terrible, broke after one use.",
                "Never received my order and tracking shows delivered.",
                "Return process is a nightmare, very difficult to complete.",
            ],
            "neutral": [
                "The product is okay, nothing special but works as expected.",
                "Delivery was on time but packaging could be better.",
                "Support agent was polite but couldn't resolve my issue.",
                "Average quality for the price, would consider alternatives.",
                "Returns process took longer than expected but completed.",
            ],
            "positive": [
                "Amazing product quality, exceeded my expectations completely!",
                "Super fast delivery, received in just one day. Very happy!",
                "Customer support was incredibly helpful and resolved my issue quickly.",
                "Best purchase I've made this year, highly recommend!",
                "The refund was processed instantly, excellent service!",
                "Product works perfectly, great value for money.",
            ],
        }
        categories = ["billing", "delivery", "product", "support", "returns"]
        urgency = ["low", "medium", "high"]

        rows = []
        sentiments = ["negative", "neutral", "positive"]
        weights = [0.4, 0.2, 0.4]
        for _ in range(n):
            sentiment = np.random.choice(sentiments, p=weights)
            text = np.random.choice(templates[sentiment])
            rows.append({
                "text": text,
                "sentiment": sentiment,
                "category": np.random.choice(categories),
                "urgency": np.random.choice(urgency, p=[0.5, 0.3, 0.2]),
                "churn_risk": int(sentiment == "negative" and np.random.random() > 0.4),
                "source": "synthetic",
            })
        return pd.DataFrame(rows)

    def load_and_merge(self, max_samples: int = 30000) -> pd.DataFrame:
        """Load, merge, and return a ready-to-use DataFrame."""
        try:
            df = self.load_amazon_reviews(max_samples=max_samples)
        except Exception as e:
            logger.warning(f"Amazon dataset failed: {e}. Using synthetic.")
            df = self._generate_synthetic_data(max_samples)

        # Add derived labels if not present
        if "category" not in df.columns:
            df["category"] = np.random.choice(
                ["billing", "delivery", "product", "support", "returns"], size=len(df)
            )
        if "urgency" not in df.columns:
            df["urgency"] = np.where(
                df["sentiment"] == "negative",
                np.random.choice(["medium", "high"], size=len(df), p=[0.5, 0.5]),
                "low",
            )
        if "churn_risk" not in df.columns:
            df["churn_risk"] = (df["sentiment"] == "negative").astype(int)

        logger.info(f"Dataset loaded: {len(df)} samples")
        logger.info(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
        return df.reset_index(drop=True)
