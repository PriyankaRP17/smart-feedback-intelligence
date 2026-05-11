"""
preprocessor.py — EDA, feature engineering, topic modeling (LDA).
Run standalone: python -m src.data.preprocessor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import logging
from pathlib import Path
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from src.utils.config import PROCESSED_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """EDA, feature engineering, and topic modeling."""

    def run_eda(self, df: pd.DataFrame, save_dir: Path = PROCESSED_DIR):
        """Generate and save EDA charts."""
        save_dir.mkdir(exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Customer Feedback — EDA Dashboard", fontsize=16, fontweight="bold")

        # Sentiment distribution
        df["sentiment"].value_counts().plot(
            kind="bar", ax=axes[0, 0], color=["#D85A30", "#BA7517", "#1D9E75"],
            edgecolor="white"
        )
        axes[0, 0].set_title("Sentiment Distribution")
        axes[0, 0].tick_params(axis="x", rotation=0)

        # Category distribution
        df["category"].value_counts().plot(
            kind="bar", ax=axes[0, 1], color="#7F77DD", edgecolor="white"
        )
        axes[0, 1].set_title("Category Distribution")
        axes[0, 1].tick_params(axis="x", rotation=30)

        # Urgency distribution
        df["urgency"].value_counts().plot(
            kind="pie", ax=axes[0, 2],
            colors=["#1D9E75", "#BA7517", "#D85A30"],
            autopct="%1.1f%%", startangle=90
        )
        axes[0, 2].set_title("Urgency Distribution")

        # Text length distribution
        df["text_length"] = df["text"].str.len()
        axes[1, 0].hist(df["text_length"], bins=50, color="#378ADD", edgecolor="white")
        axes[1, 0].set_title("Review Length Distribution")
        axes[1, 0].set_xlabel("Characters")

        # Sentiment vs Category heatmap
        cross = pd.crosstab(df["category"], df["sentiment"])
        sns.heatmap(cross, ax=axes[1, 1], cmap="YlOrRd", fmt="d", annot=True)
        axes[1, 1].set_title("Category vs Sentiment")

        # Churn risk by sentiment
        churn_df = df.groupby("sentiment")["churn_risk"].mean().reset_index()
        axes[1, 2].bar(
            churn_df["sentiment"], churn_df["churn_risk"],
            color=["#D85A30", "#BA7517", "#1D9E75"]
        )
        axes[1, 2].set_title("Churn Risk by Sentiment")
        axes[1, 2].set_ylabel("Avg Churn Rate")

        plt.tight_layout()
        chart_path = save_dir / "eda_dashboard.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✅ EDA chart saved: {chart_path}")
        return chart_path

    def run_topic_modeling(
        self, processed_texts: list, n_topics: int = 8, n_words: int = 10
    ) -> dict:
        """Run LDA topic modeling and save results."""
        logger.info(f"Running LDA topic modeling (n_topics={n_topics})...")

        vectorizer = CountVectorizer(
            max_features=5000, min_df=3, max_df=0.9, ngram_range=(1, 2)
        )
        X = vectorizer.fit_transform(processed_texts)
        vocab = vectorizer.get_feature_names_out()

        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=20,
            learning_method="online",
            random_state=42,
            n_jobs=-1,
        )
        lda.fit(X)

        # Extract top words per topic
        topics = {}
        topic_labels = [
            "Product Quality", "Delivery Issues", "Customer Support",
            "Billing & Payments", "Returns & Refunds",
            "User Experience", "Shipping Speed", "General Feedback"
        ]
        for i, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [vocab[j] for j in top_indices]
            label = topic_labels[i] if i < len(topic_labels) else f"Topic {i+1}"
            topics[f"topic_{i+1}"] = {
                "label": label,
                "top_words": top_words,
                "weight": float(topic.sum()),
            }

        # Save
        MODELS_DIR.mkdir(exist_ok=True)
        topics_path = MODELS_DIR / "topics.json"
        topics_path.write_text(json.dumps(topics, indent=2))
        joblib.dump(lda, MODELS_DIR / "lda_model.pkl")
        joblib.dump(vectorizer, MODELS_DIR / "lda_vectorizer.pkl")

        logger.info(f"✅ Topics saved to: {topics_path}")
        for k, v in topics.items():
            logger.info(f"  {v['label']}: {', '.join(v['top_words'][:5])}")

        return topics

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to the dataframe."""
        df = df.copy()
        df["text_length"] = df["text"].str.len()
        df["word_count"] = df["text"].str.split().str.len()
        df["avg_word_length"] = df["text"].apply(
            lambda x: np.mean([len(w) for w in str(x).split()]) if x else 0
        )
        df["exclamation_count"] = df["text"].str.count("!")
        df["question_count"] = df["text"].str.count(r"\?")
        df["uppercase_ratio"] = df["text"].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
        )
        df["has_negation"] = df["text"].str.lower().apply(
            lambda x: int(any(w in x for w in ["not", "no", "never", "didn't", "wasn't"]))
        )
        df["sentiment_score"] = df["sentiment"].map(
            {"positive": 1, "neutral": 0, "negative": -1}
        )
        logger.info(f"✅ Engineered {df.shape[1]} features")
        return df

    def save_processed(self, df: pd.DataFrame):
        """Save processed dataset."""
        PROCESSED_DIR.mkdir(exist_ok=True)
        path = PROCESSED_DIR / "processed_feedback.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"✅ Processed data saved: {path} ({len(df)} rows)")
        return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data.loader import DataLoader
    from src.nlp.pipeline import NLPPipeline

    logger.info("Running EDA + Topic Modeling...")
    loader = DataLoader()
    df = loader.load_and_merge(max_samples=5000)

    pipeline = NLPPipeline(use_spacy=False)
    df["processed_text"] = pipeline.process_batch(df["text"].tolist())

    preprocessor = DataPreprocessor()
    df = preprocessor.engineer_features(df)
    preprocessor.run_eda(df)
    preprocessor.run_topic_modeling(df["processed_text"].tolist())
    preprocessor.save_processed(df)
    logger.info("✅ Done!")
