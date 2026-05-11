"""
train.py — Full training pipeline orchestrator.
Run: python train.py

Steps:
  1. Load data
  2. NLP preprocessing
  3. Feature extraction (TF-IDF + BERT embeddings)
  4. Train + compare classical ML models
  5. Fine-tune BERT
  6. Save all models
  7. All experiments tracked in MLflow
"""

import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

from src.data.loader import DataLoader
from src.nlp.pipeline import NLPPipeline
from src.nlp.absa import ABSAnalyzer
from src.models.classical import ModelTrainer
from src.models.bert_classifier import BERTClassifier
from src.utils.config import MODELS_DIR, SENTIMENT_LABELS, CATEGORY_LABELS, URGENCY_LABELS


def main():
    logger.info("🚀 Starting Smart Feedback Intelligence Training Pipeline")

    # ── 1. Load Data ─────────────────────────────────────
    logger.info("\n📦 Step 1: Loading data...")
    loader = DataLoader()
    df = loader.load_and_merge(max_samples=30000)
    logger.info(f"Loaded {len(df)} samples")

    # ── 2. NLP Preprocessing ─────────────────────────────
    logger.info("\n🔤 Step 2: NLP preprocessing...")
    pipeline = NLPPipeline()
    df["processed_text"] = pipeline.process_batch(df["text"].tolist())
    df = df[df["processed_text"].str.len() > 5].reset_index(drop=True)

    # Text stats features
    stats_df = pipeline.get_stats_dataframe(df["text"].tolist())
    logger.info(f"Text stats features: {stats_df.shape[1]} columns")

    # ── 3. ABSA ──────────────────────────────────────────
    logger.info("\n🎯 Step 3: Aspect-Based Sentiment Analysis...")
    absa = ABSAnalyzer()
    absa_summary = absa.get_aspect_summary(df["text"].tolist()[:5000])
    logger.info(f"\nABSA Summary:\n{absa_summary}")

    # ── 4. Feature Extraction ────────────────────────────
    logger.info("\n⚙️  Step 4: Feature extraction...")
    X_tfidf = pipeline.fit_tfidf(df["processed_text"].tolist(), max_features=10000)

    # Save pipeline components
    joblib.dump(pipeline.tfidf_vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    logger.info(f"TF-IDF shape: {X_tfidf.shape}")

    # ── 5. Encode Labels ─────────────────────────────────
    logger.info("\n🏷️  Step 5: Encoding labels...")
    label_map = {}
    for task, col in [("sentiment", "sentiment"), ("category", "category"),
                      ("urgency", "urgency"), ("churn", "churn_risk")]:
        le = LabelEncoder()
        df[f"{task}_encoded"] = le.fit_transform(df[col].astype(str))
        label_map[task] = le
        joblib.dump(le, MODELS_DIR / f"{task}_label_encoder.pkl")
        logger.info(f"{task}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ── 6. Train/Test Split ──────────────────────────────
    logger.info("\n✂️  Step 6: Train/Test split (80/20)...")
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=df["sentiment_encoded"])

    X_train = X_tfidf[train_idx]
    X_test = X_tfidf[test_idx]

    y_train_dict = {task: df[f"{task}_encoded"].values[train_idx] for task in label_map}
    y_test_dict = {task: df[f"{task}_encoded"].values[test_idx] for task in label_map}

    # ── 7. Classical ML Training ─────────────────────────
    logger.info("\n🤖 Step 7: Training classical ML models...")
    trainer = ModelTrainer()
    results = trainer.train_all_models(X_train, X_test, y_train_dict, y_test_dict)
    logger.info(f"\n📊 Classical ML Results: {results}")

    # ── 8. XGBoost Hyperparameter Tuning ─────────────────
    logger.info("\n🔧 Step 8: Tuning XGBoost for sentiment...")
    tuned_xgb = trainer.tune_xgboost("sentiment", X_train, y_train_dict["sentiment"])
    joblib.dump(tuned_xgb, MODELS_DIR / "sentiment_xgboost_tuned.pkl")

    # ── 9. BERT Fine-tuning ──────────────────────────────
    logger.info("\n🧠 Step 9: Fine-tuning BERT for sentiment...")
    train_texts = df["text"].values[train_idx].tolist()
    test_texts = df["text"].values[test_idx].tolist()
    train_sentiment_labels = df["sentiment_encoded"].values[train_idx].tolist()
    test_sentiment_labels = df["sentiment_encoded"].values[test_idx].tolist()

    bert_model = BERTClassifier(num_labels=len(SENTIMENT_LABELS), task_name="sentiment")
    bert_model.build_model()
    train_loader, val_loader = bert_model.prepare_data(
        train_texts, train_sentiment_labels,
        test_texts, test_sentiment_labels
    )
    bert_results = bert_model.train(train_loader, val_loader, SENTIMENT_LABELS)
    logger.info(f"BERT Best Val F1: {bert_results['best_val_f1']:.4f}")

    # ── Done ─────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info(f"📁 Models saved to: {MODELS_DIR}")
    logger.info("📊 View MLflow dashboard: mlflow ui")
    logger.info("🚀 Start API: uvicorn api.main:app --reload")
    logger.info("="*60)


if __name__ == "__main__":
    main()
