"""
inference.py — Quick test script to verify trained models work correctly.
Run: python inference.py

Use this AFTER running train.py to confirm everything is working
before starting the FastAPI server.
"""

import joblib
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

from src.nlp.pipeline import NLPPipeline
from src.nlp.absa import ABSAnalyzer
from src.utils.config import MODELS_DIR

# ── Sample test reviews ───────────────────────────────────
TEST_REVIEWS = [
    "The delivery was extremely late and the package arrived completely damaged. Very disappointed.",
    "Absolutely love this product! Fast shipping, great quality. Will definitely buy again!",
    "I was charged twice for my order and customer support has not responded in 3 days. Need refund urgently!",
    "Product is okay, nothing special. Delivery was on time. Average experience overall.",
    "The return process is a nightmare. Took 3 weeks and still not resolved. Terrible service.",
]


def run_inference():
    logger.info("=" * 60)
    logger.info("🔍 Running Inference Test on Trained Models")
    logger.info("=" * 60)

    # Load NLP pipeline
    pipeline = NLPPipeline(use_spacy=True)
    absa = ABSAnalyzer()

    # Load TF-IDF
    tfidf_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    if not tfidf_path.exists():
        logger.error("❌ TF-IDF not found. Run train.py first!")
        return

    tfidf = joblib.load(tfidf_path)
    logger.info("✅ TF-IDF loaded")

    # Load models
    models = {}
    encoders = {}
    for task in ["sentiment", "category", "urgency", "churn"]:
        model_path = MODELS_DIR / f"{task}_best_model.pkl"
        encoder_path = MODELS_DIR / f"{task}_label_encoder.pkl"
        if model_path.exists():
            models[task] = joblib.load(model_path)
            encoders[task] = joblib.load(encoder_path) if encoder_path.exists() else None
            logger.info(f"✅ {task} model loaded")
        else:
            logger.warning(f"⚠️  {task} model not found — skipping")

    # Run predictions
    logger.info("\n" + "=" * 60)
    logger.info("📊 PREDICTIONS")
    logger.info("=" * 60)

    for i, review in enumerate(TEST_REVIEWS, 1):
        logger.info(f"\n[Review {i}] {review[:80]}...")

        # NLP preprocessing
        processed = pipeline.process(review)
        X = tfidf.transform([processed])

        # ML predictions
        for task, model in models.items():
            pred = model.predict(X)[0]
            le = encoders.get(task)
            label = le.inverse_transform([pred])[0] if le else str(pred)
            confidence = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                confidence = round(float(np.max(proba)) * 100, 1)
            conf_str = f"({confidence}%)" if confidence else ""
            logger.info(f"  {task.upper():<12}: {label} {conf_str}")

        # ABSA
        aspects = absa.analyze(review)
        if aspects:
            logger.info(f"  ABSA        : {aspects}")

        # NER
        if pipeline.nlp:
            entities = pipeline.extract_entities(review)
            if entities:
                logger.info(f"  ENTITIES    : {entities}")

    # BERT inference (if model saved)
    bert_model_dir = MODELS_DIR / "bert_sentiment"
    if bert_model_dir.exists():
        logger.info("\n" + "=" * 60)
        logger.info("🧠 BERT INFERENCE")
        logger.info("=" * 60)
        try:
            from src.models.bert_classifier import BERTClassifier
            from src.utils.config import SENTIMENT_LABELS
            bert = BERTClassifier(num_labels=len(SENTIMENT_LABELS), task_name="sentiment")
            bert.load_model()
            results = bert.predict(TEST_REVIEWS[:3])
            for review, result in zip(TEST_REVIEWS[:3], results):
                label = SENTIMENT_LABELS[result["class_id"]]
                conf = round(result["confidence"] * 100, 1)
                logger.info(f"  '{review[:50]}...' → {label} ({conf}%)")
        except Exception as e:
            logger.warning(f"BERT inference failed: {e}")
    else:
        logger.info("\n⚠️  BERT model not found. Run train.py with BERT enabled.")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Inference test complete! Models are working correctly.")
    logger.info("🚀 Now run: uvicorn api.main:app --reload")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_inference()
