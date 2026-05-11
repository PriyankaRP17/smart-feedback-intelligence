"""
FastAPI Application — Smart Customer Feedback Intelligence System
Endpoints:
  POST /analyze       → Single feedback analysis
  POST /batch         → Bulk feedback analysis
  GET  /health        → Health check
  GET  /topics        → Topic modeling results
  POST /token         → JWT auth
"""

import time
import logging
import joblib
import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from api.schemas.models import (
    FeedbackRequest, FeedbackResponse,
    BatchRequest, BatchResponse,
    TokenResponse
)
from src.nlp.pipeline import NLPPipeline
from src.nlp.absa import ABSAnalyzer
from src.utils.config import MODELS_DIR, SENTIMENT_LABELS, CATEGORY_LABELS, URGENCY_LABELS

logger = logging.getLogger(__name__)

# ── Global model state ───────────────────────────────────
class ModelStore:
    pipeline: Optional[NLPPipeline] = None
    absa: Optional[ABSAnalyzer] = None
    tfidf = None
    models: Dict = {}
    encoders: Dict = {}
    ready: bool = False


store = ModelStore()


def load_models():
    """Load all trained models into memory at startup."""
    logger.info("Loading models into memory...")
    try:
        store.pipeline = NLPPipeline()
        store.absa = ABSAnalyzer()

        tfidf_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        if tfidf_path.exists():
            store.tfidf = joblib.load(tfidf_path)

        for task in ["sentiment", "category", "urgency", "churn"]:
            model_path = MODELS_DIR / f"{task}_best_model.pkl"
            encoder_path = MODELS_DIR / f"{task}_label_encoder.pkl"
            if model_path.exists():
                store.models[task] = joblib.load(model_path)
                logger.info(f"  ✅ {task} model loaded")
            if encoder_path.exists():
                store.encoders[task] = joblib.load(encoder_path)

        store.ready = True
        logger.info("✅ All models loaded and ready!")
    except Exception as e:
        logger.error(f"❌ Model loading error: {e}")
        store.ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    logger.info("Shutting down...")


# ── App init ──────────────────────────────────────────────
app = FastAPI(
    title="Smart Feedback Intelligence API",
    description="Analyze customer feedback — sentiment, category, urgency, churn risk, ABSA",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# ── Label maps (fallback when encoder pkl is None) ───────
LABEL_MAPS = {
    "sentiment": {0: "negative", 1: "positive"},
    "category":  {0: "billing", 1: "delivery", 2: "product", 3: "returns", 4: "support"},
    "urgency":   {0: "high", 1: "low", 2: "medium"},
    "churn":     {0: "no_risk", 1: "at_risk"},
}

# ── Inference helper ─────────────────────────────────────
def run_inference(text: str) -> Dict:
    """Run all ML models on a single text."""
    if not store.ready or store.tfidf is None:
        raise HTTPException(status_code=503, detail="Models not ready. Train first.")

    processed = store.pipeline.process(text)
    X = store.tfidf.transform([processed])

    results = {}
    for task in ["sentiment", "category", "urgency", "churn"]:
        if task in store.models:
            model = store.models[task]
            pred = int(model.predict(X)[0])
            # Use label encoder if valid, else fall back to hardcoded map
            le = store.encoders.get(task)
            if le is not None and hasattr(le, "classes_"):
                label = str(le.inverse_transform([pred])[0])
            else:
                label = LABEL_MAPS.get(task, {}).get(pred, str(pred))
            confidence = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                confidence = float(np.max(proba))
            results[task] = {"label": label, "confidence": confidence}
        else:
            results[task] = {"label": "unknown", "confidence": None}

    return results


# ── Routes ────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {
        "status": "healthy" if store.ready else "models_not_loaded",
        "models_loaded": list(store.models.keys()),
        "version": "1.0.0",
    }


@app.post("/token", response_model=TokenResponse, tags=["Auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Demo auth — replace with real user DB in production."""
    if form_data.username != "admin" or form_data.password != "password":
        raise HTTPException(status_code=401, detail="Incorrect credentials")
    from jose import jwt
    from src.utils.config import SECRET_KEY, ALGORITHM
    token = jwt.encode({"sub": form_data.username}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}


@app.post("/analyze", response_model=FeedbackResponse, tags=["Analysis"])
def analyze(request: FeedbackRequest):
    """
    Analyze a single customer feedback text.
    Returns sentiment, category, urgency, churn risk, ABSA, and entities.
    """
    start = time.perf_counter()

    predictions = run_inference(request.text)

    # ABSA
    aspect_sentiments = None
    if request.include_absa and store.absa:
        aspect_sentiments = store.absa.analyze(request.text)

    # NER
    entities = None
    if request.include_entities and store.pipeline and store.pipeline.nlp:
        entities = store.pipeline.extract_entities(request.text)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return FeedbackResponse(
        text=request.text,
        sentiment=predictions["sentiment"]["label"],
        sentiment_confidence=predictions["sentiment"]["confidence"],
        category=predictions["category"]["label"],
        category_confidence=predictions["category"]["confidence"],
        urgency=predictions["urgency"]["label"],
        urgency_confidence=predictions["urgency"]["confidence"],
        churn_risk=predictions["churn"]["label"],
        churn_confidence=predictions["churn"]["confidence"],
        aspect_sentiments=aspect_sentiments,
        entities=entities,
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.post("/batch", response_model=BatchResponse, tags=["Analysis"])
def batch_analyze(request: BatchRequest):
    """
    Analyze a batch of customer feedback texts (up to 500).
    Returns all predictions + aggregate summary.
    """
    if len(request.texts) > 500:
        raise HTTPException(status_code=400, detail="Max 500 texts per batch.")

    results = []
    for text in request.texts:
        single_req = FeedbackRequest(
            text=text,
            include_absa=request.include_absa,
            include_entities=False,
        )
        result = analyze(single_req)
        results.append(result)

    # Summary
    sentiments = [r.sentiment for r in results]
    categories = [r.category for r in results]
    urgencies = [r.urgency for r in results]
    churns = [r.churn_risk for r in results]

    summary = {
        "sentiment_distribution": dict(Counter(sentiments)),
        "category_distribution": dict(Counter(categories)),
        "urgency_distribution": dict(Counter(urgencies)),
        "churn_rate_pct": round(churns.count("1") / max(len(churns), 1) * 100, 2),
        "high_urgency_count": urgencies.count("high"),
        "avg_processing_time_ms": round(
            sum(r.processing_time_ms for r in results) / len(results), 2
        ),
    }

    return BatchResponse(total=len(results), results=results, summary=summary)


@app.get("/topics", tags=["Analysis"])
def get_topics():
    """
    Return topic modeling results from latest training run.
    (Pre-computed — run LDA in train.py to populate.)
    """
    topics_path = MODELS_DIR / "topics.json"
    if topics_path.exists():
        import json
        return json.loads(topics_path.read_text())
    return {"message": "Topics not yet computed. Run train.py first."}
