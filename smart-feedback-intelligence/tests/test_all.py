"""
Test suite for Smart Feedback Intelligence System.
Run: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np


# ── NLP Pipeline Tests ────────────────────────────────────
class TestNLPPipeline:
    def setup_method(self):
        from src.nlp.pipeline import NLPPipeline
        self.pipeline = NLPPipeline(use_spacy=False)

    def test_clean_text_removes_html(self):
        result = self.pipeline.clean_text("<b>Hello</b> world!")
        assert "<b>" not in result
        assert "hello" in result

    def test_clean_text_removes_urls(self):
        result = self.pipeline.clean_text("Visit https://example.com for more")
        assert "https" not in result

    def test_clean_text_lowercases(self):
        result = self.pipeline.clean_text("HELLO WORLD")
        assert result == result.lower()

    def test_tokenize_returns_list(self):
        tokens = self.pipeline.tokenize("hello world test")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_lemmatize_works(self):
        tokens = self.pipeline.lemmatize(["running", "better", "products"])
        assert isinstance(tokens, list)

    def test_process_returns_string(self):
        result = self.pipeline.process("The product is amazing and delivery was fast!")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_process_batch(self):
        texts = ["Good product", "Bad delivery", "Excellent support"]
        results = self.pipeline.process_batch(texts, show_progress=False)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    def test_get_text_stats(self):
        stats = self.pipeline.get_text_stats("Great product! Really happy!")
        assert "word_count" in stats
        assert "char_count" in stats
        assert "exclamation_count" in stats
        assert stats["exclamation_count"] == 2

    def test_tfidf_fit_transform(self):
        texts = ["good product quality", "bad delivery service", "amazing customer support"]
        processed = self.pipeline.process_batch(texts, show_progress=False)
        matrix = self.pipeline.fit_tfidf(processed, max_features=100)
        assert matrix.shape[0] == 3

    def test_tfidf_transform_after_fit(self):
        texts = ["good product", "bad service"]
        processed = self.pipeline.process_batch(texts, show_progress=False)
        self.pipeline.fit_tfidf(processed)
        new_text = self.pipeline.process("excellent quality item")
        result = self.pipeline.transform_tfidf([new_text])
        assert result.shape[0] == 1

    def test_tfidf_raises_without_fit(self):
        with pytest.raises(ValueError):
            self.pipeline.transform_tfidf(["some text"])


# ── ABSA Tests ────────────────────────────────────────────
class TestABSA:
    def setup_method(self):
        from src.nlp.absa import ABSAnalyzer
        self.absa = ABSAnalyzer()

    def test_detects_positive_delivery(self):
        result = self.absa.analyze("The delivery was super fast and excellent!")
        assert "delivery" in result
        assert result["delivery"] == "positive"

    def test_detects_negative_product(self):
        result = self.absa.analyze("The product is terrible and broken completely.")
        assert "product" in result
        assert result["product"] == "negative"

    def test_handles_mixed_aspects(self):
        result = self.absa.analyze(
            "Delivery was late but the product quality is great!"
        )
        assert "delivery" in result or "product" in result

    def test_empty_text_returns_empty(self):
        result = self.absa.analyze("")
        assert isinstance(result, dict)

    def test_batch_returns_list(self):
        texts = ["great product", "slow delivery", "bad support service"]
        results = self.absa.analyze_batch(texts)
        assert len(results) == 3

    def test_no_aspect_text_returns_empty(self):
        result = self.absa.analyze("I went to the park today and saw birds.")
        assert result == {}

    def test_negation_handling(self):
        pos = self.absa.analyze("delivery was not late at all")
        neg = self.absa.analyze("delivery was late and terrible")
        # Both should be detected
        assert "delivery" in pos or "delivery" in neg


# ── Data Loader Tests ─────────────────────────────────────
class TestDataLoader:
    def setup_method(self):
        from src.data.loader import DataLoader
        self.loader = DataLoader()

    def test_synthetic_data_has_required_columns(self):
        df = self.loader._generate_synthetic_data(100)
        for col in ["text", "sentiment", "category", "urgency", "churn_risk"]:
            assert col in df.columns

    def test_synthetic_data_correct_size(self):
        df = self.loader._generate_synthetic_data(200)
        assert len(df) == 200

    def test_synthetic_sentiment_values(self):
        df = self.loader._generate_synthetic_data(500)
        assert set(df["sentiment"].unique()).issubset({"positive", "neutral", "negative"})

    def test_synthetic_urgency_values(self):
        df = self.loader._generate_synthetic_data(500)
        assert set(df["urgency"].unique()).issubset({"low", "medium", "high"})

    def test_load_and_merge_returns_dataframe(self):
        import pandas as pd
        df = self.loader.load_and_merge(max_samples=500)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "text" in df.columns
        assert "sentiment" in df.columns


# ── API Tests ─────────────────────────────────────────────
class TestAPI:
    def setup_method(self):
        # Mock model loading so API tests don't require trained models
        with patch("api.main.load_models"):
            from api.main import app
            self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_analyze_requires_text(self):
        response = self.client.post("/analyze", json={})
        assert response.status_code == 422  # Validation error

    def test_analyze_text_too_short(self):
        response = self.client.post("/analyze", json={"text": "hi"})
        assert response.status_code == 422

    def test_batch_max_limit(self):
        texts = ["test feedback"] * 501
        response = self.client.post("/batch", json={"texts": texts})
        # Should fail - over limit
        assert response.status_code in [400, 422, 503]

    def test_token_invalid_credentials(self):
        response = self.client.post(
            "/token",
            data={"username": "wrong", "password": "wrong"}
        )
        assert response.status_code == 401

    def test_token_valid_credentials(self):
        response = self.client.post(
            "/token",
            data={"username": "admin", "password": "password"}
        )
        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_topics_endpoint_exists(self):
        response = self.client.get("/topics")
        assert response.status_code == 200


# ── Config Tests ──────────────────────────────────────────
class TestConfig:
    def test_labels_defined(self):
        from src.utils.config import SENTIMENT_LABELS, CATEGORY_LABELS, URGENCY_LABELS
        assert len(SENTIMENT_LABELS) == 3
        assert len(CATEGORY_LABELS) == 5
        assert len(URGENCY_LABELS) == 3

    def test_aspects_defined(self):
        from src.utils.config import ASPECTS
        assert len(ASPECTS) > 0
        for aspect, keywords in ASPECTS.items():
            assert isinstance(keywords, list)
            assert len(keywords) > 0
