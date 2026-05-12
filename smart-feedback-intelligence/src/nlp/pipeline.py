import re
import string
import nltk
import spacy
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
for resource in ["stopwords", "wordnet", "punkt", "averaged_perceptron_tagger"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class NLPPipeline:
    """
    Full NLP preprocessing pipeline:
    Raw Text → Clean → Tokenize → Lemmatize → Features (TF-IDF + BERT embeddings)
    """

    def __init__(self, use_spacy: bool = True):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        # Keep negations — important for sentiment
        self.stop_words -= {"no", "not", "never", "nor", "neither", "nobody", "nothing"}
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.sentence_model = None  # Lazy loaded

        # Load spaCy for NER
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                self.nlp = None
        else:
            self.nlp = None

    # ── Step 1: Clean ────────────────────────────────────
    def clean_text(self, text: str) -> str:
        """Remove noise, HTML, URLs, special chars."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"<[^>]+>", " ", text)          # HTML tags
        text = re.sub(r"http\S+|www\S+", " ", text)   # URLs
        text = re.sub(r"@\w+|#\w+", " ", text)        # mentions/hashtags
        text = re.sub(r"\d+", " NUM ", text)           # numbers → token
        text = re.sub(r"[^\w\s]", " ", text)           # punctuation
        text = re.sub(r"\s+", " ", text).strip()       # extra spaces
        return text

    # ── Step 2: Tokenize ────────────────────────────────
    def tokenize(self, text: str) -> List[str]:
        """Word tokenization."""
        return word_tokenize(text)

    # ── Step 3: Remove stopwords ─────────────────────────
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in self.stop_words and len(t) > 2]

    # ── Step 4: Lemmatize ────────────────────────────────
    def lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    # ── Full pipeline for a single text ──────────────────
    def process(self, text: str) -> str:
        """Run full pipeline → return clean processed string."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return " ".join(tokens)

    # ── Batch process ────────────────────────────────────
    def process_batch(self, texts: List[str], show_progress: bool = True) -> List[str]:
        """Process a list of texts."""
        from tqdm import tqdm
        iterator = tqdm(texts, desc="NLP Pipeline") if show_progress else texts
        return [self.process(t) for t in iterator]

    # ── TF-IDF Features ──────────────────────────────────
    def fit_tfidf(self, texts: List[str], max_features: int = 10000) -> np.ndarray:
        """Fit TF-IDF and transform texts."""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        matrix = self.tfidf_vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF fitted: {matrix.shape[1]} features")
        return matrix

    def transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """Transform texts using fitted TF-IDF."""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF not fitted yet. Call fit_tfidf first.")
        return self.tfidf_vectorizer.transform(texts)

    # ── Sentence-BERT Embeddings ──────────────────────────
    def get_bert_embeddings(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Generate Sentence-BERT embeddings."""
        if self.sentence_model is None:
            logger.info("Loading Sentence-BERT model...")
            from sentence_transformers import SentenceTransformer  # lazy import
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.sentence_model.encode(
            texts, batch_size=batch_size, show_progress_bar=True
        )
        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    # ── Named Entity Recognition ──────────────────────────
    def extract_entities(self, text: str) -> dict:
        """Extract named entities using spaCy."""
        if self.nlp is None:
            return {}
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            label = ent.label_
            if label not in entities:
                entities[label] = []
            entities[label].append(ent.text)
        return entities

    # ── Text Statistics ───────────────────────────────────
    def get_text_stats(self, text: str) -> dict:
        """Extract hand-crafted features from text."""
        tokens = word_tokenize(text.lower())
        return {
            "char_count": len(text),
            "word_count": len(tokens),
            "avg_word_length": np.mean([len(w) for w in tokens]) if tokens else 0,
            "exclamation_count": text.count("!"),
            "question_count": text.count("?"),
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
            "has_negation": int(any(w in text.lower() for w in ["not", "no", "never", "didn't"])),
        }

    def get_stats_dataframe(self, texts: List[str]) -> pd.DataFrame:
        """Get text stats for all texts as a DataFrame."""
        return pd.DataFrame([self.get_text_stats(t) for t in texts])
