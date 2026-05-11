import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

# ── MLflow ───────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(BASE_DIR / "mlruns"))
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "smart-feedback-intelligence")

# ── Model Config ─────────────────────────────────────────
BERT_MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# ── Labels ───────────────────────────────────────────────
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
CATEGORY_LABELS = ["billing", "delivery", "product", "support", "returns"]
URGENCY_LABELS = ["low", "medium", "high"]

# ── API ──────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# ── ABSA Aspects ─────────────────────────────────────────
ASPECTS = {
    "delivery": ["delivery", "shipping", "arrived", "late", "package", "courier"],
    "product": ["product", "quality", "item", "defective", "broken", "design"],
    "support": ["support", "service", "agent", "helpful", "rude", "resolved"],
    "billing": ["price", "charge", "refund", "billing", "payment", "expensive"],
    "returns": ["return", "refund", "exchange", "policy", "difficult", "easy"]
}
