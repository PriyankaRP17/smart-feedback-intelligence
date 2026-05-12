# Smart Customer Feedback Intelligence System

> End-to-end ML + NLP system that analyzes customer feedback — classifies sentiment (aspect-level), predicts urgency and churn risk. Built with classical ML, BERT fine-tuning, FastAPI, React, Docker, and deployed on Render.

---

## Project Structure

```
smart-feedback-intelligence/
├── src/
│   ├── data/
│   │   ├── loader.py           # Dataset loading (Amazon Reviews / synthetic)
│   │   └── preprocessor.py     # EDA, feature engineering, LDA topic modeling
│   ├── nlp/
│   │   ├── pipeline.py         # Full NLP pipeline: clean → tokenize → lemmatize → TF-IDF → BERT
│   │   └── absa.py             # Aspect-Based Sentiment Analysis
│   ├── models/
│   │   ├── classical.py        # LR, SVM, XGBoost, RandomForest + MLflow tracking
│   │   └── bert_classifier.py  # BERT fine-tuning for sentiment + category
│   ├── training/
│   │   └── trainer.py          # Model evaluator + comparison charts
│   └── utils/
│       └── config.py           # All config and constants
├── api/
│   ├── main.py                 # FastAPI app — all endpoints
│   └── schemas/
│       └── models.py           # Pydantic request/response schemas
├── frontend/                   # React + Vite UI
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   └── Sidebar.jsx
│   │   └── pages/
│   │       ├── SingleAnalysis.jsx
│   │       ├── BatchAnalysis.jsx
│   │       └── Dashboard.jsx
│   ├── package.json
│   └── vite.config.js
├── saved_models/               # Trained model .pkl files (after running train.py)
├── train.py                    # Full training pipeline orchestrator
├── inference.py                # Quick model verification script
├── conftest.py                 # pytest configuration
├── Dockerfile                  # Multi-stage Docker build for API
├── Dockerfile.frontend         # React → nginx production build
├── docker-compose.yml          # API + React UI + MLflow together
├── nginx.conf                  # Nginx config for React SPA
├── render.yaml                 # Render deployment config
├── pytest.ini                  # Test configuration
└── requirements.txt
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| NLP | spaCy, NLTK, TF-IDF, Sentence-BERT, ABSA |
| Classical ML | scikit-learn (LR, SVM, XGBoost, RandomForest), GridSearchCV |
| Deep Learning | HuggingFace Transformers — BERT (architecture ready, GPU required to train) |
| Experiment Tracking | MLflow (experiments + model registry) |
| Backend | FastAPI + Pydantic + Uvicorn + JWT Auth |
| Frontend | React + Vite + Recharts (dark theme UI) |
| Containerize | Docker + Docker Compose + Nginx |
| Deploy | Render (API) + Vercel (React UI) |
| Dataset | Amazon Reviews (HuggingFace) + synthetic fallback |

---

## Quick Start (Local)

### 1. Prerequisites
- Python 3.11
- Node.js 20+

### 2. Setup environment
```powershell
# Create and activate virtual environment
python -m venv env
.\env\Scripts\activate        # Windows
# source env/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Copy env file
copy .env.example .env
```

### 3. Train all models
```powershell
python train.py
```
This will:
- Load Amazon Reviews dataset (auto-downloads ~1GB from HuggingFace)
- Run full NLP preprocessing pipeline
- Train LR, SVM, XGBoost, RandomForest — compare with F1/ROC-AUC
- Run Aspect-Based Sentiment Analysis
- Tune XGBoost with GridSearchCV
- Log all experiments to MLflow
- Save best models to `saved_models/`

> **BERT fine-tuning was NOT run** — CPU is too slow (779MB model, 25+ hrs on CPU).
> Train BERT separately on **Google Colab** (free GPU) and drop the saved model into `saved_models/bert_sentiment/`.
> Run on Google Colab with GPU for BERT training.

### 4. Verify models work
```powershell
python inference.py
```

### 5. View MLflow experiments
```powershell
mlflow ui
# Open: http://localhost:5000
```

### 6. Start FastAPI backend
```powershell
uvicorn api.main:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

### 7. Start React frontend
```powershell
cd frontend
npm install
npm run dev
# Open: http://localhost:5173
```

### 8. Run tests
```powershell
pytest tests/ -v
```

---

## Docker

```powershell
# Build and run everything
docker-compose up --build

# Services:
# React UI  → http://localhost:80
# FastAPI   → http://localhost:8000/docs
# MLflow    → http://localhost:5000
```

> First build takes 15-20 mins — downloads all packages into the container.
> Subsequent runs use cache and start in seconds.

---

## API Endpoints

### `POST /analyze` — Single feedback analysis
```json
Request:
{
  "text": "Delivery was late but product quality is amazing!",
  "include_absa": true,
  "include_entities": true
}

Response:
{
  "sentiment": "negative",
  "sentiment_confidence": 0.86,
  "category": "delivery",
  "category_confidence": 0.27,
  "urgency": "medium",
  "urgency_confidence": 0.64,
  "churn_risk": "at_risk",
  "churn_confidence": 0.86,
  "aspect_sentiments": {
    "delivery": "negative",
    "product": "positive"
  },
  "entities": {"DATE": ["3 days"]},
  "processing_time_ms": 4.5
}
```

### `POST /batch` — Bulk analysis (up to 500 texts)
```json
Request: {"texts": ["...", "..."], "include_absa": false}
Response: {
  "total": 2,
  "results": [...],
  "summary": {
    "sentiment_distribution": {"negative": 1, "positive": 1},
    "high_urgency_count": 0,
    "churn_rate_pct": 50.0
  }
}
```

### `GET /health` — Health check
### `GET /topics` — LDA topic modeling results
### `POST /token` — JWT authentication (username: admin, password: password)

---

## React UI Pages

| Page | Description |
|---|---|
| **Single Analysis** | Paste any review → live sentiment, ABSA, entities, confidence bars |
| **Batch Analysis** | Paste multiple reviews → table + summary charts + CSV download |
| **Dashboard** | Aggregate analytics — sentiment distribution, category breakdown, ABSA trends |

---

## ML Models Trained & Compared

All 4 models were trained and evaluated for every task. Best model selected by F1 macro score.

| Task | Models Compared | Best Model | F1 Score |
|---|---|---|---|
| Sentiment | LR, SVM, XGBoost, RandomForest | Logistic Regression | 0.8635 |
| Category | LR, SVM, XGBoost, RandomForest | Logistic Regression | 0.2050* |
| Urgency | LR, SVM, XGBoost, RandomForest | Logistic Regression | 0.5752 |
| Churn Risk | LR, SVM, XGBoost, RandomForest | Logistic Regression | 0.8635 |

**All experiments logged in MLflow** — metrics, params, and model artifacts tracked for every run.

> *Category F1 is low because Amazon Reviews has no category labels — randomly assigned for demo. Use a labeled dataset for production.

> **Why LR won?** Logistic Regression performs strongly on high-dimensional TF-IDF features (10,000 features). XGBoost and RandomForest are better suited for tabular/numerical data. This is exactly the kind of insight interviewers look for.

---

## Deploy on Render

```bash
# 1. Push to GitHub
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/YOUR_USERNAME/smart-feedback-intelligence.git
git push -u origin main

# 2. Go to render.com → New → Blueprint
# 3. Connect your GitHub repo
# 4. Render reads render.yaml automatically
# 5. Two services deploy: API + Static React frontend
```

> Upload `saved_models/` to Render disk after first deploy via Render Shell.

---

## What Makes This Job-Ready

- **4 ML models trained and compared** — proper evaluation with F1, ROC-AUC, confusion matrix
- **Full NLP pipeline from scratch** — tokenization, lemmatization, TF-IDF, embeddings
- **Aspect-Based Sentiment (ABSA)** — what most portfolios completely skip
- **BERT fine-tuning** — architecture built, skipped on CPU (train on Google Colab with GPU)
- **MLflow experiment tracking** — every model, metric, and param logged
- **FastAPI + JWT Auth** — production-grade API with security
- **React UI** — professional dark-theme dashboard, not just a notebook
- **Docker + docker-compose** — full containerization
- **Render deployment** — live public URL for portfolio
- **Batch processing** — real enterprise use case
- **GitHub Actions CI/CD** — automated testing and deployment

---

## Author
Priyanka R P — [LinkedIn](https://www.linkedin.com/in/priyanka-rp/) | [GitHub](https://github.com/PriyankaRP17)
