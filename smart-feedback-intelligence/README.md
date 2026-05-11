# 🧠 Smart Customer Feedback Intelligence System

> End-to-end ML + NLP system that analyzes customer feedback — classifies sentiment (aspect-level), predicts urgency and churn risk. Built with BERT fine-tuning, XGBoost, FastAPI, Docker, and deployed on Render.

---

## 📁 Project Structure

```
smart-feedback-intelligence/
├── src/
│   ├── data/
│   │   └── loader.py           # Dataset loading (Amazon Reviews / Twitter / synthetic)
│   ├── nlp/
│   │   ├── pipeline.py         # Full NLP pipeline: clean → tokenize → lemmatize → TF-IDF → BERT
│   │   └── absa.py             # Aspect-Based Sentiment Analysis
│   ├── models/
│   │   ├── classical.py        # LR, SVM, XGBoost + MLflow tracking
│   │   └── bert_classifier.py  # BERT fine-tuning for sentiment + category
│   └── utils/
│       └── config.py           # All config and constants
├── api/
│   ├── main.py                 # FastAPI app — all endpoints
│   └── schemas/
│       └── models.py           # Pydantic request/response schemas
├── frontend/
│   └── app.py                  # Streamlit UI
├── train.py                    # 🚀 Full training pipeline orchestrator
├── Dockerfile
├── docker-compose.yml
├── render.yaml
└── requirements.txt
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| NLP | spaCy, NLTK, TF-IDF, Word2Vec, Sentence-BERT |
| Classical ML | scikit-learn (LR, SVM, XGBoost), GridSearchCV |
| Fine-tune | HuggingFace Transformers — BERT |
| Tracking | MLflow (experiments + model registry) |
| Backend | FastAPI + Pydantic + Uvicorn |
| Frontend | Streamlit |
| Containerize | Docker + Docker Compose |
| Deploy | Render (API) + Vercel (UI) |

---

## ⚡ Quick Start

### 1. Clone and setup environment
```bash
git clone https://github.com/yourusername/smart-feedback-intelligence
cd smart-feedback-intelligence
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Train all models
```bash
python train.py
```
This will:
- Load Amazon Reviews dataset (auto-downloads from HuggingFace)
- Run full NLP preprocessing pipeline
- Train LR, SVM, XGBoost — compare all with F1/ROC-AUC
- Fine-tune BERT for sentiment classification
- Log everything to MLflow
- Save best models to `saved_models/`

### 4. View MLflow experiments
```bash
mlflow ui
# Open: http://localhost:5000
```

### 5. Start FastAPI backend
```bash
uvicorn api.main:app --reload
# Docs: http://localhost:8000/docs
```

### 6. Start Streamlit frontend
```bash
streamlit run frontend/app.py
# Open: http://localhost:8501
```

---

## 🐳 Docker

```bash
# Build and run everything
docker-compose up --build

# Services:
# API:      http://localhost:8000
# UI:       http://localhost:8501
# MLflow:   http://localhost:5000
```

---

## 📡 API Endpoints

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
  "sentiment": "positive",
  "sentiment_confidence": 0.82,
  "category": "delivery",
  "urgency": "medium",
  "churn_risk": "0",
  "aspect_sentiments": {
    "delivery": "negative",
    "product": "positive"
  },
  "entities": {"ORG": ["Amazon"]},
  "processing_time_ms": 12.4
}
```

### `POST /batch` — Bulk analysis (up to 500 texts)
```json
Request: {"texts": ["...", "..."], "include_absa": false}
Response: {"total": 2, "results": [...], "summary": {...}}
```

### `GET /health` — Health check
### `GET /topics` — Topic modeling results
### `POST /token` — JWT authentication

---

## 🚀 Deploy on Render

1. Push to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml`
5. Click **Deploy**

---

## 📊 Models Trained

| Task | Algorithm | Metric |
|---|---|---|
| Sentiment | LR → SVM → XGBoost → BERT | F1 macro |
| Category | LR → SVM → XGBoost | F1 macro |
| Urgency | XGBoost (text + metadata) | F1 macro |
| Churn Risk | Logistic Regression | ROC-AUC |

---

## 🎯 What Makes This Job-Ready

- ✅ **4 ML models trained from scratch** — not just wrappers
- ✅ **Full NLP pipeline** — tokenization, lemmatization, TF-IDF, embeddings
- ✅ **Aspect-Based Sentiment** — what most portfolios skip
- ✅ **BERT fine-tuning** — compared vs classical ML with metrics
- ✅ **MLflow tracking** — every experiment logged
- ✅ **FastAPI + Docker + Render** — production deployment
- ✅ **Batch processing** — real enterprise use case
- ✅ **JWT Auth** — security awareness

---

## 👤 Author
Your Name — [LinkedIn](https://linkedin.com) | [GitHub](https://github.com)
