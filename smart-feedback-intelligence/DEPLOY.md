# 🚀 Deployment Guide

## Option A — Docker (Local / VPS)

### Prerequisites
- Docker Desktop installed
- Trained models in `saved_models/` folder

### Run everything with one command
```bash
docker-compose up --build
```

### Services start at:
| Service  | URL                        |
|----------|----------------------------|
| React UI | http://localhost:80         |
| FastAPI  | http://localhost:8000/docs  |
| MLflow   | http://localhost:5000       |

### Run API only
```bash
docker-compose up api
```

### Rebuild after code changes
```bash
docker-compose up --build api
```

### Stop everything
```bash
docker-compose down
```

---

## Option B — Render (Free Cloud Deployment)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/smart-feedback-intelligence.git
git push -u origin main
```

### Step 2 — Deploy API on Render
1. Go to [render.com](https://render.com) → **New** → **Web Service**
2. Connect your GitHub repo
3. Render auto-detects `render.yaml` → click **Apply**
4. Wait for build (~5 mins)
5. Copy your API URL: `https://feedback-intelligence-api.onrender.com`

### Step 3 — Upload trained models
Since Render's free plan has no persistent disk on first run:
```bash
# Option 1: Upload via Render Shell
# Go to Render dashboard → your service → Shell
# Then upload saved_models/ files

# Option 2: Add model upload step to buildCommand in render.yaml
# (store models in cloud storage like S3 or HuggingFace Hub)
```

### Step 4 — Deploy Frontend on Render
1. Go to Render → **New** → **Static Site**
2. Connect same GitHub repo
3. Set **Root Directory**: `frontend`
4. Set **Build Command**: `npm ci && npm run build`
5. Set **Publish Directory**: `dist`
6. Add env var: `VITE_API_URL` = your API URL from Step 2
7. Click **Deploy**

### Step 5 — Update CORS (if needed)
In `api/main.py`, update `allow_origins` to your frontend URL:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.onrender.com"],
    ...
)
```

---

## Environment Variables

| Variable              | Default              | Description                    |
|-----------------------|----------------------|--------------------------------|
| `SECRET_KEY`          | auto-generated       | JWT secret key                 |
| `MLFLOW_TRACKING_URI` | `./mlruns`           | MLflow tracking location       |
| `API_HOST`            | `0.0.0.0`            | API bind host                  |
| `API_PORT`            | `8000`               | API port                       |
| `VITE_API_URL`        | `http://localhost:8000` | React → API URL             |

---

## Architecture

```
┌─────────────────┐     HTTP      ┌──────────────────┐
│   React UI      │ ──────────▶  │   FastAPI API    │
│  (nginx :80)    │              │   (:8000)        │
└─────────────────┘              └────────┬─────────┘
                                          │
                              ┌───────────┴──────────┐
                              │                      │
                    ┌─────────▼──────┐    ┌──────────▼──────┐
                    │  saved_models/ │    │    mlruns/       │
                    │  (disk volume) │    │  (MLflow data)   │
                    └────────────────┘    └─────────────────┘
```
