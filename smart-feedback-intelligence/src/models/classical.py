import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score,
    confusion_matrix, accuracy_score
)
from xgboost import XGBClassifier
import logging

from src.utils.config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    SENTIMENT_LABELS, CATEGORY_LABELS, URGENCY_LABELS, MODELS_DIR
)

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


class ModelTrainer:
    """
    Train, compare, tune, and log classical ML models with MLflow.
    Covers: Sentiment · Category · Urgency · Churn Risk
    """

    def __init__(self):
        self.models = {
            "logistic_regression": LogisticRegression(
                max_iter=1000, C=1.0, class_weight="balanced"
            ),
            "linear_svc": LinearSVC(
                max_iter=2000, C=1.0, class_weight="balanced"
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42
            ),
            "xgboost": XGBClassifier(
                n_estimators=200, learning_rate=0.1,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=42
            ),
        }
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.best_models: Dict[str, Any] = {}

    def _encode_labels(self, task: str, labels: list) -> np.ndarray:
        le = LabelEncoder()
        encoded = le.fit_transform(labels)
        self.label_encoders[task] = le
        return encoded

    def _evaluate(self, model, X_test, y_test, label_names: list) -> Dict:
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        }
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                metrics["roc_auc"] = roc_auc_score(
                    y_test, y_prob, multi_class="ovr", average="macro"
                )
        except Exception:
            pass
        # Use only labels actually present in the data to avoid mismatch
        unique_classes = sorted(set(y_test))
        actual_names = [label_names[i] for i in unique_classes if i < len(label_names)]
        logger.info(f"\n{classification_report(y_test, y_pred, target_names=actual_names)}")
        return metrics

    def train_all_models(
        self,
        X_train, X_test,
        y_train_dict: Dict[str, np.ndarray],
        y_test_dict: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Train all models for all tasks and log to MLflow."""
        results = {}
        tasks = {
            "sentiment": SENTIMENT_LABELS,
            "category": CATEGORY_LABELS,
            "urgency": URGENCY_LABELS,
            "churn": ["no_churn", "churn"],
        }

        for task, label_names in tasks.items():
            if task not in y_train_dict:
                continue
            logger.info(f"\n{'='*50}\nTraining task: {task.upper()}\n{'='*50}")
            best_f1 = 0
            best_model_name = None
            for model_name, model in self.models.items():
                with mlflow.start_run(run_name=f"{task}_{model_name}"):
                    mlflow.log_param("task", task)
                    mlflow.log_param("model", model_name)
                    mlflow.log_param("train_size", X_train.shape[0])
                    try:
                        model.fit(X_train, y_train_dict[task])
                        metrics = self._evaluate(
                            model, X_test, y_test_dict[task], label_names
                        )
                        mlflow.log_metrics(metrics)
                        mlflow.sklearn.log_model(model, f"{task}_{model_name}")
                        logger.info(
                            f"[{task}] {model_name}: F1={metrics['f1_macro']:.4f}"
                        )
                        if metrics["f1_macro"] > best_f1:
                            best_f1 = metrics["f1_macro"]
                            best_model_name = model_name
                            self.best_models[task] = model
                    except Exception as e:
                        logger.error(f"Error training {model_name} for {task}: {e}")
                        mlflow.log_param("error", str(e))

            logger.info(f"✅ Best model for [{task}]: {best_model_name} (F1={best_f1:.4f})")
            results[task] = {"best_model": best_model_name, "best_f1": best_f1}

            # Save best model to disk
            save_path = MODELS_DIR / f"{task}_best_model.pkl"
            joblib.dump(self.best_models[task], save_path)
            joblib.dump(self.label_encoders.get(task), MODELS_DIR / f"{task}_label_encoder.pkl")
            logger.info(f"💾 Saved: {save_path}")

        return results

    def tune_xgboost(self, task: str, X_train, y_train) -> XGBClassifier:
        """Hyperparameter tuning for XGBoost using GridSearchCV."""
        logger.info(f"Tuning XGBoost for {task}...")
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
        }
        xgb = XGBClassifier(
            use_label_encoder=False, eval_metric="mlogloss", random_state=42
        )
        grid_search = GridSearchCV(
            xgb, param_grid, cv=3, scoring="f1_macro", n_jobs=-1, verbose=1
        )
        with mlflow.start_run(run_name=f"{task}_xgboost_tuned"):
            grid_search.fit(X_train, y_train)
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("best_cv_f1", grid_search.best_score_)
            logger.info(f"Best params: {grid_search.best_params_}")
            logger.info(f"Best CV F1: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_

    def load_models(self) -> bool:
        """Load saved models from disk."""
        tasks = ["sentiment", "category", "urgency", "churn"]
        for task in tasks:
            model_path = MODELS_DIR / f"{task}_best_model.pkl"
            encoder_path = MODELS_DIR / f"{task}_label_encoder.pkl"
            if model_path.exists():
                self.best_models[task] = joblib.load(model_path)
                if encoder_path.exists():
                    self.label_encoders[task] = joblib.load(encoder_path)
                logger.info(f"✅ Loaded: {task} model")
            else:
                logger.warning(f"⚠️ Model not found: {model_path}")
                return False
        return True

    def predict(self, task: str, X) -> Dict[str, Any]:
        """Run prediction for a task."""
        if task not in self.best_models:
            raise ValueError(f"Model for task '{task}' not loaded.")
        model = self.best_models[task]
        pred = model.predict(X)
        le = self.label_encoders.get(task)
        label = le.inverse_transform(pred)[0] if le else str(pred[0])
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            confidence = float(np.max(proba))
        return {"label": label, "confidence": confidence}
