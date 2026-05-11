"""
trainer.py — Model evaluation utilities and comparison report generator.
Used after training to produce a summary of all model performances.
"""

import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score, accuracy_score,
    precision_recall_curve, roc_curve
)
from src.utils.config import MODELS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates and compares trained models across all tasks.
    Generates comparison charts and a JSON report.
    """

    def __init__(self):
        self.results: Dict[str, Any] = {}

    def evaluate_model(
        self,
        model,
        X_test,
        y_test: np.ndarray,
        task: str,
        model_name: str,
        label_names: List[str],
    ) -> Dict:
        """Full evaluation of a single model on a task."""
        y_pred = model.predict(X_test)

        metrics = {
            "task": task,
            "model": model_name,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
            "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
        }

        # ROC-AUC (if probabilistic model)
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
                metrics["roc_auc"] = round(
                    roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"), 4
                )
            except Exception:
                metrics["roc_auc"] = None

        # Per-class F1
        per_class = f1_score(y_test, y_pred, average=None)
        for i, label in enumerate(label_names):
            if i < len(per_class):
                metrics[f"f1_{label}"] = round(float(per_class[i]), 4)

        return metrics

    def generate_confusion_matrix_plot(
        self,
        model,
        X_test,
        y_test: np.ndarray,
        task: str,
        label_names: List[str],
        save_dir: Path = PROCESSED_DIR,
    ):
        """Plot and save confusion matrix."""
        save_dir.mkdir(exist_ok=True)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names, ax=ax
        )
        ax.set_title(f"Confusion Matrix — {task.title()}", fontsize=14)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()

        path = save_dir / f"confusion_matrix_{task}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✅ Confusion matrix saved: {path}")
        return path

    def generate_model_comparison_chart(
        self,
        comparison_df: pd.DataFrame,
        save_dir: Path = PROCESSED_DIR,
    ):
        """Bar chart comparing all models across tasks."""
        save_dir.mkdir(exist_ok=True)
        tasks = comparison_df["task"].unique()
        fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5))
        if len(tasks) == 1:
            axes = [axes]

        colors = ["#7F77DD", "#1D9E75", "#D85A30", "#BA7517"]

        for ax, task in zip(axes, tasks):
            task_df = comparison_df[comparison_df["task"] == task].sort_values("f1_macro")
            bars = ax.barh(
                task_df["model"], task_df["f1_macro"],
                color=colors[:len(task_df)], edgecolor="white", height=0.5
            )
            ax.set_title(f"{task.title()} — F1 Macro", fontweight="bold")
            ax.set_xlim(0, 1)
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
            ax.set_xlabel("F1 Macro Score")

        plt.suptitle("Model Comparison Across All Tasks", fontsize=14, fontweight="bold")
        plt.tight_layout()

        path = save_dir / "model_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✅ Comparison chart saved: {path}")
        return path

    def save_report(self, results: List[Dict], save_path: Path = None):
        """Save evaluation results to JSON."""
        if save_path is None:
            save_path = MODELS_DIR / "evaluation_report.json"
        MODELS_DIR.mkdir(exist_ok=True)

        report = {
            "total_models_evaluated": len(results),
            "results": results,
            "best_per_task": {},
        }

        df = pd.DataFrame(results)
        if not df.empty and "task" in df.columns:
            for task in df["task"].unique():
                task_df = df[df["task"] == task]
                best = task_df.loc[task_df["f1_macro"].idxmax()]
                report["best_per_task"][task] = {
                    "model": best["model"],
                    "f1_macro": best["f1_macro"],
                }

        save_path.write_text(json.dumps(report, indent=2))
        logger.info(f"✅ Evaluation report saved: {save_path}")
        return report

    def print_summary(self, report: Dict):
        """Print a clean summary table."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 MODEL EVALUATION SUMMARY")
        logger.info("=" * 60)
        for task, info in report.get("best_per_task", {}).items():
            logger.info(
                f"  {task.upper():<15} Best: {info['model']:<25} F1={info['f1_macro']:.4f}"
            )
        logger.info("=" * 60)
