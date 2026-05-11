import torch
import numpy as np
import mlflow
from pathlib import Path
from typing import List, Dict, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import logging

from src.utils.config import (
    BERT_MODEL_NAME, MAX_SEQ_LENGTH, BATCH_SIZE,
    EPOCHS, LEARNING_RATE, MODELS_DIR,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
)

logger = logging.getLogger(__name__)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class FeedbackDataset(Dataset):
    """PyTorch Dataset for BERT fine-tuning."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BERTClassifier:
    """
    Fine-tune BERT for sequence classification.
    Supports: sentiment (3-class) and category (5-class).
    """

    def __init__(self, num_labels: int, task_name: str = "sentiment"):
        self.num_labels = num_labels
        self.task_name = task_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.model: Optional[BertForSequenceClassification] = None

    def build_model(self):
        """Initialize BERT model for classification."""
        self.model = BertForSequenceClassification.from_pretrained(
            BERT_MODEL_NAME,
            num_labels=self.num_labels,
            output_attentions=False,
            output_hidden_states=False,
        )
        self.model.to(self.device)
        logger.info(f"BERT model loaded: {BERT_MODEL_NAME} → {self.num_labels} labels")

    def prepare_data(
        self,
        train_texts, train_labels,
        val_texts, val_labels,
    ) -> tuple:
        """Create DataLoaders from texts and labels."""
        train_dataset = FeedbackDataset(
            train_texts, train_labels, self.tokenizer, MAX_SEQ_LENGTH
        )
        val_dataset = FeedbackDataset(
            val_texts, val_labels, self.tokenizer, MAX_SEQ_LENGTH
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        return train_loader, val_loader

    def train(self, train_loader, val_loader, label_names: List[str]) -> Dict:
        """Fine-tune BERT with MLflow tracking."""
        if self.model is None:
            self.build_model()

        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE, eps=1e-8)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        best_val_f1 = 0
        history = []

        with mlflow.start_run(run_name=f"bert_finetune_{self.task_name}"):
            mlflow.log_params({
                "model": BERT_MODEL_NAME,
                "task": self.task_name,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LEARNING_RATE,
                "max_seq_len": MAX_SEQ_LENGTH,
                "num_labels": self.num_labels,
            })

            for epoch in range(EPOCHS):
                # ── Training ──────────────────────────────
                self.model.train()
                total_loss = 0
                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    self.model.zero_grad()
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    total_loss += loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                avg_train_loss = total_loss / len(train_loader)

                # ── Validation ────────────────────────────
                self.model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        outputs = self.model(**batch)
                        preds = torch.argmax(outputs.logits, dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(batch["labels"].cpu().numpy())

                val_f1 = f1_score(all_labels, all_preds, average="macro")
                logger.info(
                    f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val F1={val_f1:.4f}"
                )
                mlflow.log_metrics({
                    "train_loss": avg_train_loss,
                    "val_f1_macro": val_f1,
                }, step=epoch)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    self.save_model()
                    logger.info(f"✅ New best model saved (F1={val_f1:.4f})")

                history.append({"epoch": epoch+1, "train_loss": avg_train_loss, "val_f1": val_f1})

            logger.info(f"\n{classification_report(all_labels, all_preds, target_names=label_names)}")
            mlflow.log_metric("best_val_f1", best_val_f1)

        return {"best_val_f1": best_val_f1, "history": history}

    def predict(self, texts: List[str]) -> List[Dict]:
        """Run inference on a list of texts."""
        if self.model is None:
            self.load_model()
        self.model.eval()
        results = []
        for text in texts:
            encoding = self.tokenizer(
                text, max_length=MAX_SEQ_LENGTH,
                padding="max_length", truncation=True, return_tensors="pt"
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            with torch.no_grad():
                outputs = self.model(**encoding)
                probs = torch.softmax(outputs.logits, dim=1)[0]
                pred_class = torch.argmax(probs).item()
                confidence = probs[pred_class].item()
            results.append({"class_id": pred_class, "confidence": confidence, "probs": probs.cpu().numpy().tolist()})
        return results

    def save_model(self):
        """Save model and tokenizer."""
        save_dir = MODELS_DIR / f"bert_{self.task_name}"
        save_dir.mkdir(exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"💾 BERT model saved: {save_dir}")

    def load_model(self):
        """Load saved model and tokenizer."""
        save_dir = MODELS_DIR / f"bert_{self.task_name}"
        if not save_dir.exists():
            raise FileNotFoundError(f"No saved BERT model at {save_dir}")
        self.tokenizer = BertTokenizer.from_pretrained(save_dir)
        self.model = BertForSequenceClassification.from_pretrained(save_dir)
        self.model.to(self.device)
        logger.info(f"✅ BERT model loaded from {save_dir}")
