"""
Indic Stance Detection - Step 5: Transformer Fine-Tuning Pipeline
Model: google/muril-base-cased (Multilingual Representations for Indic Languages)
Target Claim: "Climate change and global warming is a serious concern."

Key Architectural Highlights:
1. Google MuRIL Backbone: Pretrained on 17 Indic languages (including Hindi & Bengali).
2. Vocabulary Expansion: Adds domain emojis (🌱, 🌍, 🙄, 🤡, ⚠️, etc.) as explicit tokens so
   they are never collapsed into [UNK].
3. Model Resizing: resizes model embeddings to learn dedicated contextual vectors for emojis.
4. Loss & Metric Optimization: Optimized for Macro-averaged F1 (the competition metric).
5. Comprehensive Diagnostics: Per-class F1, language breakdown (Hindi vs. Bengali),
   confusion matrix generation, and hidden test prediction file creation.
"""

import sys
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Fix Windows console UTF-8 output encoding
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_indic_text, EMOJI_SEQUENCE

# Paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_CORPUS_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "indic_stance_corpus_3200.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "transformer_muril")
PRED_DIR = os.path.join(PROJECT_ROOT, "predictions")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")
FIG_DIR = os.path.join(REPORT_DIR, "figures")

for d in [MODELS_DIR, PRED_DIR, REPORT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

MODEL_NAME = "google/muril-base-cased"
TARGET_CLASSES = ["AGAINST", "FAVOUR", "NONE"]
LABEL2ID = {label: idx for idx, label in enumerate(TARGET_CLASSES)}
ID2LABEL = {idx: label for idx, label in enumerate(TARGET_CLASSES)}


class IndicStanceDataset(Dataset):
    """PyTorch Dataset wrapper for Indic stance comments."""
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        self.texts = list(texts)
        self.labels = [LABEL2ID[l] for l in labels] if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }
        
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return item


def extract_unique_emojis(texts):
    """Finds all unique emoji sequences in the dataset to expand tokenizer vocabulary."""
    all_emojis = set()
    for t in texts:
        matches = EMOJI_SEQUENCE.findall(t)
        for m in matches:
            if m.strip():
                all_emojis.add(m.strip())
    return sorted(list(all_emojis))


def compute_metrics(y_true, y_pred):
    """Computes comprehensive metrics matching assignment requirements."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "per_class_f1": {
            cls: round(float(f1_score(np.array(y_true) == LABEL2ID[cls], np.array(y_pred) == LABEL2ID[cls], zero_division=0)), 4)
            for cls in TARGET_CLASSES
        }
    }


def evaluate_model(model, dataloader, device):
    """Runs evaluation over a DataLoader, returning loss, metrics, and predictions."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item() * len(input_ids)

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = round(avg_loss, 4)
    return metrics, np.array(all_preds), np.array(all_labels)


def plot_and_save_confusion_matrix(y_true, y_pred, filename="cm_transformer_muril.png"):
    """Generates high-resolution confusion matrix heatmap for technical report."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=TARGET_CLASSES,
        yticklabels=TARGET_CLASSES,
        cbar=False
    )
    plt.title("Confusion Matrix: Google MuRIL (Step 5)", fontsize=12, pad=12)
    plt.xlabel("Predicted Stance", fontsize=10)
    plt.ylabel("True Stance", fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def train_transformer(
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 128,
    device_override: str = None
):
    """
    Main fine-tuning routine for Google MuRIL.
    Can run on GPU or CPU.
    """
    device = torch.device(device_override if device_override else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Executing MuRIL Fine-Tuning on Device: {device}")
    
    # 1. Load Data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), keep_default_na=False)
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"), keep_default_na=False)
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), keep_default_na=False)
    hidden_df = pd.read_csv(os.path.join(DATA_DIR, "hidden_test_unlabeled.csv"), keep_default_na=False)
    
    print(f"Data Loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}, Hidden={len(hidden_df)}")
    
    # 2. Tokenizer & Vocabulary Expansion (Adding Emojis)
    print(f"\n[1/5] Loading MuRIL Tokenizer from '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    unique_emojis = extract_unique_emojis(train_df["cleaned_text"].tolist() + test_df["cleaned_text"].tolist())
    print(f"  -> Discovered {len(unique_emojis)} unique emojis in corpus: {unique_emojis}")
    num_added = tokenizer.add_tokens(unique_emojis)
    print(f"  -> Successfully added {num_added} new emoji tokens to MuRIL vocabulary (New vocab size: {len(tokenizer):,d})")
    
    # 3. Model Initialization
    print(f"\n[2/5] Initializing MuRIL Sequence Classification Head (3 Classes)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print("  -> Resized model token embeddings to accommodate newly added emoji tokens.")
        
    model.to(device)
    
    # 4. DataLoaders
    train_dataset = IndicStanceDataset(train_df["cleaned_text"], train_df["stance"], tokenizer, max_length)
    val_dataset = IndicStanceDataset(val_df["cleaned_text"], val_df["stance"], tokenizer, max_length)
    test_dataset = IndicStanceDataset(test_df["cleaned_text"], test_df["stance"], tokenizer, max_length)
    hidden_dataset = IndicStanceDataset(hidden_df["cleaned_text"], None, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    hidden_loader = DataLoader(hidden_dataset, batch_size=batch_size, shuffle=False)
    
    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # 6. Training Loop
    print(f"\n[3/5] Starting Training: {epochs} Epochs, {len(train_loader)} batches/epoch (Batch Size={batch_size}, LR={lr})...")
    print("-" * 80)
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val Acc':<9} | {'Val Macro F1':<12} | Status")
    print("-" * 80)
    
    best_val_f1 = -1.0
    history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * len(input_ids)
            
        avg_train_loss = train_loss / len(train_dataset)
        val_metrics, _, _ = evaluate_model(model, val_loader, device)
        
        is_best = val_metrics["macro_f1"] > best_val_f1
        status = "★ BEST" if is_best else ""
        if is_best:
            best_val_f1 = val_metrics["macro_f1"]
            # Save checkpoint
            model.save_pretrained(MODELS_DIR)
            tokenizer.save_pretrained(MODELS_DIR)
            
        print(f"{epoch:<6} | {avg_train_loss:12.4f} | {val_metrics['loss']:10.4f} | {val_metrics['accuracy']*100:7.2f}% | {val_metrics['macro_f1']:12.4f} | {status}")
        
        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"]
        })
        
    print("-" * 80)
    print(f"Best Validation Macro F1: {best_val_f1:.4f} (Model saved to {MODELS_DIR})")
    
    # 7. Final Test Set Evaluation
    print(f"\n[4/5] Evaluating Best Checkpoint on Test Set (320 rows)...")
    best_model = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR).to(device)
    test_metrics, test_preds, test_labels = evaluate_model(best_model, test_loader, device)
    
    print("\n" + "=" * 80)
    print("GOOGLE MuRIL TEST SET PERFORMANCE")
    print("=" * 80)
    print(f"  Test Accuracy:        {test_metrics['accuracy']*100:.2f}%")
    print(f"  Test Macro F1:        {test_metrics['macro_f1']:.4f}")
    print(f"  Test Macro Precision: {test_metrics['macro_precision']:.4f}")
    print(f"  Test Macro Recall:    {test_metrics['macro_recall']:.4f}")
    print(f"  Per-Class F1 Scores:  {test_metrics['per_class_f1']}")
    
    y_test_pred_labels = [ID2LABEL[p] for p in test_preds]
    y_test_true_labels = [ID2LABEL[l] for l in test_labels]
    print("\nDetailed Classification Report:")
    print(classification_report(y_test_true_labels, y_test_pred_labels, target_names=TARGET_CLASSES, digits=4))
    
    # Save Confusion Matrix figure
    cm_path = plot_and_save_confusion_matrix(test_labels, test_preds)
    print(f"  -> Saved Confusion Matrix Heatmap to {cm_path}")
    
    # Language Breakdown
    print("\n--- Language Disaggregation ---")
    lang_results = {}
    for lang in ["Hindi", "Bengali"]:
        mask = (test_df["language"] == lang).values
        sub_preds = test_preds[mask]
        sub_labels = test_labels[mask]
        sub_metrics = compute_metrics(sub_labels, sub_preds)
        lang_results[lang] = sub_metrics
        print(f"  {lang:<10} (N={mask.sum()}): Macro F1 = {sub_metrics['macro_f1']:.4f} | Accuracy = {sub_metrics['accuracy']*100:.2f}% | F1: {sub_metrics['per_class_f1']}")
        
    # 8. Hidden Test Predictions
    print(f"\n[5/5] Generating Predictions on Hidden Test Set...")
    best_model.eval()
    hidden_preds_list = []
    with torch.no_grad():
        for batch in hidden_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            hidden_preds_list.extend(preds)
            
    hidden_pred_labels = [ID2LABEL[p] for p in hidden_preds_list]
    pred_df = pd.DataFrame({
        "id": hidden_df["id"],
        "predicted_stance": hidden_pred_labels
    })
    pred_path = os.path.join(PRED_DIR, "transformer_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"  -> Saved {len(pred_df)} competition predictions to {pred_path}")
    print(f"  -> Hidden Test Stance Distribution: {pred_df['predicted_stance'].value_counts().to_dict()}")
    
    # Save Results JSON
    summary = {
        "model_name": MODEL_NAME,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "best_val_macro_f1": best_val_f1,
        "test_metrics": test_metrics,
        "language_breakdown": lang_results,
        "history": history,
        "cm_figure_path": cm_path
    }
    summary_path = os.path.join(REPORT_DIR, "transformer_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  -> Saved Transformer Evaluation Summary to {summary_path}")
    
    return summary


if __name__ == "__main__":
    # Check if run with arguments or default demonstration
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune MuRIL for Indic Stance Detection")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()
    
    train_transformer(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
