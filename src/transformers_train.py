"""
Transformer Fine-Tuning Module for Indic Stance Detection
Supported Models:
- google/muril-base-cased (Multilingual Representations for Indian Languages)
- ai4bharat/indic-bert
- bert-base-multilingual-cased (mBERT)
- xlm-roberta-base
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from evaluate import evaluate_predictions

LABEL2ID = {"Against": 0, "Favour": 1, "None": 2}
ID2LABEL = {0: "Against", 1: "Favour", 2: "None"}

class IndicStanceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
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
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(LABEL2ID[self.labels[idx]], dtype=torch.long)
        return item

def train_transformer(
    model_name="bert-base-multilingual-cased",
    train_path="d:/nlp/data/processed/train.csv",
    val_path="d:/nlp/data/processed/val.csv",
    test_path="d:/nlp/data/processed/test.csv",
    output_dir="d:/nlp/models/transformer_stance",
    epochs=2,
    batch_size=16,
    lr=2e-5,
    max_length=128,
    max_train_samples=None
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SETUP] Training {model_name} on device: {device}")
    
    # Load splits with keep_default_na=False to preserve 'None'
    train_df = pd.read_csv(train_path, keep_default_na=False)
    val_df = pd.read_csv(val_path, keep_default_na=False)
    test_df = pd.read_csv(test_path, keep_default_na=False)
    
    if max_train_samples and max_train_samples < len(train_df):
        print(f"[INFO] Subsampling training set to {max_train_samples} for rapid execution.")
        train_df = train_df.sample(n=max_train_samples, random_state=42).reset_index(drop=True)
        
    print(f"[DATA] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    ).to(device)
    
    train_dataset = IndicStanceDataset(train_df["cleaned_text"].values, train_df["stance"].values, tokenizer, max_length)
    val_dataset = IndicStanceDataset(val_df["cleaned_text"].values, val_df["stance"].values, tokenizer, max_length)
    test_dataset = IndicStanceDataset(test_df["cleaned_text"].values, test_df["stance"].values, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    
    best_val_f1 = -1.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
            if (step + 1) % 20 == 0 or (step + 1) == len(train_loader):
                print(f"Epoch {epoch+1}/{epochs} | Step {step+1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")
                
        avg_train_loss = total_loss / len(train_loader)
        print(f"\n--- Epoch {epoch+1} Completed | Average Train Loss: {avg_train_loss:.4f} ---")
        
        # Validation Evaluation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend([ID2LABEL[p] for p in preds])
                val_trues.extend([ID2LABEL[l] for l in batch["labels"].numpy()])
                
        metrics = evaluate_predictions(val_trues, val_preds, model_name=f"{model_name} (Val Epoch {epoch+1})")
        if metrics["macro_f1"] > best_val_f1:
            best_val_f1 = metrics["macro_f1"]
            print(f"[SAVE] New best validation Macro F1: {best_val_f1:.4f}. Saving checkpoint to {output_dir}")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
    # Test Set Evaluation with Best Model
    print("\n--- Running Final Evaluation on Test Set ---")
    best_model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    best_model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            test_preds.extend([ID2LABEL[p] for p in preds])
            test_trues.extend([ID2LABEL[l] for l in batch["labels"].numpy()])
            
    cm_path = os.path.join("d:/nlp/report", "cm_transformer.png")
    test_metrics = evaluate_predictions(test_trues, test_preds, model_name=f"{model_name} (Test)", save_cm_path=cm_path)
    return test_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/muril-base-cased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_train_samples", type=int, default=None)
    args = parser.parse_args()
    
    train_transformer(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train_samples=args.max_train_samples
    )
