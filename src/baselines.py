"""
Indic Stance Detection - Step 4: Baseline Machine Learning Models
Target Claim: "Climate change and global warming is a serious concern."
Languages: Pure Hindi (Devanagari) & Pure Bengali (Bangla)

Features:
- Dual-Level TF-IDF:
    1. Word N-grams (1-2) using custom Indic tokenizer (capturing whole words & emojis)
    2. Character N-grams (2-4) capturing morphological root stems & agglutinative affixes
- Zero Data Leakage: Vectorizer is fitted strictly on Train split.

Classifiers:
1. Complement Naive Bayes (ComplementNB - handles class imbalance & text sparsity)
2. Logistic Regression (class_weight='balanced', L2 regularization)
3. Linear Support Vector Classifier (LinearSVC - maximum-margin hyperplane)
4. Random Forest Classifier (class_weight='balanced', non-linear tree ensemble)

Diagnostics for Technical Report:
- Macro F1 (primary metric), Accuracy, Precision, Recall
- Per-class F1 (FAVOUR, AGAINST, NONE)
- Language breakdown (Hindi vs. Bengali test performance)
- Sarcasm stress-test (evaluating classical ML on sarcastic comments)
- Feature attributions (Top 15 features per class from Logistic Regression)
- Confusion matrices saved to report/figures/
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend to avoid tkinter threading issues
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Add project root to sys.path so 'src' can be imported anywhere
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Custom Indic tokenizer from Step 3
from src.preprocessing import indic_tokenize

# Fix Windows console UTF-8 output encoding
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Paths
DATA_DIR = "d:/nlp/data/processed"
RAW_CORPUS_PATH = "d:/nlp/data/raw/indic_stance_corpus_3200.csv"
MODELS_DIR = "d:/nlp/models"
PRED_DIR = "d:/nlp/predictions"
REPORT_DIR = "d:/nlp/report"
FIG_DIR = os.path.join(REPORT_DIR, "figures")

for d in [MODELS_DIR, PRED_DIR, REPORT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

TARGET_CLASSES = ["AGAINST", "FAVOUR", "NONE"]


def load_datasets():
    """Loads preprocessed datasets and raw metadata mapping."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), keep_default_na=False)
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"), keep_default_na=False)
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), keep_default_na=False)
    hidden_df = pd.read_csv(os.path.join(DATA_DIR, "hidden_test_unlabeled.csv"), keep_default_na=False)
    
    # Load subcategory metadata for sarcasm stress-test
    raw_df = pd.read_csv(RAW_CORPUS_PATH, keep_default_na=False)
    id_to_subcat = dict(zip(raw_df["id"], raw_df["subcategory"]))
    
    test_df["subcategory"] = test_df["id"].map(id_to_subcat)
    val_df["subcategory"] = val_df["id"].map(id_to_subcat)
    train_df["subcategory"] = train_df["id"].map(id_to_subcat)
    
    print(f"Loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}, Hidden={len(hidden_df)}")
    return train_df, val_df, test_df, hidden_df


def build_feature_extractor():
    """
    Dual-Level TF-IDF Feature Extractor:
    - Track A: Word N-grams (1-2) with custom indic_tokenize (preserving words & emojis)
    - Track B: Subword Character N-grams (2-4) capturing Indic morphology & stems
    """
    word_vectorizer = TfidfVectorizer(
        tokenizer=indic_tokenize,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=3,
        sublinear_tf=True
    )
    
    return FeatureUnion([
        ("word", word_vectorizer),
        ("char", char_vectorizer)
    ])


def plot_and_save_confusion_matrix(cm, model_name, filename, labels=TARGET_CLASSES):
    """Generates and saves a high-resolution confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False
    )
    plt.title(f"Confusion Matrix: {model_name}", fontsize=12, pad=12)
    plt.xlabel("Predicted Stance", fontsize=10)
    plt.ylabel("True Stance", fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def extract_top_features(model, vectorizer, class_labels=TARGET_CLASSES, top_n=15):
    """Extracts top positive feature weights per stance class from Logistic Regression."""
    if not hasattr(model, "coef_"):
        return {}
        
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_features = {}
    
    for i, label in enumerate(class_labels):
        coef = model.coef_[i]
        top_indices = np.argsort(coef)[::-1][:top_n]
        top_features[label] = [
            (feature_names[idx], round(float(coef[idx]), 4))
            for idx in top_indices
        ]
        
    return top_features


def run_sarcasm_evaluation(model, X_test, test_df):
    """Evaluates performance strictly on sarcastic comments to highlight classical ML limits."""
    sarcastic_mask = test_df["subcategory"].str.contains("Sarcastic", case=False, na=False).values
    if sarcastic_mask.sum() == 0:
        return {"count": 0, "macro_f1": 0.0, "accuracy": 0.0}
        
    X_sarcastic = X_test[sarcastic_mask]
    y_sarcastic = test_df.loc[sarcastic_mask, "stance"].values
    
    y_pred = model.predict(X_sarcastic)
    
    return {
        "count": int(sarcastic_mask.sum()),
        "accuracy": round(float(accuracy_score(y_sarcastic, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_sarcastic, y_pred, average="macro", zero_division=0)), 4),
        "per_class_f1": {
            cls: round(float(f1_score(y_sarcastic == cls, y_pred == cls, zero_division=0)), 4)
            for cls in TARGET_CLASSES
        }
    }


def run_language_breakdown(model, X_test, test_df):
    """Evaluates test performance disaggregated by language (Hindi vs. Bengali)."""
    results = {}
    for lang in ["Hindi", "Bengali"]:
        mask = (test_df["language"] == lang).values
        if mask.sum() == 0:
            continue
        X_sub = X_test[mask]
        y_sub = test_df.loc[mask, "stance"].values
        y_pred = model.predict(X_sub)
        
        results[lang] = {
            "samples": int(mask.sum()),
            "accuracy": round(float(accuracy_score(y_sub, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_sub, y_pred, average="macro", zero_division=0)), 4),
            "per_class_f1": {
                cls: round(float(f1_score(y_sub == cls, y_pred == cls, zero_division=0)), 4)
                for cls in TARGET_CLASSES
            }
        }
    return results


def main():
    print("=" * 80)
    print("STEP 4: BASELINE MACHINE LEARNING MODELS EXECUTION")
    print("=" * 80)
    
    # 1. Load Data
    train_df, val_df, test_df, hidden_df = load_datasets()
    
    y_train = train_df["stance"].values
    y_val = val_df["stance"].values
    y_test = test_df["stance"].values
    
    # 2. Build & Fit Feature Extractor (Strictly on Train to guard against leakage)
    print("\n[1/5] Fitting Dual-Level TF-IDF Feature Extractor on Train Split...")
    vectorizer = build_feature_extractor()
    X_train = vectorizer.fit_transform(train_df["cleaned_text"])
    X_val = vectorizer.transform(val_df["cleaned_text"])
    X_test = vectorizer.transform(test_df["cleaned_text"])
    X_hidden = vectorizer.transform(hidden_df["cleaned_text"])
    
    num_features = X_train.shape[1]
    print(f"  -> Extracted {num_features:,d} combined features (Word N-grams + Char N-grams)")
    print(f"  -> Feature Matrix Shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # Save fitted vectorizer
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, vec_path)
    print(f"  -> Vectorizer saved to {vec_path}")
    
    # 3. Model Definitions
    models = {
        "ComplementNB": ComplementNB(alpha=1.0),
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42),
        "LinearSVC": LinearSVC(class_weight="balanced", max_iter=2000, C=1.0, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    }
    
    model_evaluations = {}
    best_model_name = None
    best_macro_f1 = -1.0
    best_model_obj = None
    best_test_preds = None
    
    print("\n[2/5] Training and Evaluating Baseline Classifiers...")
    print("-" * 80)
    print(f"{'Model':<20} | {'Val Acc':<8} | {'Val F1':<8} | {'Test Acc':<8} | {'Test F1 (Macro)':<16} | Status")
    print("-" * 80)
    
    for name, clf in models.items():
        # Train
        clf.fit(X_train, y_train)
        
        # Save individual model checkpoint for frontend exploration
        clf_path = os.path.join(MODELS_DIR, f"model_{name.lower()}.pkl")
        joblib.dump(clf, clf_path)
        
        # Validation predictions
        val_preds = clf.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        
        # Test predictions
        test_preds = clf.predict(X_test)
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
        test_prec = precision_score(y_test, test_preds, average="macro", zero_division=0)
        test_rec = recall_score(y_test, test_preds, average="macro", zero_division=0)
        
        # Per-class F1
        per_class = {
            cls: round(float(f1_score(y_test == cls, test_preds == cls, zero_division=0)), 4)
            for cls in TARGET_CLASSES
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, test_preds, labels=TARGET_CLASSES)
        cm_file = f"cm_{name.lower()}.png"
        cm_path = plot_and_save_confusion_matrix(cm, name, cm_file)
        
        # Sarcasm stress-test
        sarcasm_res = run_sarcasm_evaluation(clf, X_test, test_df)
        
        # Language breakdown
        lang_res = run_language_breakdown(clf, X_test, test_df)
        
        model_evaluations[name] = {
            "validation": {
                "accuracy": round(float(val_acc), 4),
                "macro_f1": round(float(val_f1), 4)
            },
            "test": {
                "accuracy": round(float(test_acc), 4),
                "macro_f1": round(float(test_f1), 4),
                "macro_precision": round(float(test_prec), 4),
                "macro_recall": round(float(test_rec), 4),
                "per_class_f1": per_class
            },
            "language_breakdown": lang_res,
            "sarcasm_stress_test": sarcasm_res,
            "confusion_matrix": cm.tolist(),
            "cm_figure_path": cm_path
        }
        
        is_champion = test_f1 > best_macro_f1
        status = "★ BEST" if is_champion else ""
        if is_champion:
            best_macro_f1 = test_f1
            best_model_name = name
            best_model_obj = clf
            best_test_preds = test_preds
            
        print(f"{name:<20} | {val_acc*100:6.2f}% | {val_f1:7.4f} | {test_acc*100:7.2f}% | {test_f1:15.4f} | {status}")
        
    print("-" * 80)
    print(f"\n[3/5] Champion Baseline Model: {best_model_name} (Test Macro F1 = {best_macro_f1:.4f})")
    
    # Save Champion Model
    best_model_path = os.path.join(MODELS_DIR, "best_baseline_model.pkl")
    joblib.dump(best_model_obj, best_model_path)
    print(f"  -> Saved Champion Model to {best_model_path}")
    
    # 4. Feature Attributions (Logistic Regression Interpretability)
    print("\n[4/5] Extracting Top Interpretability Features (XAI) from Logistic Regression...")
    lr_model = models["LogisticRegression"]
    top_features = extract_top_features(lr_model, vectorizer)
    model_evaluations["logistic_regression_top_features"] = top_features
    
    for cls in TARGET_CLASSES:
        feats = top_features.get(cls, [])[:8]
        feat_str = ", ".join([f"{f} (+{w})" for f, w in feats])
        print(f"  [{cls} Top Features]: {feat_str}")
        
    # 5. Predictions on Hidden Test Set
    print("\n[5/5] Generating Predictions for Hidden Test Set...")
    hidden_preds = best_model_obj.predict(X_hidden)
    pred_df = pd.DataFrame({
        "id": hidden_df["id"],
        "predicted_stance": hidden_preds
    })
    pred_path = os.path.join(PRED_DIR, "baseline_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"  -> Saved {len(pred_df)} predictions to {pred_path}")
    print(f"  -> Hidden Test Stance Distribution: {pred_df['predicted_stance'].value_counts().to_dict()}")
    
    # Save full evaluation summary for technical report
    summary_path = os.path.join(REPORT_DIR, "baseline_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(model_evaluations, f, indent=2, ensure_ascii=False)
    print(f"  -> Saved Full Baseline Evaluation to {summary_path}")
    
    # Detailed Diagnostics Printout
    print("\n" + "=" * 80)
    print(f"DETAILED DIAGNOSTICS: {best_model_name.upper()} (CHAMPION BASELINE)")
    print("=" * 80)
    print(classification_report(y_test, best_test_preds, target_names=TARGET_CLASSES, digits=4))
    
    print("\n--- Language Disaggregation ---")
    best_lang = model_evaluations[best_model_name]["language_breakdown"]
    for lang, metrics in best_lang.items():
        print(f"  {lang:<10} (N={metrics['samples']}): Macro F1 = {metrics['macro_f1']:.4f} | Accuracy = {metrics['accuracy']*100:.2f}% | F1: {metrics['per_class_f1']}")
        
    print("\n--- Sarcasm Stress-Test (Where Classical ML Struggles) ---")
    best_sarcasm = model_evaluations[best_model_name]["sarcasm_stress_test"]
    print(f"  Sarcastic Samples Count: {best_sarcasm['count']}")
    print(f"  Sarcastic Subset Accuracy: {best_sarcasm['accuracy']*100:.2f}%")
    print(f"  Sarcastic Subset Macro F1: {best_sarcasm['macro_f1']:.4f}")
    print(f"  Per-Class F1 on Sarcasm:   {best_sarcasm['per_class_f1']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
