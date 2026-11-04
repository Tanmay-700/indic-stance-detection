"""
Classical NLP Baselines for Indic Stance Detection
Features: Word & Character N-Gram TF-IDF
Classifiers: Logistic Regression, Linear SVM (Calibrated), MultinomialNB, Random Forest
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from evaluate import evaluate_predictions

def train_and_eval_baselines(
    train_path="d:/nlp/data/processed/train.csv",
    val_path="d:/nlp/data/processed/val.csv",
    test_path="d:/nlp/data/processed/test.csv",
    models_dir="d:/nlp/models",
    report_dir="d:/nlp/report"
):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    print("Loading data...")
    train_df = pd.read_csv(train_path, keep_default_na=False)
    val_df = pd.read_csv(val_path, keep_default_na=False)
    test_df = pd.read_csv(test_path, keep_default_na=False)
    
    X_train = train_df["cleaned_text"].fillna("")
    y_train = train_df["stance"]
    
    X_val = val_df["cleaned_text"].fillna("")
    y_val = val_df["stance"]
    
    X_test = test_df["cleaned_text"].fillna("")
    y_test = test_df["stance"]
    
    print(f"Train samples: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Combined Word + Character N-Gram TF-IDF Vectorizer
    print("Extracting Word & Character N-Gram TF-IDF features...")
    word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=15000, sublinear_tf=True)
    
    vectorizer = FeatureUnion([
        ("word", word_vec),
        ("char", char_vec)
    ])
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    
    classifiers = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=42),
        "Linear SVM (Calibrated)": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    }
    
    best_f1 = -1.0
    best_model_name = None
    best_model = None
    results = []
    
    for name, clf in classifiers.items():
        print(f"\n--- Training {name} ---")
        clf.fit(X_train_vec, y_train)
        
        # Validation score
        val_preds = clf.predict(X_val_vec)
        test_preds = clf.predict(X_test_vec)
        
        cm_path = os.path.join(report_dir, f"cm_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png")
        eval_metrics = evaluate_predictions(y_test, test_preds, model_name=name, save_cm_path=cm_path)
        eval_metrics["classifier"] = clf
        results.append(eval_metrics)
        
        if eval_metrics["macro_f1"] > best_f1:
            best_f1 = eval_metrics["macro_f1"]
            best_model_name = name
            best_model = clf
            
    print(f"\n[BEST MODEL] Best Classical Baseline: {best_model_name} with Test Macro F1: {best_f1:.4f}")
    
    # Save best baseline & vectorizer
    joblib.dump(best_model, os.path.join(models_dir, "best_baseline_model.pkl"))
    joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    print(f"Saved best model and vectorizer to {models_dir}")
    
    return results

if __name__ == "__main__":
    train_and_eval_baselines()
