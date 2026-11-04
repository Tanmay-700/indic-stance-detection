"""
Comprehensive Error Analysis Module for Indic Stance Detection (Bonus Marks Requirement)
Analyzes cross-lingual disparities (Hindi vs Bengali), confusion patterns, and edge cases.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def perform_error_analysis(
    model_path="d:/nlp/models/best_baseline_model.pkl",
    vectorizer_path="d:/nlp/models/tfidf_vectorizer.pkl",
    test_path="d:/nlp/data/processed/test.csv",
    output_dir="d:/nlp/report"
):
    os.makedirs(output_dir, exist_ok=True)
    
    test_df = pd.read_csv(test_path, keep_default_na=False)
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    X_test = vectorizer.transform(test_df["cleaned_text"].fillna(""))
    y_test = test_df["stance"].values
    preds = model.predict(X_test)
    
    test_df["predicted"] = preds
    test_df["is_error"] = test_df["stance"] != test_df["predicted"]
    
    print("[ERROR ANALYSIS] Total Test Samples:", len(test_df))
    print(f"[ERROR ANALYSIS] Total Misclassifications: {test_df['is_error'].sum()} ({test_df['is_error'].mean()*100:.2f}%)")
    
    # 1. Linguistic Disparity Analysis (Hindi vs Bengali)
    hindi_df = test_df[test_df["language"] == "Hindi"]
    bengali_df = test_df[test_df["language"] == "Bengali"]
    
    hindi_rep = classification_report(hindi_df["stance"], hindi_df["predicted"], output_dict=True, zero_division=0)
    bengali_rep = classification_report(bengali_df["stance"], bengali_df["predicted"], output_dict=True, zero_division=0)
    
    analysis_report = {
        "overall_accuracy": float((test_df["stance"] == test_df["predicted"]).mean()),
        "hindi_performance": {
            "macro_f1": float(hindi_rep["macro avg"]["f1-score"]),
            "accuracy": float(hindi_rep["accuracy"])
        },
        "bengali_performance": {
            "macro_f1": float(bengali_rep["macro avg"]["f1-score"]),
            "accuracy": float(bengali_rep["accuracy"])
        },
        "error_categories": [
            {
                "type": "Sarcasm & Rhetorical Denial",
                "description": "Comments using religious or cynical tropes (e.g., kalyug/qiyamat) to dismiss human cause without explicit climate policy denial.",
                "mitigation": "Include conversational context and fine-tune Indic-specific transformers (MuRIL)."
            },
            {
                "type": "Short Informational Descriptive Comments",
                "description": "Comments mentioning rising temperature as a factual observation without explicit urgency or fear, teetering between None and Favour.",
                "mitigation": "Contextual claim-conditioned stance prompting."
            },
            {
                "type": "Transliterated Social Media Slang",
                "description": "YouTube informal spelling variants and loanwords (e.g. 'ड्रामा', 'নাটক').",
                "mitigation": "Character n-gram tokenization and subword BPE representations."
            }
        ],
        "misclassified_samples": test_df[test_df["is_error"]][["id", "language", "stance", "predicted", "text"]].to_dict(orient="records")
    }
    
    out_json = os.path.join(output_dir, "error_analysis.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(analysis_report, f, ensure_ascii=False, indent=2)
        
    print(f"[ERROR ANALYSIS] Report saved to: {out_json}")
    return analysis_report

if __name__ == "__main__":
    perform_error_analysis()
