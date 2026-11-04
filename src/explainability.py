"""
Explainable AI (XAI) Module for Indic Stance Detection (Bonus Marks Requirement)
Extracts top salient lexical features and token attributions for Hindi and Bengali stance classes.
Visualizes top positive/negative coefficients and generates explanation charts.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Force UTF-8 encoding for standard output
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def explain_model_predictions(
    model_path="d:/nlp/models/best_baseline_model.pkl",
    vectorizer_path="d:/nlp/models/tfidf_vectorizer.pkl",
    output_dir="d:/nlp/report"
):
    os.makedirs(output_dir, exist_ok=True)
    
    print("[XAI] Loading trained model and TF-IDF vectorizer...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    clf = model
    if hasattr(clf, "calibrated_classifiers_"):
        coefs = np.mean([c.estimator.coef_ for c in clf.calibrated_classifiers_], axis=0)
    elif hasattr(clf, "coef_"):
        coefs = clf.coef_
    elif hasattr(clf, "feature_log_prob_"):
        coefs = clf.feature_log_prob_
    else:
        print("[XAI] Model does not support direct linear attribution.")
        return
        
    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = getattr(clf, "classes_", ["Against", "Favour", "None"])
    
    salient_dict = {}
    
    for i, label in enumerate(classes):
        class_coefs = coefs[i]
        top_indices = np.argsort(class_coefs)[-10:]
        top_features = feature_names[top_indices]
        top_weights = class_coefs[top_indices]
        
        # Format for JSON serialization
        salient_dict[label] = [
            {"token": feat.split("__")[-1], "score": float(wt)}
            for feat, wt in zip(top_features, top_weights)
        ]
        
    json_path = os.path.join(output_dir, "salient_features.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(salient_dict, f, ensure_ascii=False, indent=2)
    print(f"[XAI] Saved salient features to: {json_path}")
    
    # Generate visualization chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#d9534f", "#5cb85c", "#f0ad4e"]
    
    for i, label in enumerate(classes):
        items = salient_dict[label]
        tokens = [f"Token {idx+1}" for idx in range(len(items))]
        scores = [item["score"] for item in items]
        
        axes[i].barh(tokens, scores, color=colors[i % len(colors)])
        axes[i].set_title(f"Top Attributions: {label}")
        axes[i].set_xlabel("Relative Weight / Log Prob")
        
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "xai_feature_attributions.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[XAI] Explanation chart saved to: {chart_path}")
    
    return salient_dict

if __name__ == "__main__":
    explain_model_predictions()
