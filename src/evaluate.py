"""
Evaluation Module for Indic Stance Detection
Computes Primary Metric: Macro F1-Score
Additional Metrics: Accuracy, Precision, Recall, Confusion Matrix Plot.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

LABELS = ["Against", "Favour", "None"]

def evaluate_predictions(y_true, y_pred, model_name="Model", save_cm_path=None):
    """
    Evaluates predictions against true labels.
    Returns metrics dict and prints complete evaluation table.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
    )
    
    print(f"\n{'='*55}")
    print(f"[EVALUATION] Results for: {model_name}")
    print(f"{'='*55}")
    print(f"- Primary Metric (Macro F1-Score): {f1_macro:.4f}")
    print(f"- Overall Accuracy:               {acc:.4f} ({acc*100:.2f}%)")
    print(f"- Macro Precision:                {prec_macro:.4f}")
    print(f"- Macro Recall:                   {rec_macro:.4f}")
    print(f"- Weighted F1-Score:              {f1_weighted:.4f}")
    print(f"{'='*55}")
    
    print("\nDetailed Per-Class Performance:")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=4, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    
    if save_cm_path:
        os.makedirs(os.path.dirname(save_cm_path), exist_ok=True)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=LABELS,
            yticklabels=LABELS
        )
        plt.title(f"Confusion Matrix: {model_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(save_cm_path, dpi=300)
        plt.close()
        print(f"Confusion matrix saved to: {save_cm_path}")
        
    return {
        "model": model_name,
        "macro_f1": f1_macro,
        "accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "weighted_f1": f1_weighted,
        "confusion_matrix": cm
    }
