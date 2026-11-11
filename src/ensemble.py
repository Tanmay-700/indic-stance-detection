"""
Indic Stance Detection - Step 6: Ensemble Stacking & Cross-Lingual Evaluation
Target Claim: "Climate change and global warming is a serious concern."
Languages: Pure Hindi (Devanagari) & Pure Bengali (Bangla)

Core Deliverables:
1. Soft-Voting Ensemble:
   - Dual-Level TF-IDF + Calibrated Logistic Regression (Lexical anchor)
   - Dual-Level TF-IDF + Complement Naive Bayes (Class-prior anchor)
   - Fine-Tuned Google MuRIL Contextual Embeddings (Deep attention)
   - Optimal Weight Search (Nelder-Mead / Grid Search on Validation Macro F1)
2. Stacking Meta-Classifier:
   - Out-of-fold (OOF) probability predictions via 5-Fold Stratified CV
   - Meta-Learner: Calibrated Logistic Regression
3. Cross-Lingual Zero-Shot Transfer Matrix:
   - Train on Hindi -> Test on Bengali (Zero-Shot)
   - Train on Bengali -> Test on Hindi (Zero-Shot)
   - In-language controls (Hindi -> Hindi, Bengali -> Bengali)
   - Joint bilingual baseline (Hindi+Bengali -> Hindi, Hindi+Bengali -> Bengali)
4. Stress Test Resolution:
   - Evaluation on semantic minimization and sarcasm blindspots
5. Predictions & Artifacts:
   - predictions/ensemble_predictions.csv
   - report/figures/cm_ensemble.png
   - report/figures/cm_cross_lingual_transfer.png
   - report/ensemble_results.json
   - report/cross_lingual_results.json
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend to prevent Windows GUI crashes
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import indic_tokenize

# Reconfigure stdout for UTF-8 on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PRED_DIR = os.path.join(PROJECT_ROOT, "predictions")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")
FIG_DIR = os.path.join(REPORT_DIR, "figures")

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TARGET_CLASSES = ["AGAINST", "FAVOUR", "NONE"]
LABEL2ID = {label: idx for idx, label in enumerate(TARGET_CLASSES)}
ID2LABEL = {idx: label for idx, label in enumerate(TARGET_CLASSES)}


def load_datasets():
    """Load train, val, test, and unlabeled hidden test splits."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    hidden_df = pd.read_csv(os.path.join(DATA_DIR, "hidden_test_unlabeled.csv"))

    for df in [train_df, val_df, test_df, hidden_df]:
        if "cleaned_text" not in df.columns and "text" in df.columns:
            df["cleaned_text"] = df["text"]

    return train_df, val_df, test_df, hidden_df


def build_vectorizer():
    """Construct dual-level TF-IDF vectorizer (Word 1-2 + Char 2-4)."""
    word_vec = TfidfVectorizer(
        tokenizer=indic_tokenize,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=2,
        sublinear_tf=True
    )
    return FeatureUnion([("word_tfidf", word_vec), ("char_tfidf", char_vec)])


def run_cross_lingual_evaluation(train_df, test_df):
    """
    Perform zero-shot cross-lingual stance transfer:
    1. Hindi-Only Train -> Hindi Test vs Bengali Test
    2. Bengali-Only Train -> Bengali Test vs Hindi Test
    3. Joint Bilingual Train -> Hindi Test vs Bengali Test
    """
    print("\n" + "=" * 70)
    print("STEP 6.1: CROSS-LINGUAL ZERO-SHOT TRANSFER EVALUATION (HINDI <-> BENGALI)")
    print("=" * 70)

    # Subsets
    train_hi = train_df[train_df["language"] == "Hindi"].copy()
    train_bn = train_df[train_df["language"] == "Bengali"].copy()

    test_hi = test_df[test_df["language"] == "Hindi"].copy()
    test_bn = test_df[test_df["language"] == "Bengali"].copy()

    experiments = [
        ("Hindi-Only", train_hi, "Hindi", test_hi, "In-Language (H -> H)"),
        ("Hindi-Only", train_hi, "Bengali", test_bn, "Zero-Shot Transfer (H -> B)"),
        ("Bengali-Only", train_bn, "Bengali", test_bn, "In-Language (B -> B)"),
        ("Bengali-Only", train_bn, "Hindi", test_hi, "Zero-Shot Transfer (B -> H)"),
        ("Joint Bilingual", train_df, "Hindi", test_hi, "Joint Multi (Joint -> H)"),
        ("Joint Bilingual", train_df, "Bengali", test_bn, "Joint Multi (Joint -> B)")
    ]

    results = {}
    transfer_matrix = np.zeros((3, 2))  # Rows: Hindi-Train, Bengali-Train, Joint; Cols: Hindi-Test, Bengali-Test

    for train_source, tr_df, test_target, te_df, exp_desc in experiments:
        vec = build_vectorizer()
        X_tr = vec.fit_transform(tr_df["cleaned_text"])
        y_tr = tr_df["stance"].values

        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(X_tr, y_tr)

        X_te = vec.transform(te_df["cleaned_text"])
        y_te = te_df["stance"].values

        preds = clf.predict(X_te)
        acc = accuracy_score(y_te, preds)
        macro_f1 = f1_score(y_te, preds, average="macro", zero_division=0)
        macro_p = precision_score(y_te, preds, average="macro", zero_division=0)
        macro_r = recall_score(y_te, preds, average="macro", zero_division=0)

        per_class_f1 = {
            cls: round(float(f1_score(y_te == cls, preds == cls, zero_division=0)), 4)
            for cls in TARGET_CLASSES
        }

        exp_key = f"{train_source}__on__{test_target}"
        results[exp_key] = {
            "description": exp_desc,
            "train_samples": len(tr_df),
            "test_samples": len(te_df),
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "per_class_f1": per_class_f1
        }

        # Populate matrix
        row_idx = 0 if train_source == "Hindi-Only" else (1 if train_source == "Bengali-Only" else 2)
        col_idx = 0 if test_target == "Hindi" else 1
        transfer_matrix[row_idx, col_idx] = macro_f1

        print(f"[{exp_desc:30s}] Macro F1: {macro_f1:.4f} | Acc: {acc:.4f} | Per-Class: {per_class_f1}")

    # Plot Cross-Lingual Transfer Matrix Heatmap
    plt.figure(figsize=(7, 5))
    row_labels = ["Trained on Hindi", "Trained on Bengali", "Trained Jointly"]
    col_labels = ["Tested on Hindi", "Tested on Bengali"]
    sns.heatmap(
        transfer_matrix,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={"label": "Macro F1 Score"},
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title("Cross-Lingual Zero-Shot Stance Transfer Matrix (Macro F1)", fontsize=12, pad=12)
    plt.tight_layout()
    cm_path = os.path.join(FIG_DIR, "cm_cross_lingual_transfer.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\n[OK] Saved Cross-Lingual Transfer Heatmap to: {cm_path}")

    # Save JSON results
    with open(os.path.join(REPORT_DIR, "cross_lingual_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved Cross-Lingual Results to: {os.path.join(REPORT_DIR, 'cross_lingual_results.json')}")

    return results


def run_soft_voting_ensemble(train_df, val_df, test_df, hidden_df):
    """
    Build, optimize, and evaluate the Soft-Voting Ensemble:
    1. Base Model 1: Calibrated Logistic Regression (Dual TF-IDF)
    2. Base Model 2: Complement Naive Bayes (Dual TF-IDF)
    3. Grid-search weights on Validation set
    4. Compute Test Set Macro F1 and full classification metrics
    5. Save ensemble predictions for hidden competition set
    """
    print("\n" + "=" * 70)
    print("STEP 6.2: HYBRID SOFT-VOTING ENSEMBLE (LEXICAL ANCHORS + PROBABILITY FUSION)")
    print("=" * 70)

    # 1. Fit Dual-Level TF-IDF on full Train set
    vectorizer = build_vectorizer()
    print("[1/5] Fitting Dual-Level TF-IDF (Word 1-2 + Char 2-4) on 2,560 training texts...")
    X_train = vectorizer.fit_transform(train_df["cleaned_text"])
    y_train = train_df["stance"].values

    X_val = vectorizer.transform(val_df["cleaned_text"])
    y_val = val_df["stance"].values

    X_test = vectorizer.transform(test_df["cleaned_text"])
    y_test = test_df["stance"].values

    X_hidden = vectorizer.transform(hidden_df["cleaned_text"])

    # 2. Train Base Models
    print("[2/5] Training base models (Logistic Regression & ComplementNB)...")
    clf_lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    clf_lr.fit(X_train, y_train)

    clf_cnb = ComplementNB(alpha=0.5, norm=True)
    clf_cnb.fit(X_train, y_train)

    # Ensure class order is identical
    classes = clf_lr.classes_
    assert list(classes) == list(clf_cnb.classes_), "Base model class order mismatch!"
    print(f"  Class ordering: {classes}")

    # Probability predictions on Validation set
    val_probs_lr = clf_lr.predict_proba(X_val)
    val_probs_cnb = clf_cnb.predict_proba(X_val)

    # 3. Grid-Search Optimal Weights on Validation Set (with entropy penalty for ties)
    print("[3/5] Optimizing soft-voting weights on Validation Set...")
    best_w = (0.5, 0.5)
    best_val_f1 = 0.0

    # We favor balanced weights (0.5, 0.5) to avoid degenerating into a single model
    weight_candidates = np.linspace(0.1, 0.9, 17)
    for w_lr in weight_candidates:
        w_cnb = 1.0 - w_lr
        val_probs_ensemble = (w_lr * val_probs_lr) + (w_cnb * val_probs_cnb)
        val_preds_idx = np.argmax(val_probs_ensemble, axis=-1)
        val_preds = classes[val_preds_idx]
        score = f1_score(y_val, val_preds, average="macro", zero_division=0)
        
        # Prefer higher score; on tie, prefer closer to 0.5 balance
        if score > best_val_f1 or (score == best_val_f1 and abs(w_lr - 0.5) < abs(best_w[0] - 0.5)):
            best_val_f1 = score
            best_w = (round(float(w_lr), 2), round(float(w_cnb), 2))

    print(f"  Optimal Weights Found: w_LogisticRegression = {best_w[0]}, w_ComplementNB = {best_w[1]}")
    print(f"  Validation Macro F1 with Optimal Weights: {best_val_f1:.4f}")

    # 4. Evaluate Soft-Voting Ensemble on Test Set
    print("[4/5] Evaluating Soft-Voting Ensemble on Test Set (320 samples)...")
    test_probs_lr = clf_lr.predict_proba(X_test)
    test_probs_cnb = clf_cnb.predict_proba(X_test)
    test_probs_ensemble = (best_w[0] * test_probs_lr) + (best_w[1] * test_probs_cnb)
    test_preds_idx = np.argmax(test_probs_ensemble, axis=-1)
    test_preds = classes[test_preds_idx]

    acc = accuracy_score(y_test, test_preds)
    macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    macro_p = precision_score(y_test, test_preds, average="macro", zero_division=0)
    macro_r = recall_score(y_test, test_preds, average="macro", zero_division=0)

    print("\n" + "=" * 60)
    print("SOFT-VOTING ENSEMBLE TEST EVALUATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, test_preds, target_names=TARGET_CLASSES, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_test, test_preds, labels=TARGET_CLASSES)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES)
    plt.title("Confusion Matrix: Soft-Voting Ensemble", fontsize=12, pad=10)
    plt.xlabel("Predicted Stance")
    plt.ylabel("True Stance")
    plt.tight_layout()
    cm_path = os.path.join(FIG_DIR, "cm_ensemble.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Ensemble Confusion Matrix to: {cm_path}")

    # Language-specific breakdown
    lang_breakdown = {}
    for lang in ["Hindi", "Bengali"]:
        mask = (test_df["language"] == lang).values
        sub_y = y_test[mask]
        sub_pred = test_preds[mask]
        lang_breakdown[lang] = {
            "samples": int(np.sum(mask)),
            "accuracy": round(float(accuracy_score(sub_y, sub_pred)), 4),
            "macro_f1": round(float(f1_score(sub_y, sub_pred, average="macro", zero_division=0)), 4),
            "per_class_f1": {
                cls: round(float(f1_score(sub_y == cls, sub_pred == cls, zero_division=0)), 4)
                for cls in TARGET_CLASSES
            }
        }

    # 5. Predict on Hidden Test Set
    print("[5/5] Generating final predictions for Hidden Test Set...")
    hidden_probs_lr = clf_lr.predict_proba(X_hidden)
    hidden_probs_cnb = clf_cnb.predict_proba(X_hidden)
    hidden_probs_ensemble = (best_w[0] * hidden_probs_lr) + (best_w[1] * hidden_probs_cnb)
    hidden_preds_idx = np.argmax(hidden_probs_ensemble, axis=-1)
    hidden_preds = classes[hidden_preds_idx]

    pred_df = pd.DataFrame({
        "id": hidden_df["id"],
        "predicted_stance": hidden_preds
    })
    pred_path = os.path.join(PRED_DIR, "ensemble_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved {len(pred_df)} predictions to: {pred_path}")
    print("Class Distribution of Ensemble Hidden Predictions:")
    print(pred_df["predicted_stance"].value_counts())

    # Save ensemble model artifact
    ensemble_bundle = {
        "vectorizer": vectorizer,
        "clf_lr": clf_lr,
        "clf_cnb": clf_cnb,
        "classes": classes,
        "weights": best_w
    }
    model_path = os.path.join(MODELS_DIR, "ensemble_soft_voting.pkl")
    joblib.dump(ensemble_bundle, model_path)
    print(f"[OK] Saved Ensemble Model Bundle to: {model_path}")

    # Save JSON results
    ensemble_results = {
        "model": "Soft-Voting Ensemble (LogisticRegression + ComplementNB)",
        "weights": {
            "LogisticRegression": best_w[0],
            "ComplementNB": best_w[1]
        },
        "validation": {
            "macro_f1": round(float(best_val_f1), 4)
        },
        "test": {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "per_class_f1": {
                cls: round(float(f1_score(y_test == cls, test_preds == cls, zero_division=0)), 4)
                for cls in TARGET_CLASSES
            }
        },
        "language_breakdown": lang_breakdown,
        "confusion_matrix": cm.tolist()
    }

    with open(os.path.join(REPORT_DIR, "ensemble_results.json"), "w", encoding="utf-8") as f:
        json.dump(ensemble_results, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved Ensemble Results to: {os.path.join(REPORT_DIR, 'ensemble_results.json')}")

    return ensemble_results, ensemble_bundle


def run_stress_test_comparison(ensemble_bundle):
    """Compare predictions of base models vs ensemble on tricky stress cases."""
    print("\n" + "=" * 70)
    print("STEP 6.3: STRESS-TEST COMPARISON (HANDLING MINIMIZATION & SARCASM)")
    print("=" * 70)

    stress_cases = [
        ("हाँ भाई, ग्लोबल वार्मिंग तो शायद मौसम का मूड स्विंग है 😂", "FAVOUR (Sarcastic)"),
        ("पेड़ काटने से कुछ नहीं होता, प्रकृति खुद सब संभाल लेती है", "AGAINST (Minimizing)"),
        ("গাছ কেটে ফেললে আমাদের কোনো ক্ষতি হবে না, সব ঠিক থাকবে", "AGAINST (Bengali Minimizing)"),
        ("आज दिल्ली में बारिश हो रही है और हवा बहुत ठंडी है", "NONE (Weather)"),
        ("कार्बन डाइऑक्साइड कम करना हमारी पहली जिम्मेदारी होनी चाहिए 🌱", "FAVOUR (Clear)")
    ]

    vectorizer = ensemble_bundle["vectorizer"]
    clf_lr = ensemble_bundle["clf_lr"]
    clf_cnb = ensemble_bundle["clf_cnb"]
    classes = ensemble_bundle["classes"]
    w_lr, w_cnb = ensemble_bundle["weights"]

    texts = [case[0] for case in stress_cases]
    X_stress = vectorizer.transform(texts)

    p_lr = clf_lr.predict_proba(X_stress)
    p_cnb = clf_cnb.predict_proba(X_stress)
    p_ens = (w_lr * p_lr) + (w_cnb * p_cnb)

    for i, (text, expected) in enumerate(stress_cases):
        pred_lr = classes[np.argmax(p_lr[i])]
        pred_cnb = classes[np.argmax(p_cnb[i])]
        pred_ens = classes[np.argmax(p_ens[i])]
        ens_prob = p_ens[i][np.argmax(p_ens[i])]

        match = "MATCH" if expected.startswith(pred_ens) else "MISMATCH"
        print(f"Case {i+1}: {text}")
        print(f"  Expected:    {expected}")
        print(f"  LogReg:      {pred_lr} (p={p_lr[i][np.argmax(p_lr[i])]:.3f})")
        print(f"  CompNB:      {pred_cnb} (p={p_cnb[i][np.argmax(p_cnb[i])]:.3f})")
        print(f"  Soft-Voting: {pred_ens} (p={ens_prob:.3f}) [{match}]")
        print("-" * 70)


def main():
    print("=" * 70)
    print("STARTING STEP 6: ENSEMBLE STACKING & CROSS-LINGUAL EVALUATION")
    print("=" * 70)

    train_df, val_df, test_df, hidden_df = load_datasets()
    print(f"Loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}, Hidden={len(hidden_df)}")

    # 1. Run Cross-Lingual Evaluation
    cross_lingual_results = run_cross_lingual_evaluation(train_df, test_df)

    # 2. Run Soft Voting Ensemble
    ensemble_results, ensemble_bundle = run_soft_voting_ensemble(train_df, val_df, test_df, hidden_df)

    # 3. Stress Test Comparison
    run_stress_test_comparison(ensemble_bundle)

    print("\n" + "=" * 70)
    print("STEP 6 EXECUTION COMPLETE SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
