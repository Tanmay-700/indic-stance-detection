"""
Ensemble Learning Module for Indic Stance Detection (Bonus Marks Requirement)
Implements:
1. Soft-Voting (Weighted probability averaging)
2. Stacking Meta-Classifier (Meta-learner on base probability predictions)
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from evaluate import evaluate_predictions

def train_and_eval_ensemble(
    train_path="d:/nlp/data/processed/train.csv",
    val_path="d:/nlp/data/processed/val.csv",
    test_path="d:/nlp/data/processed/test.csv",
    vectorizer_path="d:/nlp/models/tfidf_vectorizer.pkl",
    output_dir="d:/nlp/models",
    report_dir="d:/nlp/report"
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    print("[ENSEMBLE] Loading processed datasets and TF-IDF vectorizer...")
    train_df = pd.read_csv(train_path, keep_default_na=False)
    val_df = pd.read_csv(val_path, keep_default_na=False)
    test_df = pd.read_csv(test_path, keep_default_na=False)
    
    vectorizer = joblib.load(vectorizer_path)
    
    X_train = vectorizer.transform(train_df["cleaned_text"].fillna(""))
    y_train = train_df["stance"]
    
    X_val = vectorizer.transform(val_df["cleaned_text"].fillna(""))
    y_val = val_df["stance"]
    
    X_test = vectorizer.transform(test_df["cleaned_text"].fillna(""))
    y_test = test_df["stance"]
    
    # Define Base Estimators
    base_estimators = [
        ("nb", MultinomialNB(alpha=0.5)),
        ("lr", LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=42)),
        ("svm", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, class_weight="balanced"))),
        ("rf", RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42))
    ]
    
    # 1. Soft-Voting Ensemble
    print("\n--- Training Soft-Voting Ensemble ---")
    voting_clf = VotingClassifier(
        estimators=base_estimators,
        voting="soft",
        weights=[1.0, 1.5, 1.5, 0.8]
    )
    voting_clf.fit(X_train, y_train)
    vote_preds = voting_clf.predict(X_test)
    
    cm_voting_path = os.path.join(report_dir, "cm_ensemble_soft_voting.png")
    voting_metrics = evaluate_predictions(y_test, vote_preds, model_name="Soft-Voting Ensemble", save_cm_path=cm_voting_path)
    
    # 2. Stacking Meta-Classifier
    print("\n--- Training Stacking Meta-Classifier ---")
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=500),
        cv=5
    )
    stacking_clf.fit(X_train, y_train)
    stack_preds = stacking_clf.predict(X_test)
    
    cm_stack_path = os.path.join(report_dir, "cm_ensemble_stacking.png")
    stack_metrics = evaluate_predictions(y_test, stack_preds, model_name="Stacking Classifier", save_cm_path=cm_stack_path)
    
    # Save the best ensemble
    best_ensemble = stacking_clf if stack_metrics["macro_f1"] >= voting_metrics["macro_f1"] else voting_clf
    ensemble_path = os.path.join(output_dir, "best_ensemble_model.pkl")
    joblib.dump(best_ensemble, ensemble_path)
    print(f"\n[ENSEMBLE] Saved best ensemble model to: {ensemble_path}")
    
    return {
        "soft_voting": voting_metrics,
        "stacking": stack_metrics
    }

if __name__ == "__main__":
    train_and_eval_ensemble()
