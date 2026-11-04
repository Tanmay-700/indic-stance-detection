"""
Prediction Module for Indic Stance Detection
Generates the official submission file containing 'id' and 'predicted_label'.
"""

import os
import sys
import argparse
import joblib
import pandas as pd
from preprocessing import IndicPreprocessor

# Reconfigure stdout for utf-8 on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def generate_predictions(
    input_file="d:/nlp/data/hidden_test_unlabeled.csv",
    output_file="d:/nlp/predictions/test_predictions.csv",
    model_path="d:/nlp/models/best_ensemble_model.pkl",
    vectorizer_path="d:/nlp/models/tfidf_vectorizer.pkl"
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"[PREDICT] Loading input file: {input_file}")
    df = pd.read_csv(input_file, keep_default_na=False)
    
    if "cleaned_text" not in df.columns:
        print("[PREDICT] Applying text preprocessing...")
        preprocessor = IndicPreprocessor(remove_stopwords=False)
        df["cleaned_text"] = df.apply(
            lambda r: preprocessor.preprocess(r["text"], r.get("language", "Hindi")), axis=1
        )
        
    print(f"[PREDICT] Loading model from: {model_path}")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    X_vec = vectorizer.transform(df["cleaned_text"].fillna(""))
    preds = model.predict(X_vec)
    
    submission_df = pd.DataFrame({
        "id": df["id"],
        "predicted_label": preds
    })
    
    submission_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"[PREDICT] Successfully generated predictions for {len(submission_df)} samples!")
    print(f"[PREDICT] Submission file saved to: {output_file}")
    
    print("\nPrediction Label Distribution:")
    print(submission_df["predicted_label"].value_counts())
    return submission_df

def predict_single_comment(comment_text, language="Hindi", model_path="d:/nlp/models/best_ensemble_model.pkl", vectorizer_path="d:/nlp/models/tfidf_vectorizer.pkl"):
    preprocessor = IndicPreprocessor(remove_stopwords=False)
    cleaned = preprocessor.preprocess(comment_text, language)
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0] if hasattr(model, "predict_proba") else None
    
    print(f"\nComment:  '{comment_text}'")
    print(f"Language: {language}")
    print(f"Stance:   {pred}")
    if proba is not None:
        classes = getattr(model, "classes_", ["Against", "Favour", "None"])
        probs = {c: f"{p:.4f}" for c, p in zip(classes, proba)}
        print(f"Confidence: {probs}")
    return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="d:/nlp/data/hidden_test_unlabeled.csv")
    parser.add_argument("--output", type=str, default="d:/nlp/predictions/test_predictions.csv")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--lang", type=str, default="Hindi")
    args = parser.parse_args()
    
    if args.text:
        predict_single_comment(args.text, args.lang)
    else:
        generate_predictions(args.input, args.output)
