"""
Interactive Testing Tool for Indic Stance Detection
Allows users to type any Hindi or Bengali comment and get instant real-time predictions.
"""

import os
import sys
import joblib

# Reconfigure stdin/stdout for utf-8 on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stdin.encoding != "utf-8":
    try:
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

from preprocessing import IndicPreprocessor

def detect_language(text: str) -> str:
    # Check Unicode script ranges: Bengali: \u0980-\u09FF, Devanagari: \u0900-\u097F
    has_bengali = any("\u0980" <= ch <= "\u09FF" for ch in text)
    has_hindi = any("\u0900" <= ch <= "\u097F" for ch in text)
    
    if has_bengali and not has_hindi:
        return "Bengali"
    return "Hindi"

def main():
    model_path = "d:/nlp/models/best_ensemble_model.pkl"
    vectorizer_path = "d:/nlp/models/tfidf_vectorizer.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("[ERROR] Model files not found. Please run 'python src/baselines.py' first.")
        return
        
    print("=" * 65)
    print("      🌿 Indic Stance Detection System - Live Testing Tool 🌿")
    print("Claim: 'Climate change and global warming is a serious concern.'")
    print("Classes: Favour | Against | None")
    print("=" * 65)
    print("Type any comment in Hindi or Bengali (or type 'exit' / 'q' to quit):\n")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    preprocessor = IndicPreprocessor(remove_stopwords=False)
    classes = getattr(model, "classes_", ["Against", "Favour", "None"])
    
    while True:
        try:
            comment = input("Enter comment > ").strip()
        except (KeyboardInterrupt, EOFError):
            break
            
        if not comment:
            continue
        if comment.lower() in ["exit", "quit", "q"]:
            print("\nExiting. Thank you!")
            break
            
        lang = detect_language(comment)
        cleaned = preprocessor.preprocess(comment, lang)
        vec = vectorizer.transform([cleaned])
        
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0] if hasattr(model, "predict_proba") else None
        
        # Color coding for terminal if supported
        print("\n" + "-" * 50)
        print(f" Language Detected : {lang}")
        print(f" Predicted Stance  : >>> {pred.upper()} <<<")
        if proba is not None:
            print(" Confidence Breakdown:")
            for c, p in zip(classes, proba):
                bar = "█" * int(p * 20)
                print(f"   • {c:8s}: {p*100:5.1f}% | {bar}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
