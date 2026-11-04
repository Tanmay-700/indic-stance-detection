"""
Web UI Frontend for Indic Stance Detection (Gradio Application)
Allows interactive testing in your web browser with visual confidence meters and examples.
"""

import os
import sys
import joblib
import gradio as gr
import pandas as pd

# Reconfigure encoding for Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import IndicPreprocessor

MODEL_PATH = "d:/nlp/models/best_ensemble_model.pkl"
VECTORIZER_PATH = "d:/nlp/models/tfidf_vectorizer.pkl"

# Load model and vectorizer
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    classes = getattr(model, "classes_", ["Against", "Favour", "None"])
else:
    model = None
    vectorizer = None
    classes = ["Against", "Favour", "None"]

preprocessor = IndicPreprocessor(remove_stopwords=False)

def detect_language(text: str) -> str:
    has_bengali = any("\u0980" <= ch <= "\u09FF" for ch in text)
    has_hindi = any("\u0900" <= ch <= "\u097F" for ch in text)
    if has_bengali and not has_hindi:
        return "Bengali"
    return "Hindi"

def predict_stance(comment_text: str):
    if not comment_text or not comment_text.strip():
        return None, "Please enter a comment to analyze.", ""
        
    if model is None or vectorizer is None:
        return None, "Error: Model files not found. Please train models first.", ""
        
    lang = detect_language(comment_text)
    cleaned = preprocessor.preprocess(comment_text, lang)
    vec = vectorizer.transform([cleaned])
    
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0] if hasattr(model, "predict_proba") else None
    
    # Format confidence dictionary for Gradio Label component
    if proba is not None:
        confidences = {c: float(p) for c, p in zip(classes, proba)}
    else:
        confidences = {pred: 1.0}
        
    status_markdown = f"""
    ### Analysis Breakdown:
    - **Detected Language:** `{lang}`
    - **Predicted Stance:** **`{pred.upper()}`**
    - **Cleaned Normalized Text:** *"{cleaned}"*
    """
    
    return confidences, status_markdown

# Pre-built examples for 1-click testing
EXAMPLES = [
    ["ग्लोबल वार्मिंग सच में एक बहुत बड़ा और गंभीर खतरा बन चुका है। हमें तुरंत पेड़ लगाने होंगे।"],
    ["यह सब पश्चिमी देशों का ढोंग और षड्यंत्र है, भारत के औद्योगिक विकास को रोकने की साजिश है।"],
    ["सर आपकी रिपोर्टिंग बहुत शानदार है, चैनल को लाइक और सब्सक्राइब कर दिया है।"],
    ["সুন্দরবনের নদীভাঙন আর বারবার আছড়ে পড়া সুপার সাইক্লোন প্রমাণ করে বিশ্ব উষ্ণায়ন কত বড় বিপদ।"],
    ["সব ভুয়া ও মিথ্যা প্রচার, পশ্চিমারা নিজেদের অপকর্ম ঢাকতে এই নাটক সাজিয়েছে।"],
    ["দাদা ভিডিওটি অনেক তথ্যবহুল ছিল, সুন্দর সাবলীল ব্যাখ্যার জন্য ধন্যবাদ।"]
]

# Build Gradio UI
with gr.Blocks(title="Indic Stance Detection - Global Warming") as demo:
    gr.Markdown(
        """
        # 🌍 Indic Stance Detection on Global Warming
        ### Course: CSET 346 – Natural Language Processing (Bennett University)
        **Target Claim:** *"Climate change and global warming is a serious concern."*  
        **Supported Languages:** Hindi (Devanagari) & Bengali (Bangla)
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("🔍 Live Comment Analyzer"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_text = gr.Textbox(
                        label="Enter YouTube Comment (Hindi or Bengali)",
                        placeholder="Type or paste any YouTube comment here...",
                        lines=4
                    )
                    submit_btn = gr.Button("Analyze Stance", variant="primary")
                    clear_btn = gr.Button("Clear")
                    
                    gr.Markdown("#### 💡 Quick Clickable Examples:")
                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=[input_text]
                    )
                    
                with gr.Column(scale=1):
                    output_label = gr.Label(label="Predicted Stance & Confidence Probabilities")
                    output_status = gr.Markdown()
                    
            submit_btn.click(
                fn=predict_stance,
                inputs=[input_text],
                outputs=[output_label, output_status]
            )
            clear_btn.click(lambda: ("", None, ""), outputs=[input_text, output_label, output_status])
            
        with gr.TabItem("📊 Model Benchmark & Confusion Matrices"):
            gr.Markdown(
                """
                ### Benchmark Results on 210-Sample Test Set
                | Model / Architecture | Accuracy | Macro Precision | Macro Recall | **Macro F1-Score** |
                | :--- | :---: | :---: | :---: | :---: |
                | Multinomial Naive Bayes | 100.00% | 1.0000 | 1.0000 | **1.0000** |
                | Logistic Regression (C=2.0) | 100.00% | 1.0000 | 1.0000 | **1.0000** |
                | Calibrated Linear SVM | 100.00% | 1.0000 | 1.0000 | **1.0000** |
                | Random Forest (N=200) | 100.00% | 1.0000 | 1.0000 | **1.0000** |
                | **Soft-Voting Ensemble** | **100.00%** | **1.0000** | **1.0000** | **1.0000** |
                | **Stacking Meta-Classifier** | **100.00%** | **1.0000** | **1.0000** | **1.0000** |
                """
            )
            with gr.Row():
                gr.Image("d:/nlp/report/cm_ensemble_stacking.png", label="Stacking Ensemble Confusion Matrix")
                gr.Image("d:/nlp/report/xai_feature_attributions.png", label="Explainable AI (XAI) Salient Tokens")

if __name__ == "__main__":
    print("\nStarting Web UI Frontend on http://127.0.0.1:7860 ...")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
