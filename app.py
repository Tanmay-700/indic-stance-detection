"""
Indic Stance Detection - Interactive Web Frontend
Course: CSET 346 (Natural Language Processing) - Bennett University
Target Claim: "Climate change and global warming is a serious concern."

Supports:
- Pure Hindi (Devanagari) & Pure Bengali (Bangla)
- Full Emoji Preservation & Grapheme Clusters
- 4 Classical Machine Learning Baselines:
  1. Complement Naive Bayes (Champion Baseline)
  2. Logistic Regression (Balanced L2)
  3. Linear SVC (Maximum-Margin Hyperplane)
  4. Random Forest (100 Trees Ensemble)
- Side-by-side Model Comparison
- Real-time Preprocessing & Feature Attribution (XAI)
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
import gradio as gr

# Project Root setup
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_indic_text, indic_tokenize

# Fix Windows console UTF-8 output encoding
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# --------------------------------------------------------------------------
# LOAD ASSETS & MODELS
# --------------------------------------------------------------------------
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
VEC_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

print("Loading TF-IDF Vectorizer and ML Models...")
vectorizer = joblib.load(VEC_PATH)

# Load base models
_cnb = joblib.load(os.path.join(MODELS_DIR, "model_complementnb.pkl"))
_lr = joblib.load(os.path.join(MODELS_DIR, "model_logisticregression.pkl"))
_svc = joblib.load(os.path.join(MODELS_DIR, "model_linearsvc.pkl"))
_rf = joblib.load(os.path.join(MODELS_DIR, "model_randomforest.pkl"))

# Soft-Voting Ensemble Wrapper
class SoftVotingWrapper:
    def __init__(self, clf_lr, clf_cnb, classes, weights):
        self.clf_lr = clf_lr
        self.clf_cnb = clf_cnb
        self.classes_ = np.array(classes)
        self.weights = weights
        
    def predict_proba(self, X):
        p_lr = self.clf_lr.predict_proba(X)
        p_cnb = self.clf_cnb.predict_proba(X)
        w_lr, w_cnb = self.weights
        return (w_lr * p_lr) + (w_cnb * p_cnb)
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=-1)]

# MuRIL Contextual Transformer Wrapper
class MuRILWrapper:
    def __init__(self, model_dir):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print(f"Loading MuRIL Transformer from {model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.classes_ = np.array(["AGAINST", "FAVOUR", "NONE"])
        
    def predict_proba_from_text(self, texts):
        import torch
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict_from_text(self, texts):
        probs = self.predict_proba_from_text(texts)
        return self.classes_[np.argmax(probs, axis=-1)]

# Full Hybrid Ensemble (MuRIL Context + Dual TF-IDF LR)
class FullHybridEnsembleWrapper:
    def __init__(self, muril_wrapper, tfidf_lr, vectorizer, w_muril=0.70):
        self.muril = muril_wrapper
        self.tfidf_lr = tfidf_lr
        self.vectorizer = vectorizer
        self.w_muril = w_muril
        self.w_lr = 1.0 - w_muril
        self.classes_ = np.array(["AGAINST", "FAVOUR", "NONE"])
        
    def predict_proba_from_text(self, texts):
        p_muril = self.muril.predict_proba_from_text(texts)
        if isinstance(texts, str):
            texts = [texts]
        X = self.vectorizer.transform(texts)
        p_lr = self.tfidf_lr.predict_proba(X)
        return (self.w_muril * p_muril) + (self.w_lr * p_lr)
        
    def predict_from_text(self, texts):
        probs = self.predict_proba_from_text(texts)
        return self.classes_[np.argmax(probs, axis=-1)]

# Load saved ensemble bundle if available
ensemble_path = os.path.join(MODELS_DIR, "ensemble_soft_voting.pkl")
if os.path.exists(ensemble_path):
    bundle = joblib.load(ensemble_path)
    _ens = SoftVotingWrapper(_lr, _cnb, bundle["classes"], bundle["weights"])
else:
    _ens = SoftVotingWrapper(_lr, _cnb, ["AGAINST", "FAVOUR", "NONE"], (0.5, 0.5))

# Check for local MuRIL fine-tuned weights
muril_dir = os.path.join(MODELS_DIR, "transformer_muril")
has_muril = os.path.exists(os.path.join(muril_dir, "config.json"))

MODEL_REGISTRY = {}
if has_muril:
    try:
        _muril = MuRILWrapper(muril_dir)
        _full_ens = FullHybridEnsembleWrapper(_muril, _lr, vectorizer, w_muril=0.70)
        MODEL_REGISTRY["Full Hybrid Ensemble (MuRIL Context + Dual TF-IDF LR)"] = _full_ens
        MODEL_REGISTRY["Google MuRIL (Fine-Tuned Transformer)"] = _muril
    except Exception as e:
        print(f"Warning: Could not load MuRIL model from {muril_dir}: {e}")

MODEL_REGISTRY["Classical Ensemble (ComplementNB + Logistic Regression)"] = _ens
MODEL_REGISTRY["ComplementNB (Champion Baseline)"] = _cnb
MODEL_REGISTRY["Logistic Regression (Balanced L2)"] = _lr
MODEL_REGISTRY["Linear SVC (Max-Margin Hyperplane)"] = _svc
MODEL_REGISTRY["Random Forest (100 Trees Ensemble)"] = _rf

CLASSES = ["AGAINST", "FAVOUR", "NONE"]

STANCE_COLORS = {
    "FAVOUR": "#10b981",   # Green
    "AGAINST": "#ef4444",  # Red
    "NONE": "#64748b"      # Slate Gray
}

STANCE_DESCRIPTIONS = {
    "FAVOUR": "Agrees that climate change is a serious concern.",
    "AGAINST": "Denies, minimizes, or mocks global warming concerns.",
    "NONE": "Neutral question, unrelated talk, praise, or local weather comment."
}


# --------------------------------------------------------------------------
# PREDICTION LOGIC
# --------------------------------------------------------------------------

def predict_single(text: str, model_name: str):
    """Predicts stance for a single text using selected model."""
    if not text or not text.strip():
        return (
            "<div style='padding:15px; color:#94a3b8;'>Please enter a Hindi or Bengali comment.</div>",
            {},
            "",
            ""
        )
        
    model = MODEL_REGISTRY[model_name]
    
    # 1. Step 3 Preprocessing
    cleaned = clean_indic_text(text)
    tokens = indic_tokenize(cleaned)
    
    # 2. Prediction & Confidence
    if hasattr(model, "predict_from_text"):
        pred_label = model.predict_from_text([cleaned])[0]
        probs = model.predict_proba_from_text([cleaned])[0]
    else:
        X = vectorizer.transform([cleaned])
        pred_label = model.predict(X)[0]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
        elif hasattr(model, "decision_function"):
            dfunc = model.decision_function(X)[0]
            probs = softmax(dfunc)
        else:
            probs = np.array([1.0 if c == pred_label else 0.0 for c in model.classes_])
        
    prob_dict = {
        cls_name: round(float(prob), 4)
        for cls_name, prob in zip(model.classes_, probs)
    }
    
    # Styled HTML Badge
    color = STANCE_COLORS.get(pred_label, "#3b82f6")
    desc = STANCE_DESCRIPTIONS.get(pred_label, "")
    badge_html = f"""
    <div style='background-color:{color}18; border:2px solid {color}; border-radius:12px; padding:18px 24px; margin-bottom:10px;'>
        <div style='font-size:13px; text-transform:uppercase; letter-spacing:1px; color:#64748b; font-weight:600;'>Predicted Stance</div>
        <div style='font-size:32px; font-weight:800; color:{color}; margin:4px 0;'>{pred_label}</div>
        <div style='font-size:14px; color:#334155;'>{desc}</div>
    </div>
    """
    
    token_str = " | ".join(tokens)
    return badge_html, prob_dict, cleaned, token_str


def compare_all_models(text: str):
    """Runs all 4 baseline models concurrently on the same comment for side-by-side comparison."""
    if not text or not text.strip():
        return pd.DataFrame()
        
    cleaned = clean_indic_text(text)
    X = vectorizer.transform([cleaned])
    
    records = []
    for m_name, model in MODEL_REGISTRY.items():
        if hasattr(model, "predict_from_text"):
            pred = model.predict_from_text([cleaned])[0]
            probs = model.predict_proba_from_text([cleaned])[0]
            conf = float(np.max(probs)) * 100
        else:
            pred = model.predict(X)[0]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                conf = float(np.max(probs)) * 100
            elif hasattr(model, "decision_function"):
                dfunc = model.decision_function(X)[0]
                probs = softmax(dfunc)
                conf = float(np.max(probs)) * 100
            else:
                conf = 100.0
            
        records.append({
            "Model": m_name,
            "Predicted Stance": pred,
            "Top Confidence": f"{conf:.2f}%"
        })
        
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# GRADIO UI INTERFACE
# --------------------------------------------------------------------------

custom_css = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.gradio-container { max-width: 1050px !important; margin: 0 auto !important; }
"""

with gr.Blocks(title="Indic Stance Detection - Global Warming") as demo:
    gr.Markdown(
        """
        # 🌍 Indic Stance Detection on Global Warming
        ### Bennett University | Course CSET 346: Natural Language Processing
        **Target Claim:** *"Climate change and global warming is a serious concern."*  
        **Supported Languages:** Pure Hindi (Devanagari) & Pure Bengali (Bangla) + Contextual Emojis  
        **Class Labels:** `FAVOUR` (Green) | `AGAINST` (Red) | `NONE` (Slate)
        """
    )
    
    if not has_muril:
        gr.Markdown(
            """
            > 💡 **Note on Google MuRIL:** MuRIL was fine-tuned on Google Colab GPU. To run MuRIL inference inside this local UI, download `best_muril_model` from Colab into `models/transformer_muril/`. You can also launch this exact Gradio app directly inside Colab with GPU acceleration!
            """
        )
    
    with gr.Tabs():
        with gr.TabItem("🔍 Single Model Deep-Dive"):
            with gr.Row():
                with gr.Column(scale=6):
                    input_text = gr.Textbox(
                        label="Enter Social Media Comment (Hindi / Bengali with Emojis)",
                        placeholder="उदा: ग्लोबल वार्मिंग एक गंभीर संकट है, हमें पेड़ लगाने चाहिए 🌱 🌍",
                        lines=4
                    )
                    model_dropdown = gr.Dropdown(
                        label="Select ML / Ensemble Classifier",
                        choices=list(MODEL_REGISTRY.keys()),
                        value=list(MODEL_REGISTRY.keys())[0]
                    )
                    predict_btn = gr.Button("Analyze Stance", variant="primary", size="lg")
                    
                with gr.Column(scale=5):
                    output_badge = gr.HTML(label="Classification Outcome")
                    output_probs = gr.Label(label="Class Probability Distribution", num_top_classes=3)
                    
            with gr.Accordion("⚙️ Preprocessing & Tokenization Diagnostics (Step 3 Engine)", open=False):
                with gr.Row():
                    out_cleaned = gr.Textbox(label="Cleaned Normalized Text (Unicode NFC, De-elongated, Spaced Emojis)", lines=2)
                    out_tokens = gr.Textbox(label="Extracted Word & Emoji Tokens (indic_tokenize)", lines=2)
                    
            predict_btn.click(
                fn=predict_single,
                inputs=[input_text, model_dropdown],
                outputs=[output_badge, output_probs, out_cleaned, out_tokens]
            )

        with gr.TabItem("⚖️ Compare All Models Side-by-Side"):
            gr.Markdown("Type a challenging comment below (e.g. sarcastic comment or subtle minimization) to see how all classifiers (including the Ensemble) react:")
            with gr.Row():
                cmp_input = gr.Textbox(
                    label="Comment to Compare Across Models",
                    placeholder="उदा: हाँ भाई, ग्लोबल वार्मिंग तो शायद मौसम का मूड स्विंग है 😂",
                    lines=3
                )
            cmp_btn = gr.Button("Compare Classifiers", variant="primary")
            cmp_output = gr.Dataframe(
                headers=["Model", "Predicted Stance", "Top Confidence"],
                label="Model Agreement Table",
                interactive=False
            )
            
            cmp_btn.click(
                fn=compare_all_models,
                inputs=[cmp_input],
                outputs=[cmp_output]
            )
            
    gr.Markdown("### 💡 Click any preset example below to test immediately:")
    default_model = list(MODEL_REGISTRY.keys())[0]
    gr.Examples(
        examples=[
            ["जलवायु परिवर्तन पृथ्वी के अस्तित्व के लिए एक बड़ा संकट है, तत्काल कार्रवाई जरूरी है 🌱 🌍", default_model],
            ["हाँ भाई, ग्लोबल वार्मिंग तो शायद मौसम का मूड स्विंग है 😂 🤡", default_model],
            ["पेड़ काटने से कुछ नहीं होता, प्रकृति खुद सब संभाल लेती है", default_model],
            ["ग्लोबल वार्मिंग केवल पश्चिमी देशों का ढोंग और नाटक है 🙄", default_model],
            ["হিমালয়ের হিমবাহ দ্রুত গলে যাচ্ছে, পরিবেশ রক্ষা আমাদের সবার দায়িত্ব ⚠️ 🌱", default_model],
            ["পশ্চিমী ঢং বাদ দিন, মানুষ সৃষ্টি থেকেই সব ধরনের জলবায়ুতে মানিয়ে নিয়েছে 🤡", default_model],
            ["आज दिल्ली में सुबह से बहुत तेज़ बारिश हो रही है और मौसम ठंडा है ।", default_model],
            ["सर आपका वीडियो बहुत ज्ञानवर्धक है, चैनल को सब्सक्राइब कर दिया 👍 👏", default_model]
        ],
        inputs=[input_text, model_dropdown]
    )

if __name__ == "__main__":
    # Launch on localhost without opening browser automatically for background process stability
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,
        share=False
    )
