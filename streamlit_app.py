"""
Streamlit Web UI Dashboard for Indic Stance Detection
Run with: streamlit run streamlit_app.py
"""

import os
import sys
import joblib
import pandas as pd
import streamlit as st

# Setup paths
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import IndicPreprocessor

MODEL_PATH = "d:/nlp/models/best_ensemble_model.pkl"
VECTORIZER_PATH = "d:/nlp/models/tfidf_vectorizer.pkl"

st.set_page_config(
    page_title="Indic Stance Detection - Global Warming",
    page_icon="🌍",
    layout="wide"
)

# Load model and vectorizer
@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    preprocessor = IndicPreprocessor(remove_stopwords=False)
    return model, vectorizer, preprocessor

try:
    model, vectorizer, preprocessor = load_resources()
    classes = getattr(model, "classes_", ["Against", "Favour", "None"])
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

def detect_language(text: str) -> str:
    has_bengali = any("\u0980" <= ch <= "\u09FF" for ch in text)
    has_hindi = any("\u0900" <= ch <= "\u097F" for ch in text)
    if has_bengali and not has_hindi:
        return "Bengali"
    return "Hindi"

# Header
st.title("🌍 Indic Stance Detection on Global Warming")
st.markdown("### Course: **CSET 346 – Natural Language Processing** (Bennett University)")
st.info("**Target Claim:** *\"Climate change and global warming is a serious concern.\"* | **Languages:** Hindi & Bengali")

# Layout
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📝 Enter YouTube Comment to Test")
    
    # Example selectors
    example_choice = st.selectbox(
        "💡 Or choose a pre-loaded sample:",
        [
            "-- Custom Input --",
            "[Hindi Favour] ग्लोबल वार्मिंग सच में एक बहुत बड़ा और गंभीर खतरा बन चुका है। हमें तुरंत पेड़ लगाने होंगे।",
            "[Hindi Against] यह सब पश्चिमी देशों का ढोंग और षड्यंत्र है, भारत के विकास को रोकने की साजिश है।",
            "[Hindi None] सर आपकी रिपोर्टिंग बहुत शानदार है, चैनल को लाइक और सब्सक्राइब कर दिया है।",
            "[Bengali Favour] সুন্দরবনের নদীভাঙন আর বারবার আছড়ে পড়া সুপার সাইক্লোন প্রমাণ করে বিশ্ব উষ্ণায়ন কত বড় বিপদ।",
            "[Bengali Against] সব ভুয়া ও মিথ্যা প্রচার, পশ্চিমারা নিজেদের অপকর্ম ঢাকতে এই নাটক সাজিয়েছে।",
            "[Bengali None] দাদা ভিডিওটি অনেক তথ্যবহুল ছিল, সুন্দর সাবলীল ব্যাখ্যার জন্য ধন্যবাদ।"
        ]
    )
    
    default_text = "" if example_choice == "-- Custom Input --" else example_choice.split("] ")[-1]
    user_input = st.text_area("YouTube Comment Text:", value=default_text, height=120)
    
    predict_clicked = st.button("🚀 Analyze Stance", type="primary")

with col2:
    st.subheader("🎯 Stance Prediction Result")
    
    if predict_clicked or default_text:
        if user_input.strip():
            lang = detect_language(user_input)
            cleaned = preprocessor.preprocess(user_input, lang)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0] if hasattr(model, "predict_proba") else None
            
            # Badge color
            if pred == "Favour":
                st.success(f"### Predicted Stance: **{pred.upper()}** (In Favour of Concern)")
            elif pred == "Against":
                st.error(f"### Predicted Stance: **{pred.upper()}** (Against the Concern / Skeptic)")
            else:
                st.warning(f"### Predicted Stance: **{pred.upper()}** (Neutral / Unrelated / None)")
                
            st.write(f"**Detected Language:** `{lang}`")
            st.write(f"**Cleaned Text:** *\"{cleaned}\"*")
            
            if proba is not None:
                st.markdown("#### Confidence Probabilities:")
                for c, p in zip(classes, proba):
                    st.write(f"**{c}**: `{p*100:.2f}%`")
                    st.progress(float(p))
        else:
            st.warning("Please enter some text to analyze.")
    else:
        st.write("👈 Enter a comment on the left or select an example to view the stance prediction.")

st.markdown("---")

# Visual Analytics Expander
with st.expander("📊 View Model Benchmarks & Confusion Matrix Heatmaps"):
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image("report/cm_ensemble_stacking.png", caption="Stacking Ensemble 3x3 Confusion Matrix")
    with col_img2:
        st.image("report/xai_feature_attributions.png", caption="Explainable AI: Top Salient Tokens per Class")
