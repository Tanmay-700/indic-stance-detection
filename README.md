# Indic Stance Detection on Global Warming (Hindi & Bengali)

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-orange.svg)](https://huggingface.co/)
[![MuRIL](https://img.shields.io/badge/Model-Google%20MuRIL-green.svg)](https://huggingface.co/google/muril-base-cased)
[![License](https://img.shields.io/badge/Course-CSET%20346%20NLP-purple.svg)](https://www.bennett.edu.in/)

**Course:** CSET 346 Natural Language Processing  
**Institution:** Bennett University (Times Group)  
**Instructor:** Dr. Supriya Chanda  
**Team / Contributor:** Tanmay Goyal (Solo Lead)  

---

## 🎯 Target Claim & Objective

> **Target Claim:** *"Climate change and global warming is a serious concern."*  
> **Stance Classes:** `FAVOUR` | `AGAINST` | `NONE`  
> **Languages:** Pure **Hindi** (Devanagari script) & Pure **Bengali** (Bangla script)

Stance detection evaluates the posture of an author toward a specific target claim. While most NLP research focuses on English, this project constructs a complete, reproducible pipeline for two of South Asia's most widely spoken languages, addressing low-resource challenges, dialectal noise, and semantic minimization.

---

## 📊 Dataset Statistics & Organization

The curated corpus consists of **3,520 labeled comments** balanced equally across Hindi and Bengali, covering 11 realistic environmental domains (deforestation, glacier retreat, renewable energy, floods/cyclones, vehicular pollution, climate denial myths, and conversational banter):

| Partition | Total Samples | Hindi (Devanagari) | Bengali (Bangla) | FAVOUR | AGAINST | NONE | Purpose |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Train** | 2,560 | 1,280 | 1,280 | 854 | 853 | 853 | 80% &bull; Model fitting |
| **Validation** | 320 | 160 | 160 | 107 | 107 | 106 | 10% &bull; Hyperparameter tuning |
| **Test** | 320 | 160 | 160 | 106 | 96 | 118 | 10% &bull; Unbiased primary evaluation |
| **Hidden Test** | 320 | 160 | 160 | — | — | — | Unlabeled external evaluation |
| **Total** | **3,520** | **1,760** | **1,760** | **1,067** | **1,056** | **1,077** | Balanced 3-class distribution |

**Noise Engine:** 35% Clean formal text, 50% Light noise (matra elongations, emojis, punctuation spam), 15% Heavy noise (sarcastic negation, colloquial idioms).

---

## 🔬 Experimental Results

All models evaluated on the held-out 320 test samples (160 Hindi, 160 Bengali). The primary evaluation metric is **Macro-Averaged F1**:

| Model Architecture | Features / Parameters | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 **Full Hybrid Ensemble (MuRIL 70% + LR 30%)** | 110M + 40,260 | **0.9844** | **0.9848** | **0.9841** | **0.9844** |
| 🥈 **Google MuRIL (Fine-Tuned Transformer)** | 110M | 0.9719 | 0.9725 | 0.9714 | 0.9718 |
| 🥉 **Classical Soft-Voting (LR 50% + CNB 50%)** | 40,260 | 0.9562 | 0.9568 | 0.9557 | 0.9562 |
| Logistic Regression (Balanced L2) | 40,260 | 0.9469 | 0.9475 | 0.9462 | 0.9468 |
| Linear Support Vector Classifier (LinearSVC) | 40,260 | 0.9344 | 0.9352 | 0.9338 | 0.9343 |
| Complement Naive Bayes (CNB) | 40,260 | 0.9188 | 0.9204 | 0.9176 | 0.9187 |
| Random Forest Classifier (200 Trees) | 40,260 | 0.8938 | 0.8951 | 0.8924 | 0.8936 |

### Cross-Lingual Zero-Shot Transfer Matrix (Macro F1)

| Training Configuration | Train Samples | Hindi Test (160) | Bengali Test (160) | Transfer Nature |
|:---|:---:|:---:|:---:|:---|
| **Hindi-Only MuRIL** | 1,280 | **0.9688** | 0.6684 | In-Language &bull; Zero-Shot Cross-Lingual |
| **Bengali-Only MuRIL** | 1,280 | 0.5982 | **0.9625** | Zero-Shot Cross-Lingual &bull; In-Language |
| **Joint Bilingual MuRIL** | 2,560 | **0.9750** | **0.9688** | Unified Multilingual Joint Training |

### Semantic Minimization Adversarial Stress Test
Minimization comment: *"ग्लोबल वार्मिंग थोड़ा ज्यादा नहीं हो रहा? 🤔"* (Actual: `AGAINST`):
- **Standalone MuRIL:** `p(AGAINST)=0.412`, `p(NONE)=0.473` &rarr; ❌ Predicted `NONE`
- **Standalone ComplementNB:** `p(AGAINST)=0.298`, `p(FAVOUR)=0.514` &rarr; ❌ Predicted `FAVOUR`
- **Full Hybrid Ensemble:** `p(AGAINST)=0.628` &rarr; **✓ Correctly Predicted `AGAINST`**

---

## 🛠️ Project Architecture & Repository Structure

```
d:\nlp\
├── data/
│   ├── raw/
│   │   └── indic_stance_corpus_3200.csv           # Master curated raw corpus
│   └── processed/
│       ├── train.csv                              # 2,560 training samples
│       ├── val.csv                                # 320 validation samples
│       ├── test.csv                               # 320 primary evaluation test samples
│       └── hidden_test_unlabeled.csv              # 320 unlabeled hidden evaluation samples
├── models/
│   ├── tfidf_vectorizer.pkl                       # 40,260-dim Dual TF-IDF FeatureUnion
│   ├── model_complementnb.pkl                     # Serialized ComplementNB
│   ├── model_logisticregression.pkl               # Serialized Logistic Regression
│   ├── model_linearsvc.pkl                        # Serialized LinearSVC
│   ├── model_randomforest.pkl                     # Serialized Random Forest
│   ├── ensemble_soft_voting.pkl                   # Calibrated Soft-Voting model
│   └── transformer_muril/                         # Fine-tuned Google MuRIL weights (safetensors)
├── notebooks/
│   └── Indic_Stance_Detection_Colab.ipynb         # 31-cell turnkey Colab GPU training notebook
├── predictions/
│   ├── baseline_predictions.csv                   # Test set predictions for 4 baselines
│   └── ensemble_predictions.csv                   # Test set predictions for ensembles
├── report/
│   ├── Round2_Interactive_Report.html         # Main IEEE-Style Interactive Technical Report & Simulator
│   ├── Round2_Progress_Report.pdf             # Formal Two-Page Progress Report (Mandatory)
│   ├── Round2_Progress_Report.html            # Printable 2-page HTML source
│   ├── baseline_results.json                      # Machine-readable baseline metrics
│   ├── ensemble_results.json                      # Machine-readable ensemble metrics
│   ├── cross_lingual_results.json                 # 6-cell cross-lingual transfer matrix
│   └── figures/
│       ├── cm_logisticregression.png              # Confusion matrix: Logistic Regression
│       ├── cm_complementnb.png                    # Confusion matrix: Complement Naive Bayes
│       ├── cm_ensemble.png                        # Confusion matrix: Soft-Voting Ensemble
│       └── cm_cross_lingual_transfer.png          # Confusion matrix: Zero-shot transfer
├── src/
│   ├── collect_and_curate_data.py                 # Step 2: Corpus curation & noise engine
│   ├── preprocessing.py                           # Step 3: Unicode NFC & tokenization
│   ├── baselines.py                               # Step 4: Classical ML baseline training
│   ├── transformers_train.py                      # Step 5: MuRIL transformer fine-tuning
│   └── ensemble.py                                # Step 6: Ensemble stacking & cross-lingual
├── app.py                                         # Interactive 7-Model Gradio Web UI
└── README.md                                      # Project documentation
```

---

## 🚀 How to Run Code Locally

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Tanmay-700/indic-stance-detection.git
cd indic-stance-detection

# Install Python requirements
pip install -r requirements.txt
# Or manually:
pip install torch transformers scikit-learn pandas numpy gradio matplotlib seaborn joblib
```

### 2. Execution Pipeline (Sequential Steps 2–6)
```bash
# Step 2: Synthesize and split the balanced corpus
python src/collect_and_curate_data.py

# Step 3: Run Unicode NFC normalization & tokenization
python src/preprocessing.py

# Step 4: Train TF-IDF FeatureUnion and 4 classical baselines
python src/baselines.py

# Step 5: Fine-tune Google MuRIL on Tesla T4 GPU (or run in Colab)
python src/transformers_train.py

# Step 6: Evaluate soft-voting ensemble & cross-lingual transfer
python src/ensemble.py
```

### 3. Launch Interactive Gradio Web Demo
```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860/` to test live stance classification across all 7 trained models with confidence probability graphs.

---

## 🔮 Future Work Before Final Evaluation (Round 3)

1. **Step 7 — Explainable AI (XAI):** Integrate LIME and Integrated Gradients for token-level visual heatmaps explaining feature importance for Indic stance decisions.
2. **Step 8 — ACL/IEEE Academic Research Paper:** Prepare an 8-page publication detailing dataset creation and zero-shot cross-lingual transfer dynamics.
3. **Code-Mixed Expansion:** Broaden evaluation to Romanized Hinglish and Banglish text.
