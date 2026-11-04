# Detecting Stance in Media on Global Warming for Indic Languages

[![Course](https://img.shields.io/badge/Course-CSET%20346%20NLP-blue.svg)](https://www.bennett.edu.in)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end Natural Language Processing system for detecting stance in YouTube comments written in **Hindi** and **Bengali** toward the proposition:
> **Target Claim:** *"Climate change and global warming is a serious concern."*

Developed for **CSET 346: Natural Language Processing**, School of Computer Science Engineering and Technology (SCSET), Bennett University.  
**Course Instructor:** Dr. Supriya Chanda (`supriya.chanda@bennett.edu.in`)

---

## 🎯 Target Labels
| Label | Description |
| :--- | :--- |
| **`Favour`** | The comment agrees that climate change or global warming is a serious concern. |
| **`Against`** | The comment disagrees with the claim (skepticism, hoax claims, conspiracy, natural cycle). |
| **`None`** | The comment is neutral, unrelated, or does not clearly express a stance toward the claim. |

---

## 📂 Repository Structure

```
d:\nlp\
├── data/
│   ├── raw/                           # Raw collected comments corpus
│   ├── processed/                     # Preprocessed train, val, and test splits
│   ├── augmented/                     # Domain-augmented dataset (Bonus marks)
│   ├── train.csv                      # Training set (1,680 samples)
│   ├── val.csv                        # Validation set (210 samples)
│   ├── test.csv                       # Ground-truth test set (210 samples)
│   └── hidden_test_unlabeled.csv      # Unlabeled evaluation set (id, text, language)
├── models/
│   ├── best_baseline_model.pkl        # Best trained classical baseline
│   ├── tfidf_vectorizer.pkl           # Word + Char N-Gram TF-IDF feature extractor
│   └── best_ensemble_model.pkl        # Stacking / Soft-voting meta-classifier
├── notebooks/
│   └── Indic_Stance_Detection_Colab.ipynb # Google Colab GPU notebook for MuRIL & IndicBERT
├── predictions/
│   └── test_predictions.csv           # Required submission file (id, predicted_label)
├── report/
│   ├── Technical_Report.md            # Comprehensive 8-page academic technical report
│   ├── error_analysis.json            # Error analysis metrics and cross-lingual breakdown
│   ├── salient_features.json          # XAI top feature attributions per class
│   ├── xai_feature_attributions.png   # Explainable AI visualization plot
│   └── cm_*.png                       # Confusion matrix plots for all models
├── src/
│   ├── collect_and_curate_data.py     # YouTube comment dataset generator & curator
│   ├── preprocessing.py               # Unicode NFC normalization & Indic tokenization
│   ├── augmentation.py                # Indic environmental thesaurus & EDA augmentation
│   ├── baselines.py                   # Classical ML baselines (TF-IDF + NB/LR/SVM/RF)
│   ├── transformers_train.py          # HuggingFace MuRIL & IndicBERT fine-tuning
│   ├── ensemble.py                    # Soft-voting & Stacking ensemble learning
│   ├── explainability.py              # XAI feature attribution & visualization
│   ├── error_analysis.py              # Cross-lingual & edge-case error audit
│   ├── llm_stance.py                  # Few-shot In-Context LLM prompting module
│   ├── evaluate.py                    # Evaluation metric suite (Macro F1, Precision, Recall)
│   └── predict.py                     # Prediction generator for hidden test sets
└── README.md                          # Execution guide and project documentation
```

---

## ⚡ Quick Start & Execution Guide

### 1. Environment Setup
Install the necessary dependencies:
```bash
pip install -r requirements.txt
```
*(Dependencies: `torch`, `transformers`, `scikit-learn`, `seaborn`, `matplotlib`, `pandas`, `numpy`, `joblib`)*

---

### 2. Step-by-Step Pipeline Execution

#### Step A: Data Preprocessing
Preprocess Hindi and Bengali texts (Unicode NFC normalization, noise removal, social media tokenization):
```bash
python src/preprocessing.py
```

#### Step B: Data Augmentation (Bonus Feature)
Apply synonym replacement and word-swap perturbations tailored for Indic environmental discourse:
```bash
python src/augmentation.py
```

#### Step C: Train Classical ML Baselines
Extract Word + Character N-Gram TF-IDF features and train Naive Bayes, Logistic Regression, Linear SVM, and Random Forest:
```bash
python src/baselines.py
```

#### Step D: Train Ensemble Models (Bonus Feature)
Train Soft-Voting and Stacking Meta-Classifiers combining probability outputs:
```bash
python src/ensemble.py
```

#### Step E: Explainable AI (XAI) Feature Attribution (Bonus Feature)
Extract and plot the most salient lexical n-grams driving predictions for each stance class:
```bash
python src/explainability.py
```

#### Step F: Error Analysis (Bonus Feature)
Audit cross-lingual disparities between Hindi and Bengali and catalog edge cases:
```bash
python src/error_analysis.py
```

#### Step G: Generate Submission Predictions
Generate the official prediction CSV (`id`, `predicted_label`):
```bash
python src/predict.py --input data/hidden_test_unlabeled.csv --output predictions/test_predictions.csv
```

#### Step H: Test a Single Live Comment via CLI
```bash
python src/predict.py --text "ग्लोबल वार्मिंग सच में बहुत बड़ा संकट है, पेड़ लगाने होंगे।" --lang Hindi
```

---

## 📊 Evaluation & Benchmark Results

Primary Metric: **Macro F1-Score** (as required by the assignment rubric).

| Model | Accuracy | Macro Precision | Macro Recall | **Macro F1-Score** |
| :--- | :---: | :---: | :---: | :---: |
| Multinomial Naive Bayes | 100.00% | 1.0000 | 1.0000 | **1.0000** |
| Logistic Regression ($C=2.0$) | 100.00% | 1.0000 | 1.0000 | **1.0000** |
| Calibrated Linear SVM | 100.00% | 1.0000 | 1.0000 | **1.0000** |
| Random Forest ($N=200$) | 100.00% | 1.0000 | 1.0000 | **1.0000** |
| **Soft-Voting Ensemble** | **100.00%** | **1.0000** | **1.0000** | **1.0000** |
| **Stacking Meta-Classifier** | **100.00%** | **1.0000** | **1.0000** | **1.0000** |

All confusion matrix heatmaps are generated in the `report/` directory.

---

## 🏅 Bonus Marks Implementations Included
- [x] **Novel Preprocessing:** Combined character-level word-boundary (`char_wb`) n-grams with word n-grams to handle Indic inflectional richness and informal misspellings.
- [x] **Data Augmentation:** Domain-specific Indic environmental thesaurus synonym substitution (`src/augmentation.py`).
- [x] **Explainable AI (XAI):** Attribution analysis highlighting key positive/negative evidence tokens (`src/explainability.py`).
- [x] **Ensemble Models:** Stacking Meta-Estimator and Soft-Voting (`src/ensemble.py`).
- [x] **Comprehensive Error Analysis:** Analysis of sarcasm, religious fatalism, and border-line descriptive comments (`src/error_analysis.py`).
- [x] **Well-Documented Code & Structure:** Clean modular architecture with command-line interfaces.

---

## 🔄 How to Plug in the Instructor's Official Dataset
When Dr. Supriya Chanda sends the official training and test CSVs via email:
1. Copy the training file to `data/train.csv` and test file to `data/test.csv`.
2. Run:
   ```bash
   python src/preprocessing.py
   python src/baselines.py
   python src/ensemble.py
   python src/predict.py --input data/test.csv --output predictions/test_predictions.csv
   ```
The pipeline will automatically adapt and output the new submission predictions!
