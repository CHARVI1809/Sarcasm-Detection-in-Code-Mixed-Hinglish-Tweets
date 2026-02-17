# Sarcasm Detection in Code-Mixed Hinglish Tweets

An NLP project that automatically detects sarcasm in Hinglish (Hindi-English code-mixed) tweets using Machine Learning and Transformer-based models.

---

## Overview

Sarcasm detection is a challenging task in Natural Language Processing because sarcasm depends on context and implicit meaning. This challenge becomes even more difficult in code-mixed languages like Hinglish.

This project builds and compares multiple models to classify tweets as:

- Sarcastic  
- Non-Sarcastic  

Three different approaches were implemented and compared:

- Baseline Machine Learning Models  
- SMOTE-based Balanced Models  
- Transformer-based Multilingual BERT (mBERT)  

The transformer model achieved the best performance.

---

## Key Features

- Works on real Hinglish tweet dataset  
- Handles class imbalance using SMOTE  
- Compares traditional ML and Transformer models  
- Uses contextual embeddings with mBERT  
- Achieves high sarcasm detection performance  

---

## Dataset

- Hinglish code-mixed tweets dataset  
- Binary classification:
  - 1 → Sarcastic  
  - 0 → Non-Sarcastic  
- Highly imbalanced dataset  

Example:
wah kya performance hai, bilkul world class


---

## Methodology

### Notebook 1: Baseline Machine Learning Models

**Goal:** Establish baseline performance.

Steps:

- Data Preparation  
- Text Preprocessing  
- Train-Test Split (80:20)  
- Feature Extraction using TF-IDF  
- Model Training:
  - Logistic Regression  
  - SVM  
  - Random Forest  
- Model Evaluation  

---

### Notebook 2: SMOTE-Based Machine Learning Models

**Goal:** Handle class imbalance and improve detection.

Steps:

- Data Preparation  
- Text Preprocessing  
- TF-IDF Feature Extraction  
- Applied SMOTE balancing  
- Model Training:
  - Logistic Regression  
  - SVM  
  - Random Forest  
  - Naive Bayes  
- Model Evaluation  

---

### Notebook 3: Transformer-Based mBERT Model

**Goal:** Use contextual embeddings for best performance.

Steps:

- Data Preparation  
- Text Preprocessing  
- Label Encoding  
- Train-Test Split  
- Tokenization using mBERT tokenizer  
- Fine-tuning pre-trained mBERT  
- Model Evaluation  

---

## Results

| Model | Accuracy | F1 Score |
|------|----------|----------|
| Logistic Regression (Baseline) | 93.29% | 0.67 |
| Support Vector Machine (SMOTE) | 94.6% | 0.75 |
| mBERT Transformer | **96.85%** | **84.79%** |

Conclusion:

- Baseline models failed due to imbalance  
- SMOTE improved sarcasm detection  
- mBERT achieved best performance  

---

## Tech Stack

**Language:**

- Python  

**Libraries:**

- Scikit-learn  
- Pandas  
- NumPy  
- Matplotlib  
- imbalanced-learn  
- HuggingFace Transformers  
- PyTorch  

---

## Project Structure

## Project Structure

```
Sarcasm-Detection-Hinglish/
│
├── dataset/
│   └── hinglish_sarcasm_dataset.csv
│
├── notebooks/
│   ├── Sarcasm Detection Baseline.ipynb
│   ├── Sarcasm detection using SMOTE.ipynb
│   └── Sarcasm detection using mBERT.ipynb
│
├── requirements.txt
│
└── README.md
```

---

## Installation

Clone the repository:
git clone https://github.com/CHARVI1809/Sarcasm-Detection-in-Code-Mixed-Hinglish-Tweets

Install dependencies:
pip install -r requirements.txt

Run:


---

## Applications

- Sentiment Analysis  
- Social Media Monitoring  
- Opinion Mining  
- Chatbots  

---

## Authors

**Charvi Gupta**  
B.Tech Computer Science and Engineering  
Manipal University Jaipur  

**Ananya Srivastava**  
B.Tech Computer Science and Engineering  
Manipal University Jaipur  
  
## Acknowledgement

Thanks to the open-source NLP community and HuggingFace for providing transformer models.

---







AUC

This notebook explores deep learning and contextual embeddings for sarcasm detection.

📊 Results Overview

Random Forest achieved highest accuracy among ML models

Logistic Regression showed balanced performance

SMOTE improved minority class detection

mBERT captures contextual sarcasm better than traditional TF-IDF models

🛠 Technologies Used

Python

Pandas

NLTK

Scikit-learn

Imbalanced-learn (SMOTE)

HuggingFace Transformers

PyTorch / TensorFlow

Matplotlib & Seaborn




