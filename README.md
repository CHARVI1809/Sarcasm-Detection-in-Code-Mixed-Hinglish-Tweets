🧠 Sarcasm Detection in Code Mixed Hinglish Tweets
📌 Project Overview

This project focuses on building a Sarcasm Detection System using NLP techniques. The goal is to classify tweets as sarcastic (1) or non-sarcastic (0) using different approaches and compare their performance.

The project is divided into three notebooks:

Baseline ML (without SMOTE)

ML with SMOTE (balanced dataset)

Deep Learning using mBERT

📓 Notebook 1: Sarcasm Detection Baseline
Workflow:

Data Preparation

Loaded dataset

Checked missing values

Converted labels (YES/NO → 1/0)

Text Preprocessing

Lowercasing

Removing punctuation

Stopword removal

Lemmatization

Feature Extraction

TF-IDF Vectorization

Train–Test Split

Model Training

Logistic Regression

SVM

Random Forest

Model Evaluation

Accuracy

F1 Score

Confusion matrix

Performance comparison graph

This notebook establishes the baseline performance.

📓 Notebook 2: Sarcasm Detection with SMOTE (Balanced Dataset)
Workflow:

Data Preparation

Loaded dataset and checked structure

Verified missing values

Converted labels (YES/NO → 1/0)

Exploratory Data Analysis (EDA)

Tweet distribution analysis

Tweet length distribution

Common and top words extraction

Text Preprocessing

Lowercasing

Removing punctuation and special characters

Stopword removal

Lemmatization

Feature Extraction

TF-IDF Vectorization

Converted text into numerical format

Handling Imbalanced Data

Applied SMOTE / Oversampling

Model Training

Logistic Regression

Support Vector Machine (SVM)

Random Forest

Naive Bayes

Model Evaluation

Accuracy

Precision

Recall

F1 Score

AUC Score

Top 15 fetures

This notebook analyzes how balancing the dataset improves performance and fairness.

📓 Notebook 3: Sarcasm Detection using mBERT (Multilingual BERT Model)
Workflow:

Data Loading and Preparation

Minimal Preprocessing

No heavy cleaning (BERT handles context internally)

Tokenization using mBERT tokenizer

Fine-tuning mBERT model for binary classification

Evaluation

Accuracy

F1 Score

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

If you want, I can also create:


