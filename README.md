# AI-Powered Credit Card Fraud Detection System

## Overview

This project is a Machine Learning-based Credit Card Fraud Detection System designed to identify fraudulent financial transactions in real time.

The system uses multiple machine learning approaches, including:

- Logistic Regression
- Random Forest Classifier
- Multi-Layer Perceptron (MLP Neural Network)
- K-Means Clustering (Unsupervised Learning)

After comparative evaluation, the Random Forest model was selected as the final deployed model due to its superior fraud detection performance.

---

## Features

### Single Transaction Prediction
- Manual transaction input
- Fraud probability estimation
- Risk level assessment

### Bulk CSV Analysis
- Upload CSV files containing multiple transactions
- Batch fraud prediction
- Download prediction results

### PDF Transaction Analysis
- Upload transaction reports in PDF format
- Automatic data extraction
- Fraud detection on extracted records

### Analytics Dashboard
- Fraud distribution visualization
- Interactive charts
- Model insights
- Feature importance analysis

### Model Performance Evaluation
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

---

## Dataset

Credit Card Fraud Detection Dataset

Source:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains anonymized PCA-transformed features:

- V1 to V28
- Amount
- Time

Target Variable:

- 0 = Legitimate Transaction
- 1 = Fraudulent Transaction

---

## Technologies Used

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Plotly
- Joblib
- PDFPlumber
- ReportLab

---

## Machine Learning Workflow

1. Data Preprocessing
2. Feature Scaling
3. Train-Test Split
4. SMOTE for Class Balancing
5. Model Training
6. Model Evaluation
7. Dashboard Deployment

---

## Trained Model

Final deployed model:

Random Forest Classifier

Saved Files:

- fraud_model.pkl
- amount_scaler.pkl
- time_scaler.pkl

---

## Installation

```bash
pip install -r requirements.txt
```

Run locally:

```bash
streamlit run app.py
```

---

## Author

Amjad Khan

BS Computer Science

Machine Learning Fraud Detection Project
