# 🛡️ FinSecure AI — Intelligent Fraud Detection System

> **FinSecure AI** is a machine learning–powered platform designed to detect fraudulent financial transactions in real time.
> It combines **advanced fraud detection algorithms, data analytics, and a Flask-based web dashboard** to provide an intelligent security solution for digital payments.

---

# Project Structure

```
AI-FRAUD-DETECTION-SYSTEM/
│
├── app/
│   ├── app.py                ← Flask application & API routes
│   ├── utils.py              ← Prediction utilities
│   └── __pycache__/
│
├── dataset/
│   └── creditcard.csv        ← Credit card fraud dataset
│
├── diagrams/                 ← System architecture diagrams
│
├── model/
│   ├── fraud_model.pkl       ← Trained fraud detection model
│   ├── metrics.pkl           ← Model evaluation metrics
│   ├── train_model.py        ← Model training script
│   └── test_fraud.py         ← Model testing script
│
├── notebooks/
│   ├── fraud_detection_analysis.ipynb
│   ├── model_evaluation.ipynb
│   └── fraud_model.pkl
│
├── research_paper/           ← Research documentation
│
├── static/
│   ├── style.css             ← Frontend styling
│   └── script.js             ← UI interaction scripts
│
├── templates/
│   └── index.html            ← Web dashboard interface
│
├── requirements.txt
└── README.md
```

---

# Quick Start

## 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Download the Dataset

Download the **Credit Card Fraud Detection Dataset** from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the dataset in:

```
dataset/creditcard.csv
```

---

## 3️⃣ Train the Model

Run the training script:

```bash
python model/train_model.py
```

This will generate the trained model and metrics:

```
model/fraud_model.pkl
model/metrics.pkl
```

---

## 4️⃣ Run the Web Dashboard

```bash
python app/app.py
```

Open the browser:

```
http://localhost:5000
```

---

# Model Architecture

| Component          | Description                                           |
| ------------------ | ----------------------------------------------------- |
| Algorithm          | Ensemble Machine Learning                             |
| Models             | Random Forest, Gradient Boosting, Logistic Regression |
| Feature Scaling    | RobustScaler                                          |
| Dataset            | Credit Card Fraud Dataset (Kaggle)                    |
| Imbalance Handling | Oversampling / SMOTE                                  |
| Output             | Fraud probability prediction                          |

---

# Model Performance

| Metric    | Description                          |
| --------- | ------------------------------------ |
| Accuracy  | Overall prediction accuracy          |
| Precision | Correct fraud predictions            |
| Recall    | Ability to detect fraud              |
| F1 Score  | Balance between precision and recall |
| ROC-AUC   | Fraud detection capability           |

Expected model performance:

```
Accuracy  ≥ 92%
ROC-AUC   ≥ 97%
```

---

# Web Dashboard

The FinSecure AI dashboard provides an interactive interface to analyze and predict fraudulent transactions.

### Features

• Enter transaction details manually
• Detect fraud probability instantly
• Visualize fraud statistics
• Monitor transaction behavior

The dashboard uses a **dark modern UI with dynamic JavaScript interactions**.

---

# API Endpoints

| Endpoint            | Method | Description                   |
| ------------------- | ------ | ----------------------------- |
| `/`                 | GET    | Main dashboard                |
| `/predict`          | POST   | Predict fraud for transaction |
| `/demo_predict`     | POST   | Run demo fraud prediction     |
| `/api/metrics`      | GET    | Model evaluation metrics      |
| `/api/stats`        | GET    | Fraud detection statistics    |
| `/api/transactions` | GET    | Recent transactions data      |

---

# Notebooks

| Notebook                       | Purpose                         |
| ------------------------------ | ------------------------------- |
| fraud_detection_analysis.ipynb | Exploratory data analysis       |
| model_evaluation.ipynb         | Model comparison and evaluation |

These notebooks help analyze fraud patterns and evaluate model performance.

---

# Technologies Used

### Programming

Python

### Machine Learning

Scikit-learn
XGBoost
LightGBM
Imbalanced-learn

### Data Processing

Pandas
NumPy

### Visualization

Matplotlib
Seaborn

### Web Development

Flask
HTML
CSS
JavaScript

---

# Fraud Detection Pipeline

```
Transaction Data
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Handling Class Imbalance
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Fraud Prediction API
      ↓
FinSecure AI Dashboard
```

---

# Future Enhancements

• Real-time fraud detection API
• Explainable AI using SHAP
• Fraud monitoring dashboard with analytics
• Deep learning anomaly detection
• Cloud deployment (AWS / Docker)

---

# Author

**Adarsh Dubey**
MCA Student | AI & Data Science Enthusiast

---

# License

This project is released under the **MIT License**.

---
