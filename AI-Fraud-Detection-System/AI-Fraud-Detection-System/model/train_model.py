import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb

warnings.filterwarnings("ignore")

print("="*60)
print("      AI FRAUD DETECTION TRAINING")
print("="*60)

# -------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------
print("\n[1/6] Loading dataset...")

df = pd.read_csv("../dataset/creditcard.csv")

print(f"Rows: {len(df):,}")
print(f"Fraud cases: {df['Class'].sum()}")
print(f"Normal cases: {(df['Class']==0).sum()}")

# -------------------------------------------------
# 2. Feature Engineering
# -------------------------------------------------
print("\n[2/6] Feature Engineering...")

df["Log_Amount"] = np.log1p(df["Amount"])

df["Hour"] = (df["Time"] % 86400) / 3600
df["Hour_sin"] = np.sin(2*np.pi*df["Hour"]/24)
df["Hour_cos"] = np.cos(2*np.pi*df["Hour"]/24)

df["V1_V2"] = df["V1"] * df["V2"]
df["V3_V4"] = df["V3"] * df["V4"]
df["Amount_V1"] = df["Log_Amount"] * df["V1"]
df["Amount_V14"] = df["Log_Amount"] * df["V14"]

df.drop(columns=["Time","Amount"], inplace=True)

# Features
feature_cols = [c for c in df.columns if c != "Class"]

X = df[feature_cols]
y = df["Class"]

# -------------------------------------------------
# 3. Train Test Split
# -------------------------------------------------
print("\n[3/6] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# 4. Handle Imbalance (SMOTE)
# -------------------------------------------------
print("\n[4/6] Applying SMOTE...")

smote = SMOTE(sampling_strategy=0.3, random_state=42)

X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("Training samples after SMOTE:", X_train_sm.shape)

# -------------------------------------------------
# 5. Train Models
# -------------------------------------------------
print("\n[5/6] Training models...")

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=250,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=30,
    random_state=42,
    eval_metric="logloss"
)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight={0:1,1:20},
    random_state=42,
    n_jobs=-1
)

# Ensemble
ensemble = VotingClassifier(
    estimators=[
        ("xgb", xgb_model),
        ("rf", rf_model)
    ],
    voting="soft",
    weights=[2,1]
)

# Pipeline
pipeline = Pipeline([
    ("scaler", RobustScaler()),
    ("model", ensemble)
])

print("Training ensemble model...")

pipeline.fit(X_train_sm, y_train_sm)

# -------------------------------------------------
# 6. Evaluation
# -------------------------------------------------
print("\n[6/6] Evaluating model...")

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:,1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)

print("\n"+"="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(f"Accuracy  : {acc*100:.2f}%")
print(f"Precision : {prec*100:.2f}%")
print(f"Recall    : {rec*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")
print(f"ROC-AUC   : {auc*100:.2f}%")

print("\nConfusion Matrix")
print(cm)

# -------------------------------------------------
# Save Model
# -------------------------------------------------

metrics = {
    "accuracy": round(acc*100,2),
    "precision": round(prec*100,2),
    "recall": round(rec*100,2),
    "f1_score": round(f1*100,2),
    "roc_auc": round(auc*100,2),
    "confusion_matrix": cm.tolist(),
    "feature_names": feature_cols
}

with open("fraud_model.pkl","wb") as f:
    pickle.dump(pipeline,f)

with open("metrics.pkl","wb") as f:
    pickle.dump(metrics,f)

print("\n✅ Model saved successfully!")
print("Files created:")
print("   fraud_model.pkl")
print("   metrics.pkl")