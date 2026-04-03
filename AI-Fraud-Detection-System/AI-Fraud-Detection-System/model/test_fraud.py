import pickle
import pandas as pd
import numpy as np
import os

# Load model and metrics
model_path = 'fraud_model.pkl'
metrics_path = 'metrics.pkl'

if not os.path.exists(model_path):
    print("❌ Model not found! Please train the model first.")
    exit()

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(metrics_path, 'rb') as f:
    metrics = pickle.load(f)

# Feature engineering function (copied from utils.py – ensures no import issues)
def engineer_features(raw_dict):
    df = pd.DataFrame([raw_dict])
    df['Log_Amount'] = np.log1p(df['Amount'])
    df['Hour'] = (df['Time'] % 86400) / 3600
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['V1_V2'] = df['V1'] * df['V2']
    df['V3_V4'] = df['V3'] * df['V4']
    df['Amount_V1'] = df['Log_Amount'] * df['V1']
    df['Amount_V14'] = df['Log_Amount'] * df['V14']
    df.drop(columns=['Time', 'Amount'], inplace=True)
    
    feature_names = metrics.get('feature_names', [])
    if feature_names:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df[feature_names]
    return df

# Load a real fraud transaction from dataset (if dataset available)
dataset_path = os.path.join('..', 'dataset', 'creditcard.csv')
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
    fraud_row = df[df['Class'] == 1].iloc[0].to_dict()
    # Prepare raw input
    raw = {k: v for k, v in fraud_row.items() if k in ['Time', 'Amount'] or k.startswith('V')}
    # Fill missing V's with 0
    for i in range(1, 29):
        raw.setdefault(f'V{i}', 0.0)
else:
    # Manual fraud example (if dataset not found)
    print("Dataset not found, using manual fraud example...")
    raw = {
        'Time': 406.0,
        'Amount': 2.50,
        'V1': -2.345708, 'V2': 1.548354, 'V3': 0.482741, 'V4': -0.838408,
        'V5': -0.625984, 'V6': 0.248282, 'V7': -0.366628, 'V8': 0.300354,
        'V9': -0.209590, 'V10': -0.359144, 'V11': 0.124442, 'V12': 0.273213,
        'V13': -0.388343, 'V14': -0.242090, 'V15': 0.234654, 'V16': 0.343678,
        'V17': -0.492953, 'V18': -0.191802, 'V19': 0.184071, 'V20': -0.116438,
        'V21': -0.171463, 'V22': 0.332375, 'V23': -0.187419, 'V24': -0.083629,
        'V25': 0.128257, 'V26': -0.225342, 'V27': -0.019424, 'V28': 0.078754
    }

# Predict
X = engineer_features(raw)
prob = model.predict_proba(X)[0][1]
label = 'Fraud' if prob >= 0.3 else 'Normal'
risk = 'HIGH' if prob >= 0.6 else 'MEDIUM' if prob >= 0.3 else 'LOW'
confidence = prob * 100

print("\n" + "="*50)
print("   FRAUD DETECTION TEST")
print("="*50)
print(f"Transaction Amount: ₹{raw['Amount']}")
print(f"Fraud Probability  : {confidence:.2f}%")
print(f"Prediction         : {label}")
print(f"Risk Level         : {risk}")
print("="*50)

if label == 'Normal' and confidence < 5:
    print("\n⚠️ WARNING: Model predicted NORMAL even for fraud example.")
    print("   Your model may not be trained properly.")
    print("   Please retrain using the improved train_model.py I provided earlier.")