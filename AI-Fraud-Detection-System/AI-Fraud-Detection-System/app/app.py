from flask import Flask, render_template, request, jsonify
import os
import random
from datetime import datetime
from utils import predict_transaction, load_metrics

# -------------------------------------------------
# Flask Setup
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

app.config['SECRET_KEY'] = "fraud-detection-secret"

# -------------------------------------------------
# In-memory transaction storage
# -------------------------------------------------

transaction_log = []
MAX_LOG = 100   # Prevent memory overflow


# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------

@app.route("/")
def index():

    metrics = load_metrics()

    fraud_txns = [t for t in transaction_log if t["label"] == "Fraud"]
    normal_txns = [t for t in transaction_log if t["label"] == "Normal"]

    stats = {
        "total": len(transaction_log),
        "fraud_count": len(fraud_txns),
        "normal_count": len(normal_txns),
        "accuracy": metrics.get("accuracy", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "f1_score": metrics.get("f1_score", 0),
        "roc_auc": metrics.get("roc_auc", 0),
    }

    return render_template(
        "index.html",
        stats=stats,
        transactions=transaction_log[::-1][:20]
    )


# -------------------------------------------------
# PREDICT TRANSACTION
# -------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data:
            return jsonify({"success": False, "error": "No input data"}), 400

        # Build transaction dictionary
        raw = {
            "Time": float(data.get("Time", data.get("time", 0))),
            "Amount": float(data.get("Amount", data.get("amount", 0))),
        }

        # PCA Features
        for i in range(1, 29):
            key = f"V{i}"
            raw[key] = float(data.get(key, 0))

        # Predict
        result = predict_transaction(raw)

        # Create transaction record
        txn = {
            "id": f"TXN{random.randint(100000,999999)}",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "amount": raw["Amount"],
            "label": result["label"],
            "confidence": round(result["confidence"], 2),
            "risk": result["risk_level"],
        }

        transaction_log.append(txn)

        # Limit log size
        if len(transaction_log) > MAX_LOG:
            transaction_log.pop(0)

        return jsonify({
            "success": True,
            "prediction": result,
            "transaction": txn
        })

    except FileNotFoundError:
        return jsonify({
            "success": False,
            "error": "Model not found. Train the model first."
        }), 500

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# -------------------------------------------------
# METRICS API
# -------------------------------------------------

@app.route("/api/metrics")
def api_metrics():
    return jsonify(load_metrics())


# -------------------------------------------------
# TRANSACTION HISTORY API
# -------------------------------------------------

@app.route("/api/transactions")
def api_transactions():
    return jsonify(transaction_log[::-1][:30])


# -------------------------------------------------
# STATS API
# -------------------------------------------------

@app.route("/api/stats")
def api_stats():

    fraud = [t for t in transaction_log if t["label"] == "Fraud"]
    normal = [t for t in transaction_log if t["label"] == "Normal"]

    total = len(transaction_log)

    return jsonify({
        "total": total,
        "fraud_count": len(fraud),
        "normal_count": len(normal),
        "fraud_rate": round(len(fraud) / max(total, 1) * 100, 2)
    })


# -------------------------------------------------
# START SERVER
# -------------------------------------------------

if __name__ == "__main__":

    print("\n🚀 Starting Fraud Detection Server...")
    print("Open browser → http://127.0.0.1:5000\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )