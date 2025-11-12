from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import joblib, os, traceback

# ==========================================================
# CONFIGURATION
# ==========================================================
SEQ_LEN = int(os.environ.get("SEQ_LEN", 400))  # GRU model input length
MODEL_PATH = os.environ.get("MODEL_PATH", "models/parkinson_final_gru_model.keras")
SCALER_PATH = os.environ.get("SCALER_PATH", "models/scaler.pkl")

# ==========================================================
# APP INITIALIZATION
# ==========================================================
app = Flask(__name__)
CORS(app, origins=[
    'http://localhost:3000',
    'http://localhost:5173',
    'http://localhost:8080',
    'http://localhost:4173'
], supports_credentials=True)

print("🔹 Loading model:", MODEL_PATH)
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

scaler = None
if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
        print("✅ Scaler loaded:", SCALER_PATH)
    except Exception as e:
        print("❌ Scaler load failed:", e)
else:
    print(" Scaler file not found:", SCALER_PATH)

# ==========================================================
# UTILITY: SAFE FLOAT PARSER
# ==========================================================
def safe_float(value):
    """Safely convert to float, replace invalid or None values with 0.0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

# ==========================================================
# PREPROCESS FUNCTION
# ==========================================================
def preprocess_input(keystrokes, seq_len=SEQ_LEN):
    """
    Extract only holdTime, flightTime, latencyTime features for prediction.
    Returns np.array shaped (1, seq_len, 3)
    """
    try:
        arr = np.array([
            [
                safe_float(k.get('holdTime')),
                safe_float(k.get('flightTime')),
                safe_float(k.get('latencyTime'))
            ]
            for k in keystrokes if isinstance(k, dict)
        ], dtype=np.float32)

        print(f"📥 Received {len(arr)} keystrokes for prediction.")
        if len(arr) == 0:
            return None

        # Apply scaling if scaler exists
        if scaler is not None:
            try:
                arr = scaler.transform(arr)
                print(" Applied feature scaling.")
            except Exception as e:
                print(" Scaler transform failed:", e)

        # Pad or truncate to seq_len
        n, f = arr.shape
        if n < seq_len:
            pad = np.tile(arr[-1], (seq_len - n, 1))
            arr = np.vstack([arr, pad])
        elif n > seq_len:
            arr = arr[:seq_len]

        print(f"✅ Final shape for model: {arr.shape}")
        return arr.reshape(1, seq_len, f)

    except Exception as e:
        print(" Preprocessing error:", e)
        traceback.print_exc()
        return None

# ==========================================================
# ROUTES
# ==========================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        ks = payload.get("keystrokes", None)

        if not ks or not isinstance(ks, list):
            return jsonify({"error": "keystrokes must be a non-empty list"}), 400

        # ✅ Only preprocess for prediction — no saving
        x = preprocess_input(ks)
        if x is None:
            return jsonify({"error": "preprocessing failed"}), 400

        # ✅ Run model prediction
        print("⚙️ Running model prediction...")
        pred = model.predict(x, verbose=0)
        prob = float(pred[0][0])
        prob = max(0.0, min(1.0, prob))  # Ensure [0,1]

        # ✅ Determine label & severity
        label = "Parkinson’s Detected" if prob >= 0.5 else "Healthy"
        if prob >= 0.85:
            severity = "Severe"
        elif prob >= 0.65:
            severity = "Moderate"
        elif prob >= 0.5:
            severity = "Mild"
        else:
            severity = "None"

        response = {
            "label": label,
            "probability": round(prob, 4),
            "severity": severity
        }

        # ✅ Debug Output
        print("📊 Prediction Result")
        print(f" Label: {response['label']}")
        print(f" Probability: {response['probability'] * 100:.2f}%")
        print(f" Severity: {response['severity']}")
        print("============================\n")

        return jsonify(response), 200

    except Exception as e:
        print(" Prediction error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ==========================================================
# MAIN ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
