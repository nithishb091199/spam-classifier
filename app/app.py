"""
app/app.py

Serves:
- GET  /            — demo UI (serves app/static/index.html)
- GET  /health      — health check
- POST /predict     — prediction API

Instructions:
- Make sure your venv is active:
    . .venv\\Scripts\\Activate.ps1
- Install CORS helper (only once):
    pip install flask-cors
- Run:
    python app/app.py
- Visit: http://127.0.0.1:5000/
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from joblib import load
from flask_cors import CORS

# Paths (relative to project root)
MODEL_PATH = os.path.join("data", "processed", "spam_model.joblib")
# Static files live in app/static/
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Create Flask app and enable CORS (helps if you open index.html from file:// or another origin)
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
CORS(app)

# Load model once at startup
try:
    model = load(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"Failed to load model: {e}")


@app.route("/", methods=["GET"])
def index():
    """
    Serve the demo index.html when visiting root.
    Falls back to a simple JSON message if the file is missing.
    """
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        # send_from_directory prevents Path issues
        return send_from_directory(STATIC_DIR, "index.html")
    return jsonify({"message": "Demo UI not found. Place index.html in app/static/."}), 200


@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON:
        { "text": "your message here" }
    Returns JSON:
        { "prediction": "spam"/"ham", "label": 1/0, "confidence": float }
    """
    if model is None:
        return jsonify({"error": "Model not loaded on server."}), 500

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request must be JSON with a 'text' field."}), 400

    text = data.get("text", "")
    if not isinstance(text, str) or text.strip() == "":
        return jsonify({"error": "Field 'text' must be a non-empty string."}), 400

    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text])[0]
            max_idx = int(probs.argmax())
            label_num = int(model.classes_[max_idx])
            confidence = float(probs[max_idx])
        else:
            label_num = int(model.predict([text])[0])
            confidence = None

        label_str = "spam" if label_num == 1 else "ham"

        response = {
            "prediction": label_str,
            "label": label_num,
            "confidence": confidence,
        }
        return jsonify(response), 200

    except Exception as ex:
        return jsonify({"error": f"Prediction failed: {ex}"}), 500


if __name__ == "__main__":
    # Helpful dev options; remember this is the development server only.
    app.run(host="127.0.0.1", port=5000, debug=True)
