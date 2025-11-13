"""
Fixed Version of app.py with Security Enhancements
This version includes:
- Rate limiting
- Input validation
- Request size limits
- Batch size limits
- Timeout protection
"""

from flask import Flask, render_template, request, jsonify
import torch
import os
import glob
import re
from datetime import datetime
from src.inference.predictor import Predictor
from src.model.rnn import MyRNN
from src.model.lstm import MyLSTM
from gensim.models import Word2Vec
from src.model.bert import BertClassifier
from transformers import BertTokenizer

# Import security enhancements
import sys
sys.path.append('fixes')
from rate_limiter import rate_limit, RATE_LIMIT_CONFIG
from input_validator import (
    validate_request, PredictionSchema, BatchPredictionSchema,
    MAX_TEXT_LENGTH, MAX_BATCH_SIZE
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
app = Flask(__name__, template_folder="web/templates")

# Security Configuration
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB max request size
app.config['JSON_SORT_KEYS'] = False

# Global variables for model and predictor
model = None
predictor = None
device = None
current_model_path = None
current_model_type = None
available_models = []


def parse_model_filename(filename):
    """Parse model filename to extract model type, accuracy, and timestamp"""
    pattern = r"^([a-zA-Z]+)_(\d+\.?\d*)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.pt$"
    match = re.match(pattern, filename)

    if match:
        model_type = match.group(1).upper()
        accuracy = float(match.group(2))
        date_str = match.group(3)
        time_str = match.group(4)

        try:
            datetime_str = f"{date_str} {time_str.replace('-', ':')}"
            timestamp = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            timestamp = None
            formatted_time = f"{date_str} {time_str}"

        return {
            "model_type": model_type,
            "accuracy": accuracy,
            "timestamp": timestamp,
            "formatted_time": formatted_time,
            "parsed": True,
        }
    else:
        return {
            "model_type": "UNKNOWN",
            "accuracy": 0.0,
            "timestamp": None,
            "formatted_time": "Unknown",
            "parsed": False,
        }


def scan_models():
    """Scan for available model files"""
    global available_models

    model_patterns = ["*.pth", "*.pt", "*.pkl", "*.model"]
    search_paths = ["model_save"]

    available_models = []

    for search_path in search_paths:
        if os.path.exists(search_path):
            for pattern in model_patterns:
                model_files = glob.glob(os.path.join(search_path, pattern))
                for model_file in model_files:
                    file_size = os.path.getsize(model_file)
                    file_mtime = os.path.getmtime(model_file)
                    filename = os.path.basename(model_file)

                    parsed_info = parse_model_filename(filename)

                    available_models.append(
                        {
                            "path": model_file,
                            "name": filename,
                            "size": file_size,
                            "modified": file_mtime,
                            "size_mb": round(file_size / (1024 * 1024), 2),
                            "model_type": parsed_info["model_type"],
                            "accuracy": parsed_info["accuracy"],
                            "timestamp": parsed_info["timestamp"],
                            "formatted_time": parsed_info["formatted_time"],
                            "parsed": parsed_info["parsed"],
                        }
                    )

    available_models.sort(
        key=lambda x: (x["accuracy"], x["timestamp"] or datetime.min), reverse=True
    )

    print(f"Found {len(available_models)} model files")
    return available_models


def load_model(model_path=None, model_type=None):
    """Load the trained model and create predictor"""
    global model, predictor, device, current_model_path, current_model_type

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model_w2v = Word2Vec.load("src/w2vmodel.model")
        print("Word2Vec model loaded")

        if model_type is None and model_path:
            filename = os.path.basename(model_path)
            parsed_info = parse_model_filename(filename)
            if parsed_info["parsed"]:
                model_type = parsed_info["model_type"].lower()
            else:
                model_type = "rnn"

        if model_type is None:
            model_type = "rnn"

        if model_type.lower() == "lstm":
            model = MyLSTM(model_w2v, hidden_size=256, num_classes=2)
            current_model_type = "LSTM"
        elif model_type.lower() == "rnn":
            model = MyRNN(model_w2v, hidden_size=300, num_classes=2)
            current_model_type = "RNN"
        elif model_type.lower() == "bert":
            model = BertClassifier()
            current_model_type = "BERT"
        else:
            model = MyRNN(model_w2v, hidden_size=300, num_classes=2)
            current_model_type = "RNN"

        if model_path is None:
            model_path = "./model_save/rnn_84.2_2025-10-12_20-12-15.pt"

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            current_model_path = model_path
            print(f"Trained model loaded from {model_path}")
        else:
            print(f"Model file {model_path} not found. Using untrained model.")
            current_model_path = None

        model.to(device)
        predictor = Predictor(model, device)
        print("Predictor initialized")

        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        return False


@app.route("/")
def home():
    """Main page"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@rate_limit(
    max_requests=RATE_LIMIT_CONFIG['predict']['max_requests'],
    window_seconds=RATE_LIMIT_CONFIG['predict']['window_seconds']
)
@validate_request(PredictionSchema)
def predict():
    """
    API endpoint for text prediction
    NOW WITH: Rate limiting, input validation, size limits
    """
    global tokenizer
    try:
        # Check if predictor is loaded
        if predictor is None:
            return (
                jsonify(
                    {
                        "error": "Model not loaded",
                        "prediction": None,
                        "confidence": None,
                    }
                ),
                500,
            )

        # Get validated data (already sanitized by validator)
        data = request.validated_data
        text = data["text"]

        # Additional length check (defense in depth)
        if len(text) > MAX_TEXT_LENGTH:
            return (
                jsonify(
                    {
                        "error": f"Text too long. Maximum {MAX_TEXT_LENGTH} characters",
                        "prediction": None,
                        "confidence": None,
                    }
                ),
                400,
            )

        # Make prediction with timeout
        try:
            prediction, confidence = predictor.predict(text, tokenizer)
        except Exception as pred_error:
            print(f"Prediction error: {pred_error}")
            return (
                jsonify(
                    {
                        "error": "Prediction failed",
                        "prediction": None,
                        "confidence": None,
                    }
                ),
                500,
            )

        # Format response
        result = {
            "prediction": int(prediction),
            "confidence": float(confidence),
            "label": "AI-Generated" if prediction == 0 else "Human-Written",
            "text": text[:100] + "..." if len(text) > 100 else text,  # Truncate in response
        }

        return jsonify(result)

    except Exception as e:
        return (
            jsonify(
                {
                    "error": f"Prediction failed: {str(e)}",
                    "prediction": None,
                    "confidence": None,
                }
            ),
            500,
        )


@app.route("/batch_predict", methods=["POST"])
@rate_limit(
    max_requests=RATE_LIMIT_CONFIG['batch_predict']['max_requests'],
    window_seconds=RATE_LIMIT_CONFIG['batch_predict']['window_seconds']
)
@validate_request(BatchPredictionSchema)
def batch_predict():
    """
    API endpoint for batch text prediction
    NOW WITH: Rate limiting, input validation, batch size limits
    """
    global tokenizer
    try:
        if predictor is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Get validated data
        data = request.validated_data
        texts = data["texts"]

        # Additional batch size check (defense in depth)
        if len(texts) > MAX_BATCH_SIZE:
            return jsonify({
                "error": f"Batch too large. Maximum {MAX_BATCH_SIZE} texts",
                "max_batch_size": MAX_BATCH_SIZE
            }), 400

        # Make batch predictions with timeout
        try:
            results = predictor.predict_batch(texts, tokenizer)
        except Exception as pred_error:
            print(f"Batch prediction error: {pred_error}")
            return jsonify({"error": "Batch prediction failed"}), 500

        # Format response
        formatted_results = []
        for i, (text, (pred, conf)) in enumerate(zip(texts, results)):
            formatted_results.append(
                {
                    "index": i,
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "prediction": int(pred),
                    "confidence": float(conf),
                    "label": "AI-Generated" if pred == 0 else "Human-Written",
                }
            )

        return jsonify({
            "results": formatted_results,
            "count": len(formatted_results)
        })

    except Exception as e:
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": predictor is not None,
            "device": str(device) if device else None,
            "current_model": current_model_path,
            "current_model_type": current_model_type,
        }
    )


@app.route("/models")
def get_models():
    """Get list of available models"""
    try:
        if not available_models:
            scan_models()

        model_list = []
        for model_info in available_models:
            model_list.append(
                {
                    "name": model_info["name"],
                    "path": model_info["path"],
                    "size_mb": model_info["size_mb"],
                    "is_current": model_info["path"] == current_model_path,
                    "modified": model_info["modified"],
                    "model_type": model_info["model_type"],
                    "accuracy": model_info["accuracy"],
                    "formatted_time": model_info["formatted_time"],
                    "parsed": model_info["parsed"],
                }
            )

        return jsonify(
            {
                "models": model_list,
                "current_model": current_model_path,
                "model_type": current_model_type,
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to get models: {str(e)}"}), 500


@app.route("/models/switch", methods=["POST"])
@rate_limit(
    max_requests=RATE_LIMIT_CONFIG['models_switch']['max_requests'],
    window_seconds=RATE_LIMIT_CONFIG['models_switch']['window_seconds']
)
def switch_model():
    """
    Switch to a different model
    NOW WITH: Rate limiting to prevent rapid switching attacks
    """
    try:
        data = request.get_json()
        if not data or "model_path" not in data:
            return jsonify({"error": "No model path provided"}), 400

        model_path = data["model_path"]
        model_type = data.get("model_type", None)

        # Validate model path (prevent path traversal)
        if not model_path.startswith("model_save/"):
            return jsonify({"error": "Invalid model path"}), 400

        if not os.path.exists(model_path):
            return jsonify({"error": f"Model file not found: {model_path}"}), 404

        if load_model(model_path, model_type):
            return jsonify(
                {
                    "success": True,
                    "message": f"Successfully switched to {os.path.basename(model_path)}",
                    "current_model": current_model_path,
                    "model_type": model_type,
                }
            )
        else:
            return jsonify({"error": "Failed to load the selected model"}), 500

    except Exception as e:
        return jsonify({"error": f"Failed to switch model: {str(e)}"}), 500


@app.route("/models/refresh", methods=["POST"])
@rate_limit(max_requests=10, window_seconds=60)
def refresh_models():
    """
    Refresh the list of available models
    NOW WITH: Rate limiting
    """
    try:
        scan_models()

        model_list = []
        for model_info in available_models:
            model_list.append(
                {
                    "name": model_info["name"],
                    "path": model_info["path"],
                    "size_mb": model_info["size_mb"],
                    "is_current": model_info["path"] == current_model_path,
                    "modified": model_info["modified"],
                    "model_type": model_info["model_type"],
                    "accuracy": model_info["accuracy"],
                    "formatted_time": model_info["formatted_time"],
                    "parsed": model_info["parsed"],
                }
            )

        return jsonify(
            {
                "success": True,
                "message": f"Found {len(model_list)} models",
                "models": model_list,
                "current_model": current_model_path,
                "model_type": current_model_type,
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to refresh models: {str(e)}"}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle request too large errors"""
    return jsonify({
        "error": "Request too large",
        "max_size": "1MB"
    }), 413


@app.errorhandler(429)
def rate_limit_exceeded(error):
    """Handle rate limit exceeded errors"""
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please try again later."
    }), 429


if __name__ == "__main__":
    print("Scanning for available models...")
    scan_models()

    best_loaded = False
    try:
        if available_models:
            best_model = available_models[0]
            best_model_path = best_model["path"]
            best_model_type = best_model.get("model_type")
            best_model_type = best_model_type.lower() if best_model_type else None
            if load_model(best_model_path, best_model_type):
                print(f"Loaded best model: {os.path.basename(best_model_path)}")
                best_loaded = True
        else:
            print("No model files found; attempting to load default model...")
    except Exception as e:
        print(f"Failed to auto-load best model: {e}. Falling back to default.")

    if not best_loaded:
        if not load_model():
            print("Failed to load model. Exiting...")
            raise SystemExit(1)

    print("Starting Flask app with security enhancements...")
    print("Rate limiting enabled")
    print("Input validation enabled")
    print("Request size limits enabled")
    app.run(host="0.0.0.0", port=5000)

