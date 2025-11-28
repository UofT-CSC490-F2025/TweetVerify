from flask import Flask, render_template, request, jsonify
import torch
import os
import glob
import re
from datetime import datetime
from gensim.models import Word2Vec
import threading
from contextlib import contextmanager

from src.inference.predictor import Predictor
from src.model.rnn import MyRNN
from src.model.lstm import MyLSTM
from src.model.bert import BertClassifier
from src.model.deberta import DebertaV3
from src.model.roberta import MyRobertaForBinaryClassification
from transformers import BertTokenizer, AutoConfig, AutoTokenizer

# Security-related imports
from src.security import (
    rate_limit,
    RATE_LIMIT_CONFIG,
    validate_request,
    PredictionSchema,
    BatchPredictionSchema,
    MAX_TEXT_LENGTH,
    MAX_BATCH_SIZE,
)

# ---------------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------------

app = Flask(
    __name__,
    template_folder="/home/richard8/projects/aip-agoldenb/richard8/TweetVerify/src/web/templates",
)

# Security configuration: limit request size and JSON behavior
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1MB max request size
app.config["JSON_SORT_KEYS"] = False

# Global Word2Vec model used for RNN/LSTM classifiers
model_w2v = Word2Vec.load("datasets/w2vmodel.model")

# Global state related to models (but NOT directly shared between threads)
available_models = []  # List of discovered model files with metadata
device = None          # torch.device used for all models


# ---------------------------------------------------------------------------
# Thread-safe Model Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Thread-safe registry for loaded models.

    Responsibilities:
    - Store all loaded models and their predictors.
    - Track which model is the current active model.
    - Provide atomic "snapshot" reads for use by request handlers.
    - Provide atomic "switch model" operation.
    """

    def __init__(self):
        # {model_path: {"model": model, "predictor": Predictor, "model_type": str}}
        self._models = {}
        self._current_path = None
        self._lock = threading.RLock()
        self._version = 0  # Optional version counter for debugging / change detection

    def register_model(self, model_path: str, model, predictor, model_type_str: str):
        """
        Register or update a loaded model in a thread-safe way.
        If this is the first model, it also becomes the current model.
        """
        with self._lock:
            self._models[model_path] = {
                "model": model,
                "predictor": predictor,
                "model_type": model_type_str,
            }
            # If there was no current model, set this one as current
            if self._current_path is None:
                self._current_path = model_path
            self._version += 1

    def is_model_loaded(self, model_path: str) -> bool:
        """Check if the given model path is already loaded."""
        with self._lock:
            return model_path in self._models

    def switch_model(self, model_path: str):
        """
        Atomically switch the current model to `model_path`.

        Raises:
            ValueError: if the model path is not in the registry.
        """
        with self._lock:
            if model_path not in self._models:
                raise ValueError(f"Model {model_path} not loaded")
            old_path = self._current_path
            self._current_path = model_path
            self._version += 1
            return old_path

    @contextmanager
    def get_model_context(self):
        """
        Context manager to obtain the current predictor safely.

        Usage:
            with model_registry.get_model_context() as predictor:
                if predictor is None:
                    # no model loaded
                else:
                    # safe to call predictor.predict(...)

        Implementation detail:
        - The lock is held only while reading the pointer to the predictor.
        - The actual prediction happens outside the lock, since models are
          effectively read-only after loading.
        """
        with self._lock:
            path = self._current_path
            predictor = None
            if path and path in self._models:
                predictor = self._models[path]["predictor"]
        # Lock is released here; predictor is a stable reference
        yield predictor

    def snapshot(self):
        """
        Take a consistent snapshot of the current registry state.

        Returns:
            dict with keys:
                - "current_path": path of current model (or None)
                - "current_type": model type string (or None)
                - "loaded_models": shallow copy of internal models dict
        """
        with self._lock:
            current_path = self._current_path
            current_type = None
            if current_path and current_path in self._models:
                current_type = self._models[current_path]["model_type"]
            loaded_copy = dict(self._models)
            return {
                "current_path": current_path,
                "current_type": current_type,
                "loaded_models": loaded_copy,
            }


# Single global instance of the registry
model_registry = ModelRegistry()


# ---------------------------------------------------------------------------
# Model file parsing and scanning
# ---------------------------------------------------------------------------

def parse_model_filename(filename: str):
    """
    Parse a model filename to extract model type, accuracy, and timestamp.

    Expected pattern:
        {model_type}_{accuracy}_{date}_{time}.pt
    Example:
        lstm_92.8_2025-10-12_18-23-37.pt
    """
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
        # Fallback for files that don't match the pattern
        return {
            "model_type": "UNKNOWN",
            "accuracy": 0.0,
            "timestamp": None,
            "formatted_time": "Unknown",
            "parsed": False,
        }


def scan_models():
    """
    Scan the filesystem for available model files and populate `available_models`.

    Returns:
        list of model metadata dictionaries.
    """
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

    # Sort by accuracy desc, then timestamp desc (newest first)
    available_models.sort(
        key=lambda x: (x["accuracy"], x["timestamp"] or datetime.min),
        reverse=True,
    )

    print(f"Found {len(available_models)} model files:")
    for model_info in available_models:
        if model_info["parsed"]:
            print(
                f"  - {model_info['name']} ({model_info['size_mb']} MB) "
                f"- {model_info['model_type']} {model_info['accuracy']:.1f}% "
                f"({model_info['formatted_time']})"
            )
        else:
            print(
                f"  - {model_info['name']} ({model_info['size_mb']} MB) - Unknown format"
            )

    return available_models


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_single_model(model_path: str, model_type: str = None) -> bool:
    """
    Load a single model from disk (if not already loaded) and register it in ModelRegistry.

    Args:
        model_path: Path to the model file.
        model_type: Optional explicit type (e.g., "rnn", "lstm", "bert", ...).
                    If None, it will be inferred from the filename.

    Returns:
        True if loading succeeded or the model was already loaded; False otherwise.
    """
    global device, model_w2v

    # Skip if already loaded
    if model_registry.is_model_loaded(model_path):
        print(f"⏭️  Model {os.path.basename(model_path)} already loaded, skipping...")
        return True

    try:
        # Initialize device lazily
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

        # Determine model type if not provided
        if model_type is None:
            filename = os.path.basename(model_path)
            parsed_info = parse_model_filename(filename)
            if parsed_info["parsed"]:
                model_type = parsed_info["model_type"].lower()
            else:
                model_type = "rnn"  # default fallback

        tokenizer = None

        # Create model instance based on type
        if model_type.lower() == "lstm":
            model = MyLSTM(model_w2v, hidden_size=256, num_classes=2)
            model_type_str = "LSTM"
            print(f"✅ Created LSTM model for {os.path.basename(model_path)}")

        elif model_type.lower() == "rnn":
            model = MyRNN(model_w2v, hidden_size=300, num_classes=2)
            model_type_str = "RNN"
            print(f"✅ Created RNN model for {os.path.basename(model_path)}")

        elif model_type.lower() == "bert":
            model = BertClassifier()
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model_type_str = "BERT"
            print(f"✅ Created BERT model for {os.path.basename(model_path)}")

        elif model_type.lower() == "deberta":
            model_name = "microsoft/deberta-v3-large"
            config = AutoConfig.from_pretrained(model_name)
            config.num_labels = 2
            model = DebertaV3.from_pretrained(model_name, config=config)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model_type_str = "DeBERTaV3"
            print(f"✅ Created DeBERTaV3 model for {os.path.basename(model_path)}")

        elif model_type.lower() == "roberta":
            model_name = "FacebookAI/roberta-large"
            model = MyRobertaForBinaryClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model_type_str = "RoBERTa"
            print(f"✅ Created RoBERTa model for {os.path.basename(model_path)}")

        else:
            # Default to RNN for unknown types
            model = MyRNN(model_w2v, hidden_size=300, num_classes=2)
            model_type_str = "RNN"
            print(
                f"✅ Created RNN model (default for type: {model_type}) "
                f"for {os.path.basename(model_path)}"
            )

        # Load trained weights if file exists
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Trained model loaded from {model_path}")
        else:
            print(f"⚠️  Model file {model_path} not found. Using untrained model.")

        # Move model to device and create predictor
        model.to(device)
        predictor = Predictor(model, device, tokenizer)
        print(f"✅ Predictor initialized for {os.path.basename(model_path)}")

        # Register in the thread-safe registry
        model_registry.register_model(
            model_path=model_path,
            model=model,
            predictor=predictor,
            model_type_str=model_type_str,
        )

        return True

    except Exception as e:
        print(f"❌ Error loading model {model_path}: {e}")
        return False


def load_all_models() -> bool:
    """
    Load all discovered models into memory and register them in the registry.

    Returns:
        True if at least one model was successfully loaded, False otherwise.
    """
    global available_models, device

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    if not available_models:
        scan_models()

    if not available_models:
        print("⚠️  No model files found.")
        return False

    print(f"\n🔄 Loading {len(available_models)} models into memory...")
    loaded_count = 0

    for model_info in available_models:
        model_path = model_info["path"]
        model_type = model_info.get("model_type")
        model_type = model_type.lower() if model_type else None

        if load_single_model(model_path, model_type):
            loaded_count += 1

    snapshot = model_registry.snapshot()
    if snapshot["current_path"]:
        print(
            f"🎯 Current model set to: "
            f"{os.path.basename(snapshot['current_path'])} ({snapshot['current_type']})"
        )

    print(f"\n✅ Successfully loaded {loaded_count}/{len(available_models)} models")
    return loaded_count > 0


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    """Main index page (serves HTML frontend)."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@rate_limit(
    max_requests=RATE_LIMIT_CONFIG["predict"]["max_requests"],
    window_seconds=RATE_LIMIT_CONFIG["predict"]["window_seconds"],
)
@validate_request(PredictionSchema)
def predict():
    """
    API endpoint for single-text prediction.

    Security:
      - Rate limited.
      - Input validated via PredictionSchema.
      - Request size restricted by Flask config.
    Concurrency:
      - Uses ModelRegistry.get_model_context() to get a stable predictor snapshot.
    """
    try:
        # Atomically get current predictor snapshot
        with model_registry.get_model_context() as predictor:
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

        # Request has already been validated and attached as request.validated_data
        data = request.validated_data
        text = data["text"]

        # Extra length check (defense in depth)
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

        # Perform prediction
        try:
            prediction, confidence = predictor.predict(text)
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

        result = {
            "prediction": int(prediction),
            "confidence": float(confidence),
            "label": "AI-Generated" if prediction == 0 else "Human-Written",
            "text": text[:100] + "..." if len(text) > 100 else text,
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


@app.route("/health")
def health():
    """
    Simple health-check endpoint.

    Returns:
        - Whether the service is up.
        - Whether a model is currently loaded.
        - Which device is used.
        - The current model and type.
        - Total number of loaded models.
    """
    snapshot = model_registry.snapshot()
    loaded_models = snapshot["loaded_models"]
    current_path = snapshot["current_path"]
    current_type = snapshot["current_type"]

    return jsonify(
        {
            "status": "healthy",
            "model_loaded": current_path is not None and current_path in loaded_models,
            "device": str(device) if device else None,
            "current_model": current_path,
            "current_model_type": current_type,
            "total_loaded_models": len(loaded_models),
        }
    )


@app.route("/models")
def get_models():
    """
    Get metadata for all available models.

    This uses:
      - `available_models` (filesystem scan result)
      - Snapshot from ModelRegistry for "is_current" and "is_loaded"
    """
    try:
        if not available_models:
            scan_models()

        snapshot = model_registry.snapshot()
        loaded_models = snapshot["loaded_models"]
        current_path = snapshot["current_path"]
        current_type = snapshot["current_type"]

        model_list = []
        for model_info in available_models:
            model_path = model_info["path"]
            model_list.append(
                {
                    "name": model_info["name"],
                    "path": model_path,
                    "size_mb": model_info["size_mb"],
                    "is_current": model_path == current_path,
                    "is_loaded": model_path in loaded_models,
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
                "current_model": current_path,
                "model_type": current_type,
                "total_loaded": len(loaded_models),
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to get models: {str(e)}"}), 500


@app.route("/models/switch", methods=["POST"])
@rate_limit(
    max_requests=RATE_LIMIT_CONFIG["models_switch"]["max_requests"],
    window_seconds=RATE_LIMIT_CONFIG["models_switch"]["window_seconds"],
)
def switch_model():
    """
    Switch the current model to a different one (must be in model_save path).

    Security:
      - Rate limited.
      - Validates that the path belongs under model_save/.
      - Uses ModelRegistry.switch_model() for atomic updates.
    """
    try:
        data = request.get_json()
        if not data or "model_path" not in data:
            return jsonify({"error": "No model path provided"}), 400

        model_path = data["model_path"]

        # Basic path validation to prevent path traversal / arbitrary file loading
        if not model_path.startswith("model_save/") and not model_path.startswith(
            "./model_save/"
        ):
            return jsonify({"error": "Invalid model path"}), 400

        if not os.path.exists(model_path):
            return jsonify({"error": f"Model file not found: {model_path}"}), 404

        # Load model if not already registered
        if not model_registry.is_model_loaded(model_path):
            model_type = data.get("model_type", None)
            if not load_single_model(model_path, model_type):
                return jsonify(
                    {"error": f"Model not loaded and failed to load: {model_path}"}
                ), 404

        # Atomically switch current model
        model_registry.switch_model(model_path)
        snapshot = model_registry.snapshot()

        return jsonify(
            {
                "success": True,
                "message": f"Successfully switched to {os.path.basename(model_path)}",
                "current_model": snapshot["current_path"],
                "model_type": snapshot["current_type"],
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to switch model: {str(e)}"}), 500


@app.route("/models/refresh", methods=["POST"])
@rate_limit(max_requests=10, window_seconds=60)
def refresh_models():
    """
    Refresh the list of available models on disk and load any new ones.

    - Re-scans the filesystem.
    - Loads any models that are not yet in the registry.
    """
    try:
        scan_models()  # Update available_models from disk

        # Load any newly discovered models
        for model_info in available_models:
            model_path = model_info["path"]
            if not model_registry.is_model_loaded(model_path):
                model_type = model_info.get("model_type")
                model_type = model_type.lower() if model_type else None
                load_single_model(model_path, model_type)

        snapshot = model_registry.snapshot()
        loaded_models = snapshot["loaded_models"]
        current_path = snapshot["current_path"]
        current_type = snapshot["current_type"]

        model_list = []
        for model_info in available_models:
            model_path = model_info["path"]
            model_list.append(
                {
                    "name": model_info["name"],
                    "path": model_path,
                    "size_mb": model_info["size_mb"],
                    "is_current": model_path == current_path,
                    "is_loaded": model_path in loaded_models,
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
                "message": f"Found {len(model_list)} models, {len(loaded_models)} loaded",
                "models": model_list,
                "current_model": current_path,
                "model_type": current_type,
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to refresh models: {str(e)}"}), 500


@app.route("/batch_predict", methods=["POST"])
@rate_limit(
    max_requests=RATE_LIMIT_CONFIG["batch_predict"]["max_requests"],
    window_seconds=RATE_LIMIT_CONFIG["batch_predict"]["window_seconds"],
)
@validate_request(BatchPredictionSchema)
def batch_predict():
    """
    API endpoint for batch prediction over multiple texts.

    Security:
      - Rate limited.
      - Input validated via BatchPredictionSchema.
      - Enforces MAX_BATCH_SIZE.
    Concurrency:
      - Uses ModelRegistry.get_model_context() for safe predictor retrieval.
    """
    try:
        with model_registry.get_model_context() as predictor:
            if predictor is None:
                return jsonify({"error": "Model not loaded"}), 500

        data = request.validated_data
        texts = data["texts"]

        if len(texts) > MAX_BATCH_SIZE:
            return (
                jsonify(
                    {
                        "error": f"Batch too large. Maximum {MAX_BATCH_SIZE} texts",
                        "max_batch_size": MAX_BATCH_SIZE,
                    }
                ),
                400,
            )

        # Perform batch prediction
        try:
            results = predictor.predict_batch(texts)
        except Exception as pred_error:
            print(f"Batch prediction error: {pred_error}")
            return jsonify({"error": "Batch prediction failed"}), 500

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

        return jsonify({"results": formatted_results, "count": len(formatted_results)})

    except Exception as e:
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle payload-too-large errors."""
    return (
        jsonify({"error": "Request too large", "max_size": "1MB"}),
        413,
    )


@app.errorhandler(429)
def rate_limit_exceeded(error):
    """Handle rate-limit exceeded errors."""
    return (
        jsonify(
            {
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please try again later.",
            }
        ),
        429,
    )


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Scan for models on startup
    print("🔍 Scanning for available models...")
    scan_models()

    # Try to load all discovered models
    if not load_all_models():
        print("⚠️  No models were loaded. Application may not function correctly.")

        # Try a default fallback model if available
        default_model_path = "./model_save/rnn_84.2_2025-10-12_20-12-15.pt"
        if os.path.exists(default_model_path):
            print(f"🔄 Attempting to load default model: {default_model_path}")
            if load_single_model(default_model_path, "rnn"):
                # Ensure the default model is the current one
                model_registry.switch_model(default_model_path)
                print("✅ Loaded default model as fallback")
            else:
                print("❌ Failed to load default model. Exiting...")
                raise SystemExit(1)
        else:
            print("❌ No models found and no default model available. Exiting...")
            raise SystemExit(1)

    snapshot = model_registry.snapshot()
    loaded_models_snapshot = snapshot["loaded_models"]
    current_path = snapshot["current_path"]

    print(f"🚀 Starting Flask app with {len(loaded_models_snapshot)} model(s) loaded...")
    current_name = os.path.basename(current_path) if current_path else "None"
    print(f"🎯 Current model: {current_name}")

    app.run(host="0.0.0.0", port=5000)
