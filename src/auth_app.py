from flask import Flask, request, jsonify, session, render_template, redirect, url_for
import psycopg2
from psycopg2.extras import RealDictCursor
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import glob
import re
import uuid
import json
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from aws_training_manager import aws_training_manager

app = Flask(__name__, template_folder="web/templates")
app.secret_key = os.urandom(24)


UPLOAD_FOLDER = "model_save"
ALLOWED_EXTENSIONS = {"pt", "pth", "pkl", "model"}
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def get_db_connection():
    conn = psycopg2.connect(
        host="database-1-instance-1.cypeciyoo3i1.us-east-1.rds.amazonaws.com",
        database="postgres",
        user="postgres",
        password="20050101",
    )
    return conn


@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    hashed_password = generate_password_hash(password)

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cur.fetchone():
            return jsonify({"error": "Username already exists"}), 400

        cur.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            (username, hashed_password),
        )
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"message": "User registered successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return jsonify(
                {"message": "Login successful", "redirect": url_for("dashboard")}
            )
        else:
            return jsonify({"error": "Invalid username or password"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/status")
def status():
    if "user_id" in session:
        return jsonify({"logged_in": True, "username": session["username"]})
    else:
        return jsonify({"logged_in": False})


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("index"))
    return render_template("dashboard.html", username=session.get("username"))


@app.route("/models")
def models():
    if "user_id" not in session:
        return redirect(url_for("index"))

    available_models = scan_models()

    return render_template(
        "models.html", username=session.get("username"), models=available_models
    )


@app.route("/api/models")
def api_get_models():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        available_models = scan_models()

        model_list = []
        for model_info in available_models:
            model_list.append(
                {
                    "name": model_info["name"],
                    "path": model_info["path"],
                    "size_mb": model_info["size_mb"],
                    "modified": model_info["modified"],
                    "model_type": model_info["model_type"],
                    "accuracy": model_info["accuracy"],
                    "formatted_time": model_info["formatted_time"],
                    "parsed": model_info["parsed"],
                }
            )

        return jsonify(
            {"success": True, "models": model_list, "count": len(model_list)}
        )

    except Exception as e:
        return jsonify({"error": f"Failed to get models: {str(e)}"}), 500


@app.route("/api/models/delete", methods=["POST"])
def api_delete_model():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json()
        if not data or "model_path" not in data:
            return jsonify({"error": "No model path provided"}), 400

        model_path = data["model_path"]

        if not model_path.startswith("model_save/") and not model_path.startswith(
            "./model_save/"
        ):
            return jsonify({"error": "Invalid model path"}), 400

        if not os.path.exists(model_path):
            return jsonify({"error": "Model file not found"}), 404

        os.remove(model_path)

        return jsonify(
            {
                "success": True,
                "message": f"Model {os.path.basename(model_path)} deleted successfully",
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to delete model: {str(e)}"}), 500


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/api/models/upload", methods=["POST"])
def api_upload_model():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not allowed_file(file.filename):
            return (
                jsonify(
                    {
                        "error": f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
                    }
                ),
                400,
            )
        filename = secure_filename(file.filename)

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if os.path.exists(file_path):
            return jsonify({"error": "File with this name already exists"}), 409

        file.save(file_path)
        file_size = os.path.getsize(file_path)
        file_size_mb = round(file_size / (1024 * 1024), 2)

        parsed_info = parse_model_filename(filename)

        return jsonify(
            {
                "success": True,
                "message": f"Model {filename} uploaded successfully",
                "file_info": {
                    "filename": filename,
                    "size_mb": file_size_mb,
                    "model_type": parsed_info["model_type"],
                    "accuracy": parsed_info["accuracy"],
                    "formatted_time": parsed_info["formatted_time"],
                    "parsed": parsed_info["parsed"],
                },
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to upload model: {str(e)}"}), 500


@app.route("/training")
def training():
    if "user_id" not in session:
        return redirect(url_for("index"))

    return render_template("training.html", username=session.get("username"))


@app.route("/api/training/start", methods=["POST"])
def api_start_training():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json()
        if not data or "model_type" not in data:
            return jsonify({"error": "No model type provided"}), 400

        model_type = data["model_type"]
        if model_type not in ["rnn", "lstm", "bert"]:
            return jsonify({"error": "Invalid model type"}), 400

        # Generate unique training ID
        training_id = str(uuid.uuid4())

        # Get training parameters
        epochs = data.get("epochs", 100)
        learning_rate = data.get("learning_rate", 0.0001)
        batch_size = data.get("batch_size", 314)

        # Check if there's already a running training
        running_trainings = aws_training_manager.list_running_trainings()
        if running_trainings:
            return (
                jsonify(
                    {
                        "error": "Another training is already in progress. Please wait for it to complete before starting a new one.",
                        "running_training": running_trainings[0],
                    }
                ),
                409,
            )  # Conflict status code

        # Start AWS SageMaker training
        success = aws_training_manager.start_training(
            training_id=training_id,
            model_type=model_type,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        if success:
            return jsonify(
                {
                    "success": True,
                    "training_id": training_id,
                    "message": f"Training started for {model_type.upper()} model",
                }
            )
        else:
            return jsonify({"error": "Failed to start training"}), 500

    except Exception as e:
        return jsonify({"error": f"Failed to start training: {str(e)}"}), 500


@app.route("/api/training/status/<training_id>")
def api_get_training_status(training_id):
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        status = aws_training_manager.get_training_status(training_id)
        if status is None:
            return jsonify({"error": "Training not found"}), 404

        return jsonify({"success": True, "status": status})

    except Exception as e:
        return jsonify({"error": f"Failed to get training status: {str(e)}"}), 500


@app.route("/api/training/stop/<training_id>", methods=["POST"])
def api_stop_training(training_id):
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        success = aws_training_manager.stop_training(training_id)
        if success:
            return jsonify(
                {"success": True, "message": "Training stopped successfully"}
            )
        else:
            return jsonify({"error": "Failed to stop training"}), 500

    except Exception as e:
        return jsonify({"error": f"Failed to stop training: {str(e)}"}), 500


@app.route("/api/training/list")
def api_list_trainings():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        trainings = aws_training_manager.list_active_trainings()
        return jsonify({"success": True, "trainings": trainings})

    except Exception as e:
        return jsonify({"error": f"Failed to list trainings: {str(e)}"}), 500


@app.route("/api/training/logs/live/<training_id>")
def api_get_training_logs(training_id):
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        # Get the log file path for this training
        log_file_path = aws_training_manager.get_training_log_path()

        if not log_file_path or not os.path.exists(log_file_path):
            return jsonify({"success": False, "error": "Log file not found"}), 404

        # Read the last 100 lines of the log file
        with open(log_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Get the last 100 lines (or all lines if less than 100)
        recent_lines = lines

        # Clean up the lines and filter out HTTP requests
        cleaned_lines = []
        count = 0
        for line in reversed(recent_lines):
            cleaned_line = line.strip()
            if cleaned_line:  # Only include non-empty lines
                # Filter out HTTP request logs
                if not is_http_request_log(cleaned_line):
                    cleaned_lines.append(cleaned_line)
                    count += 1
            if count >= 100:
                break
        cleaned_lines.reverse()

        return jsonify(
            {
                "success": True,
                "logs": cleaned_lines,
                "total_lines": len(lines),
                "recent_lines": len(cleaned_lines),
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to read training logs: {str(e)}"}), 500


def is_http_request_log(line):
    """Check if a log line is an HTTP request log that should be filtered out"""
    # Common patterns for HTTP request logs
    http_patterns = [
        r"GET /",
        r"POST /",
        r"PUT /",
        r"DELETE /",
        r"PATCH /",
        r"HEAD /",
        r"OPTIONS /",
        r'"GET ',
        r'"POST ',
        r'"PUT ',
        r'"DELETE ',
        r'"PATCH ',
        r'"HEAD ',
        r'"OPTIONS ',
        r'HTTP/1.1"',
        r'HTTP/2"',
        r' - - \[.*\] ".*HTTP/',
        r"127\.0\.0\.1.*HTTP",
        r"::1.*HTTP",
        r"localhost.*HTTP",
        r'\[.*\] ".*" \d{3} -',  # Status code patterns
        r' - - \[.*\] ".*" \d{3} \d+',  # Status code with response size
    ]

    # Check if line matches any HTTP request pattern
    for pattern in http_patterns:
        if re.search(pattern, line):
            return True

    # Additional checks for common HTTP log formats
    if " - - [" in line and '"' in line and "HTTP" in line:
        return True

    if ("127.0.0.1" in line or "::1" in line) and ("GET" in line or "POST" in line):
        return True

    return False


def parse_model_filename(filename):
    """Parse model filename to extract model type, accuracy, and timestamp"""
    # Pattern: {model_type}_{accuracy}_{date}_{time}.{ext}
    # Example: lstm_92.8_2025-10-12_18-23-37.pt
    pattern = r"^([a-zA-Z]+)_(\d+\.?\d*)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.(pt|pth|pkl|model)$"
    match = re.match(pattern, filename)

    if match:
        model_type = match.group(1).upper()
        accuracy = float(match.group(2))
        date_str = match.group(3)
        time_str = match.group(4)

        # Parse datetime
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
    """Scan for available model files in model_save folder"""
    model_save_path = "model_save"
    available_models = []

    if not os.path.exists(model_save_path):
        return available_models

    # Define model file patterns
    model_patterns = ["*.pt", "*.pth", "*.pkl", "*.model"]

    for pattern in model_patterns:
        model_files = glob.glob(os.path.join(model_save_path, pattern))
        for model_file in model_files:
            # Get file info
            file_size = os.path.getsize(model_file)
            file_mtime = os.path.getmtime(model_file)
            filename = os.path.basename(model_file)

            # Parse filename for model info
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

    # Sort by accuracy (highest first), then by timestamp (newest first)
    available_models.sort(
        key=lambda x: (x["accuracy"], x["timestamp"] or datetime.min), reverse=True
    )

    return available_models


def init_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL
        );
        """
        )
        conn.commit()
        cur.close()
        conn.close()
        print("✅ users table initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize DB: {e}")


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5001)
