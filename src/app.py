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

app = Flask(__name__, template_folder='web/templates')

# Global variables for model and predictor
model = None
predictor = None
device = None
current_model_path = None
current_model_type = None
available_models = []


def parse_model_filename(filename):
    """Parse model filename to extract model type, accuracy, and timestamp"""
    # Pattern: {model_type}_{accuracy}_{date}_{time}.pt
    # Example: lstm_92.8_2025-10-12_18-23-37.pt
    pattern = r'^([a-zA-Z]+)_(\d+\.?\d*)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.pt$'
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
            'model_type': model_type,
            'accuracy': accuracy,
            'timestamp': timestamp,
            'formatted_time': formatted_time,
            'parsed': True
        }
    else:
        # Fallback for files that don't match the pattern
        return {
            'model_type': 'UNKNOWN',
            'accuracy': 0.0,
            'timestamp': None,
            'formatted_time': 'Unknown',
            'parsed': False
        }


def scan_models():
    """Scan for available model files"""
    global available_models
    
    # Define model file patterns
    model_patterns = [
        "*.pth",
        "*.pt", 
        "*.pkl",
        "*.model"
    ]
    
    # Search in current directory and common model directories
    search_paths = [
        "/home/richard8/projects/aip-agoldenb/richard8/TweetVerify",
        "/home/richard8/projects/aip-agoldenb/richard8/TweetVerify/model_save",
        "/home/richard8/projects/aip-agoldenb/richard8/TweetVerify/src/models"
    ]
    
    available_models = []
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for pattern in model_patterns:
                model_files = glob.glob(os.path.join(search_path, pattern))
                for model_file in model_files:
                    # Get file info
                    file_size = os.path.getsize(model_file)
                    file_mtime = os.path.getmtime(model_file)
                    filename = os.path.basename(model_file)
                    
                    # Parse filename for model info
                    parsed_info = parse_model_filename(filename)
                    
                    available_models.append({
                        'path': model_file,
                        'name': filename,
                        'size': file_size,
                        'modified': file_mtime,
                        'size_mb': round(file_size / (1024 * 1024), 2),
                        'model_type': parsed_info['model_type'],
                        'accuracy': parsed_info['accuracy'],
                        'timestamp': parsed_info['timestamp'],
                        'formatted_time': parsed_info['formatted_time'],
                        'parsed': parsed_info['parsed']
                    })
    
    # Sort by accuracy (highest first), then by timestamp (newest first)
    available_models.sort(key=lambda x: (x['accuracy'], x['timestamp'] or datetime.min), reverse=True)
    
    print(f"Found {len(available_models)} model files:")
    for model_info in available_models:
        if model_info['parsed']:
            print(f"  - {model_info['name']} ({model_info['size_mb']} MB) - {model_info['model_type']} {model_info['accuracy']:.1f}% ({model_info['formatted_time']})")
        else:
            print(f"  - {model_info['name']} ({model_info['size_mb']} MB) - Unknown format")
    
    return available_models


def load_model(model_path=None, model_type=None):
    """Load the trained model and create predictor"""
    global model, predictor, device, current_model_path, current_model_type

    try:
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load Word2Vec model
        model_w2v = Word2Vec.load("src/w2vmodel.model")
        print("✅ Word2Vec model loaded")

        # Determine model type
        if model_type is None and model_path:
            # Try to extract model type from filename
            filename = os.path.basename(model_path)
            parsed_info = parse_model_filename(filename)
            if parsed_info['parsed']:
                model_type = parsed_info['model_type'].lower()
            else:
                model_type = 'rnn'  # Default fallback
        
        if model_type is None:
            model_type = 'rnn'  # Default model type
        
        # Create model based on type
        if model_type.lower() == 'lstm':
            model = MyLSTM(model_w2v, hidden_size=256, num_classes=2)
            current_model_type = 'LSTM'
            print(f"✅ Created LSTM model")
        elif model_type.lower() == 'rnn':
            model = MyRNN(model_w2v, hidden_size=300, num_classes=2)
            current_model_type = 'RNN'
            print(f"✅ Created RNN model")
        else:
            # Default to RNN for unknown types
            model = MyRNN(model_w2v, hidden_size=300, num_classes=2)
            current_model_type = 'RNN'
            print(f"✅ Created RNN model (default for type: {model_type})")

        # Use provided model path or default
        if model_path is None:
            model_path = "/model_save/rnn_84.2_2025-10-12_20-12-15.pt"
        
        # Load trained weights if available
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            current_model_path = model_path
            print(f"✅ Trained model loaded from {model_path}")
        else:
            print(f"⚠️  Model file {model_path} not found. Using untrained model.")
            current_model_path = None

        # Create predictor
        predictor = Predictor(model, device)
        print("✅ Predictor initialized")

        return True

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


@app.route('/')
def home():
    """Main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for text prediction"""
    try:
        # Check if predictor is loaded
        if predictor is None:
            return jsonify({
                'error': 'Model not loaded',
                'prediction': None,
                'confidence': None
            }), 500

        # Get text from request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided',
                'prediction': None,
                'confidence': None
            }), 400

        text = data['text'].strip()
        if not text:
            return jsonify({
                'error': 'Empty text provided',
                'prediction': None,
                'confidence': None
            }), 400

        # Make prediction
        prediction, confidence = predictor.predict(text)

        # Format response
        result = {
            'prediction': int(prediction),
            'confidence': float(confidence),
            'label': 'AI-Generated' if prediction == 0 else 'Human-Written',
            'text': text
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'prediction': None,
            'confidence': None
        }), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'device': str(device) if device else None,
        'current_model': current_model_path,
        'current_model_type': current_model_type
    })


@app.route('/models')
def get_models():
    """Get list of available models"""
    try:
        # Scan for models if not already done
        if not available_models:
            scan_models()
        
        # Format model info for frontend
        model_list = []
        for model_info in available_models:
            model_list.append({
                'name': model_info['name'],
                'path': model_info['path'],
                'size_mb': model_info['size_mb'],
                'is_current': model_info['path'] == current_model_path,
                'modified': model_info['modified'],
                'model_type': model_info['model_type'],
                'accuracy': model_info['accuracy'],
                'formatted_time': model_info['formatted_time'],
                'parsed': model_info['parsed']
            })
        
        return jsonify({
            'models': model_list,
            'current_model': current_model_path
        })
    
    except Exception as e:
        return jsonify({'error': f'Failed to get models: {str(e)}'}), 500


@app.route('/models/switch', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    try:
        data = request.get_json()
        if not data or 'model_path' not in data:
            return jsonify({'error': 'No model path provided'}), 400
        
        model_path = data['model_path']
        model_type = data.get('model_type', None)  # Optional model type parameter
        
        # Validate model path exists
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model file not found: {model_path}'}), 404
        
        # Try to load the new model with specified type
        if load_model(model_path, model_type):
            return jsonify({
                'success': True,
                'message': f'Successfully switched to {os.path.basename(model_path)}',
                'current_model': current_model_path,
                'model_type': model_type
            })
        else:
            return jsonify({'error': 'Failed to load the selected model'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Failed to switch model: {str(e)}'}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch text prediction"""
    try:
        if predictor is None:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': 'No texts provided'}), 400

        texts = data['texts']
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({'error': 'Invalid texts format'}), 400

        # Make batch predictions
        results = predictor.predict_batch(texts)

        # Format response
        formatted_results = []
        for i, (text, (pred, conf)) in enumerate(zip(texts, results)):
            formatted_results.append({
                'index': i,
                'text': text,
                'prediction': int(pred),
                'confidence': float(conf),
                'label': 'AI-Generated' if pred == 0 else 'Human-Written'
            })

        return jsonify({'results': formatted_results})

    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500


if __name__ == "__main__":
    # Scan for available models
    print("🔍 Scanning for available models...")
    scan_models()
    
    # Load default model on startup
    if load_model():
        print("🚀 Starting Flask app...")
        app.run(debug=True, host='0.0.0.0', port=80)
    else:
        print("❌ Failed to load model. Exiting...")
