from flask import Flask, render_template, request, jsonify
import torch
import os
from src.inference.predictor import Predictor
from src.model.rnn import MyRNN
from gensim.models import Word2Vec

app = Flask(__name__, template_folder='web/templates')

# Global variables for model and predictor
model = None
predictor = None
device = None

def load_model():
    """Load the trained model and create predictor"""
    global model, predictor, device
    
    try:
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load Word2Vec model
        model_w2v = Word2Vec.load("src/w2vmodel.model")
        print("✅ Word2Vec model loaded")
        
        # Create RNN model
        model = MyRNN(model_w2v, hidden_size=300, num_classes=2)
        
        # Load trained weights if available
        model_path = "/home/richard8/projects/aip-agoldenb/richard8/TweetVerify/best_model.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Trained model loaded from {model_path}")
        else:
            print(f"⚠️  Model file {model_path} not found. Using untrained model.")
        
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
        'device': str(device) if device else None
    })

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
    # Load model on startup
    if load_model():
        print("🚀 Starting Flask app...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Exiting...")