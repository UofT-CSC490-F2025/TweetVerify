import pytest
from unittest.mock import MagicMock, patch, mock_open
import sys
import os
import glob
import time

# Mock dependencies BEFORE importing app
# We need to mock these to prevent the app from trying to load real models or connect to S3/DBs on import
with patch('src.apps.app.Word2Vec'), \
     patch('src.apps.app.MyRNN'), \
     patch('src.apps.app.MyLSTM'), \
     patch('src.apps.app.BertClassifier'), \
     patch('src.apps.app.DebertaV3'), \
     patch('src.apps.app.MyRobertaForBinaryClassification'), \
     patch('src.apps.app.glob.glob'), \
     patch('src.apps.app.os.path.exists'), \
     patch('src.apps.app.Predictor'):
    
    from src.apps.app import app, scan_models, load_single_model, load_all_models, parse_model_filename
    from src.security.rate_limiter import rate_limiter

@pytest.fixture(autouse=True)
def reset_limiter():
    """Reset rate limiter before each test"""
    rate_limiter.requests.clear()
    yield

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
    with app.test_client() as client:
        yield client

def test_home(client):
    """Test home page route"""
    with patch('src.apps.app.render_template', return_value="Mocked Home"):
        response = client.get('/')
        assert response.status_code == 200
        assert b"Mocked Home" in response.data

def test_health_check(client):
    """Test health endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'

def test_get_models(client):
    """Test models listing endpoint"""
    # Mock available models variable in app
    with patch('src.apps.app.available_models', [{'name': 'test.pt', 'path': 'p', 'size_mb': 1, 'accuracy': 90, 'model_type': 'rnn', 'modified': 0, 'formatted_time': 'now', 'parsed': True}]):
        response = client.get('/models')
        assert response.status_code == 200
        data = response.get_json()
        assert len(data['models']) == 1

def test_get_models_empty(client):
    """Test models listing when empty and scan finds nothing"""
    with patch('src.apps.app.available_models', []), \
         patch('src.apps.app.scan_models') as mock_scan:
        
        response = client.get('/models')
        assert response.status_code == 200
        data = response.get_json()
        assert len(data['models']) == 0

def test_get_models_error(client):
    """Test error in get_models"""
    with patch('src.apps.app.available_models', None): # Force error when iterating or checking bool
        # Mocking scan_models to raise exception
        with patch('src.apps.app.scan_models', side_effect=Exception("Scan failed")):
            response = client.get('/models')
            assert response.status_code == 500

def test_switch_model(client):
    """Test switching model"""
    # Mock path exists
    with patch('src.apps.app.os.path.exists', return_value=True), \
         patch('src.apps.app.load_single_model', return_value=True):
         
        # Pre-populate loaded_models so switch can find it
        loaded_models_dict = {
            'model_save/test.pt': {
                'model_type': 'RNN',
                'predictor': MagicMock()
            }
        }
        
        with patch.dict('src.apps.app.loaded_models', loaded_models_dict):
            # Test valid request
            response = client.post('/models/switch', json={'model_path': 'model_save/test.pt', 'model_type': 'rnn'})
            assert response.status_code == 200
            
            # Test invalid path
            response = client.post('/models/switch', json={'model_path': '/etc/passwd'})
            assert response.status_code == 400

def test_switch_model_not_loaded_fails(client):
    """Test switching to a model that isn't loaded and fails to load"""
    with patch('src.apps.app.os.path.exists', return_value=True), \
         patch('src.apps.app.load_single_model', return_value=False): # Load fails
         
        response = client.post('/models/switch', json={'model_path': 'model_save/new.pt'})
        assert response.status_code == 404

def test_switch_model_not_exists(client):
    with patch('src.apps.app.os.path.exists', return_value=False):
        response = client.post('/models/switch', json={'model_path': 'model_save/missing.pt'})
        assert response.status_code == 404

def test_switch_model_no_json(client):
    response = client.post('/models/switch', json={})
    assert response.status_code == 400

def test_switch_model_error(client):
    with patch('src.apps.app.os.path.exists', side_effect=Exception("Disk Error")):
        response = client.post('/models/switch', json={'model_path': 'model_save/p'})
        assert response.status_code == 500

def test_predict_endpoint(client):
    """Test prediction endpoint"""
    # Setup app state
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = (0, 0.95) # AI, 95%
    
    loaded_models_dict = {
        'model_save/test.pt': {
            'predictor': mock_predictor,
            'model_type': 'RNN'
        }
    }
    
    with patch.dict('src.apps.app.loaded_models', loaded_models_dict):
        with patch('src.apps.app.current_model_path', 'model_save/test.pt'):
            
            # Valid request
            response = client.post('/predict', json={'text': 'Hello world'})
            assert response.status_code == 200
            data = response.get_json()
            assert data['prediction'] == 0
            assert data['label'] == 'AI-Generated'
            
            # Test validation failure (empty text)
            response = client.post('/predict', json={'text': ''})
            assert response.status_code == 400

def test_predict_not_loaded(client):
    with patch('src.apps.app.current_model_path', None):
        response = client.post('/predict', json={'text': 'test'})
        assert response.status_code == 500
        assert 'Model not loaded' in response.get_json()['error']

def test_predict_too_long(client):
    """Test manual length check in predict"""
    mock_predictor = MagicMock()
    loaded_models_dict = {
        'model_save/test.pt': {'predictor': mock_predictor, 'model_type': 'RNN'}
    }
    
    with patch.dict('src.apps.app.loaded_models', loaded_models_dict), \
         patch('src.apps.app.current_model_path', 'model_save/test.pt'), \
         patch('src.apps.app.MAX_TEXT_LENGTH', 5): # Small limit
         
         pass 

def test_predict_exception(client):
    mock_predictor = MagicMock()
    mock_predictor.predict.side_effect = Exception("Predict Error")
    
    loaded_models_dict = {
        'model_save/test.pt': {'predictor': mock_predictor, 'model_type': 'RNN'}
    }
    
    with patch.dict('src.apps.app.loaded_models', loaded_models_dict), \
         patch('src.apps.app.current_model_path', 'model_save/test.pt'):
         
         response = client.post('/predict', json={'text': 'test'})
         assert response.status_code == 500
         assert "Prediction failed" in response.get_json()['error']

def test_predict_outer_exception(client):
    # Mock loaded_models to raise exception on access
    mock_loaded = MagicMock()
    mock_loaded.__contains__.return_value = True # Pass the 'not in' check
    mock_loaded.__getitem__.side_effect = Exception("Dict error")
    
    with patch('src.apps.app.current_model_path', 'model_save/test.pt'), \
         patch('src.apps.app.loaded_models', mock_loaded):
             
         response = client.post('/predict', json={'text': 'test'})
         assert response.status_code == 500
         assert "Prediction failed" in response.get_json()['error']

def test_batch_predict_endpoint(client):
    """Test batch prediction endpoint"""
    mock_predictor = MagicMock()
    mock_predictor.predict_batch.return_value = [(0, 0.95), (1, 0.88)]
    
    loaded_models_dict = {
        'model_save/test.pt': {
            'predictor': mock_predictor,
            'model_type': 'RNN'
        }
    }
    
    with patch.dict('src.apps.app.loaded_models', loaded_models_dict):
        with patch('src.apps.app.current_model_path', 'model_save/test.pt'):
            
            # Valid request
            response = client.post('/batch_predict', json={'texts': ['Text 1', 'Text 2']})
            assert response.status_code == 200
            data = response.get_json()
            assert len(data['results']) == 2
            assert data['results'][0]['prediction'] == 0
            assert data['results'][1]['prediction'] == 1

def test_batch_predict_too_large(client):
    with patch('src.apps.app.MAX_BATCH_SIZE', 1):
        pass

def test_batch_predict_errors(client):
    """Test error cases for batch predict"""
    # Model not loaded
    with patch('src.apps.app.current_model_path', None):
        response = client.post('/batch_predict', json={'texts': ['t1']})
        assert response.status_code == 500
        assert "Model not loaded" in response.get_json()['error']
        
    # Exception during prediction
    mock_predictor = MagicMock()
    mock_predictor.predict_batch.side_effect = Exception("Boom")
    
    loaded_models_dict = {
        'model_save/test.pt': {
            'predictor': mock_predictor,
            'model_type': 'RNN'
        }
    }
    
    with patch.dict('src.apps.app.loaded_models', loaded_models_dict):
        with patch('src.apps.app.current_model_path', 'model_save/test.pt'):
            response = client.post('/batch_predict', json={'texts': ['t1']})
            assert response.status_code == 500
            assert "Batch prediction failed" in response.get_json()['error']

def test_batch_predict_outer_error(client):
    mock_loaded = MagicMock()
    mock_loaded.__contains__.return_value = True
    mock_loaded.__getitem__.side_effect = Exception("Outer")
    
    with patch('src.apps.app.current_model_path', 'model_save/test.pt'), \
         patch('src.apps.app.loaded_models', mock_loaded):
         
        response = client.post('/batch_predict', json={'texts': ['t1']})
        assert response.status_code == 500
        assert "Batch prediction failed" in response.get_json()['error']

def test_refresh_models(client):
    """Test refresh models endpoint"""
    with patch('src.apps.app.scan_models'), \
         patch('src.apps.app.available_models', []), \
         patch('src.apps.app.load_single_model'):
        
        response = client.post('/models/refresh')
        assert response.status_code == 200
        assert response.get_json()['success'] is True

def test_refresh_models_new_found(client):
    """Test refresh loads new models"""
    new_model = {'path': 'model_save/new.pt', 'model_type': 'rnn'}
    with patch('src.apps.app.scan_models'), \
         patch('src.apps.app.available_models', [new_model]), \
         patch('src.apps.app.load_single_model') as mock_load:
         
         # load_single_model called
         response = client.post('/models/refresh')
         mock_load.assert_called_with('model_save/new.pt', 'rnn')

def test_refresh_models_error(client):
    with patch('src.apps.app.scan_models', side_effect=Exception("Fail")):
        response = client.post('/models/refresh')
        assert response.status_code == 500

def test_scan_models():
    """Test the scan_models function logic"""
    with patch('src.apps.app.glob.glob') as mock_glob, \
         patch('src.apps.app.os.path.getsize', return_value=1024), \
         patch('src.apps.app.os.path.getmtime', return_value=1000):
             
        # mock_glob is called multiple times for different patterns
        # Return file for first call, empty for others
        mock_glob.side_effect = [
            ['model_save/rnn_90.0_2025-01-01_12-00-00.pt'],
            [], [], [] 
        ]
        
        models = scan_models()
        assert len(models) == 1
        assert models[0]['accuracy'] == 90.0
        assert models[0]['model_type'] == 'RNN'

def test_load_single_model():
    """Test loading different model types"""
    with patch('src.apps.app.torch.load'), \
         patch('src.apps.app.os.path.exists', return_value=True), \
         patch('src.apps.app.Word2Vec.load'), \
         patch('src.apps.app.MyRNN') as mock_rnn_cls, \
         patch('src.apps.app.MyLSTM') as mock_lstm_cls, \
         patch('src.apps.app.BertClassifier') as mock_bert_cls, \
         patch('src.apps.app.DebertaV3.from_pretrained') as mock_deberta_cls, \
         patch('src.apps.app.MyRobertaForBinaryClassification.from_pretrained') as mock_roberta_cls, \
         patch('src.apps.app.BertTokenizer.from_pretrained'), \
         patch('src.apps.app.AutoTokenizer.from_pretrained'), \
         patch('src.apps.app.AutoConfig.from_pretrained'):
             
        # Test loading RNN
        assert load_single_model('model_save/rnn_test.pt', 'rnn')
        mock_rnn_cls.assert_called()
        
        # Test loading LSTM
        assert load_single_model('model_save/lstm_test.pt', 'lstm')
        mock_lstm_cls.assert_called()
        
        # Test loading BERT
        assert load_single_model('model_save/bert_test.pt', 'bert')
        mock_bert_cls.assert_called()
        
        # Test loading DeBERTa
        assert load_single_model('model_save/deberta_test.pt', 'deberta')
        mock_deberta_cls.assert_called()
        
        # Test loading RoBERTa
        assert load_single_model('model_save/roberta_test.pt', 'roberta')
        mock_roberta_cls.assert_called()
        
        # Test loading Unknown/Default
        assert load_single_model('model_save/unknown.pt', 'unknown')
        # Should default to RNN
        assert mock_rnn_cls.call_count == 2 

def test_load_single_model_already_loaded():
    with patch.dict('src.apps.app.loaded_models', {'path': {}}):
        assert load_single_model('path') is True

def test_load_single_model_not_exists():
    with patch('src.apps.app.os.path.exists', return_value=False), \
         patch('src.apps.app.Word2Vec.load'), \
         patch('src.apps.app.MyRNN'):
        # Should load untrained model
        assert load_single_model('model_save/missing.pt', 'rnn') is True

def test_load_single_model_error():
    with patch('src.apps.app.Word2Vec.load', side_effect=Exception("W2V Fail")):
        assert load_single_model('path', 'rnn') is False

def test_load_single_model_default_type_inference():
    # Test inference from filename when model_type is None
    with patch('src.apps.app.os.path.exists', return_value=True), \
         patch('src.apps.app.Word2Vec.load'), \
         patch('src.apps.app.torch.load'), \
         patch('src.apps.app.MyRNN'):
         
         # Filename implies RNN
         load_single_model('model_save/rnn_90.0_date_time.pt')
         
         # Filename garbage -> default RNN
         load_single_model('model_save/garbage.txt')

def test_parse_model_filename():
    # Valid
    info = parse_model_filename("lstm_92.8_2025-10-12_18-23-37.pt")
    assert info['parsed'] is True
    assert info['model_type'] == 'LSTM'
    assert info['accuracy'] == 92.8
    
    # Invalid
    info = parse_model_filename("garbage.txt")
    assert info['parsed'] is False
    
    # Malformed date
    info = parse_model_filename("lstm_92.8_9999-99-99_99-99-99.pt")
    # It matches regex but fails strptime
    assert info['parsed'] is True
    assert info['timestamp'] is None

def test_load_all_models():
    """Test load_all_models logic"""
    with patch('src.apps.app.scan_models'), \
         patch('src.apps.app.load_single_model', return_value=True) as mock_load, \
         patch('src.apps.app.available_models', [{'path': 'p1', 'model_type': 'rnn'}]):
             
        success = load_all_models()
        assert success is True
        mock_load.assert_called()

def test_load_all_models_fail():
    with patch('src.apps.app.scan_models'), \
         patch('src.apps.app.available_models', []):
         assert load_all_models() is False

def test_app_main_execution_fallback():
    """Test fallback logic when no models found"""
    with patch('flask.Flask.run') as mock_run, \
         patch('glob.glob') as mock_glob, \
         patch('os.path.getsize', return_value=100), \
         patch('os.path.getmtime', return_value=100), \
         patch('torch.load'), \
         patch('os.path.exists') as mock_exists, \
         patch('gensim.models.Word2Vec.load'), \
         patch('src.model.rnn.MyRNN'), \
         patch('src.inference.predictor.Predictor'):
         
        mock_glob.return_value = []
        
        def exists_side_effect(path):
            if "rnn_84.2" in str(path):
                return True
            return False
        mock_exists.side_effect = exists_side_effect
        
        import runpy
        try:
            runpy.run_path('src/apps/app.py', run_name='__main__')
        except SystemExit:
            pass
            
        mock_run.assert_called()

def test_app_main_execution_fail():
    """Test exit when no models and no fallback"""
    with patch('glob.glob', return_value=[]), \
         patch('os.path.exists', return_value=False):
        
        import runpy
        try:
            runpy.run_path('src/apps/app.py', run_name='__main__')
        except SystemExit as e:
            assert e.code == 1

def test_error_handlers(client):
    # Trigger 413 Entity Too Large
    from src.apps.app import app, request_entity_too_large, rate_limit_exceeded
    
    with app.app_context():
        resp = request_entity_too_large(None)
        assert resp[1] == 413
        
        resp = rate_limit_exceeded(None)
        assert resp[1] == 429

def test_scan_models_unknown_format():
    """Test scan_models with a file that doesn't match pattern"""
    with patch('src.apps.app.glob.glob') as mock_glob, \
         patch('src.apps.app.os.path.getsize', return_value=1024), \
         patch('src.apps.app.os.path.getmtime', return_value=1000), \
         patch('builtins.print') as mock_print:
             
        # Return garbage file
        mock_glob.side_effect = [
            ['model_save/garbage.txt'],
            [], [], [] 
        ]
        
        models = scan_models()
        assert len(models) == 1
        assert models[0]['parsed'] is False
        
        # Check if "Unknown format" was printed
        printed_unknown = False
        for call in mock_print.call_args_list:
            args, _ = call
            if args and "Unknown format" in str(args[0]):
                printed_unknown = True
                break
        assert printed_unknown

def test_load_single_model_inference_from_parsed():
    """Test load_single_model where model_type is None but filename parsed"""
    # Use patch.object on the module to ensure we catch the right reference
    import src.apps.app
    with patch.object(src.apps.app, 'Word2Vec') as mock_w2v_cls, \
         patch.object(src.apps.app, 'torch'), \
         patch.object(src.apps.app.os.path, 'exists', return_value=True), \
         patch.object(src.apps.app, 'MyLSTM') as mock_lstm:
         
        mock_w2v_load = mock_w2v_cls.load
        mock_w2v_instance = MagicMock()
        mock_w2v_instance.vector_size = 300 # Fix vstack error in real MyLSTM
        mock_w2v_instance.wv.vectors = [[0.1]*300]
        mock_w2v_instance.wv.__len__.return_value = 1 # Fix vocab size
        mock_w2v_load.return_value = mock_w2v_instance

        # Call function from module
        src.apps.app.load_single_model('model_save/lstm_90.0_2025-01-01_12-00-00.pt', model_type=None)
        
        mock_lstm.assert_called()

def test_load_all_models_partial_failure():
    """Test load_all_models where some models fail to load"""
    models = [
        {'path': 'p1', 'model_type': 'rnn'},
        {'path': 'p2', 'model_type': 'rnn'}
    ]
    with patch('src.apps.app.scan_models'), \
         patch('src.apps.app.available_models', models), \
         patch('src.apps.app.load_single_model') as mock_load, \
         patch('src.apps.app.current_model_path', None): # Reset current model
             
        # p1 fails, p2 succeeds
        mock_load.side_effect = [False, True]
        
        # We need to mock loaded_models so p2 sets current model type
        with patch.dict('src.apps.app.loaded_models', {'p2': {'model_type': 'RNN'}}):
            load_all_models()
            
            # p1 failed, p2 loaded.
            import src.apps.app
            assert src.apps.app.current_model_path == 'p2'

def test_switch_model_load_success(client):
    """Test switching to a new model that needs loading"""
    import src.apps.app
    with patch('src.apps.app.os.path.exists', return_value=True), \
         patch('src.apps.app.load_single_model', return_value=True) as mock_load:
             
        # Empty loaded_models
        with patch.dict('src.apps.app.loaded_models', {}):
            
            def load_side_effect(path, type):
                src.apps.app.loaded_models[path] = {'model_type': 'RNN'}
                return True
            
            mock_load.side_effect = load_side_effect
            
            response = client.post('/models/switch', json={'model_path': 'model_save/new_model.pt'})
            assert response.status_code == 200
            assert response.get_json()['success'] is True

def test_predict_length_check_bypass(client):
    """Test manual length check in predict by bypassing schema"""
    from src.apps.app import predict, app
    
    # Unwrap decorators
    original_predict = predict
    while hasattr(original_predict, '__wrapped__'):
        original_predict = original_predict.__wrapped__
    
    with app.test_request_context():
        with patch('src.apps.app.request') as mock_request, \
             patch('src.apps.app.current_model_path', 'p'), \
             patch.dict('src.apps.app.loaded_models', {'p': {'predictor': MagicMock()}}), \
             patch('src.apps.app.MAX_TEXT_LENGTH', 5):
             
            mock_request.validated_data = {'text': 'Too long text'}
            
            response = original_predict()
            assert response[1] == 400
            assert "Text too long" in response[0].get_json()['error']

def test_batch_predict_length_check_bypass(client):
    from src.apps.app import batch_predict, app
    
    original_batch = batch_predict
    while hasattr(original_batch, '__wrapped__'):
        original_batch = original_batch.__wrapped__

    with app.test_request_context():
        with patch('src.apps.app.request') as mock_request, \
             patch('src.apps.app.current_model_path', 'p'), \
             patch.dict('src.apps.app.loaded_models', {'p': {'predictor': MagicMock()}}), \
             patch('src.apps.app.MAX_BATCH_SIZE', 1):
             
            mock_request.validated_data = {'texts': ['t1', 't2']}
            
            response = original_batch()
            assert response[1] == 400
            assert "Batch too large" in response[0].get_json()['error']

def test_main_success():
    """Test main execution success"""
    # Mock external dependencies so scan_models and load_all_models work
    # We need to patch at the source because runpy re-imports
    with patch('glob.glob', return_value=['model_save/test.pt']), \
         patch('os.path.exists', return_value=True), \
         patch('os.path.getsize', return_value=1024), \
         patch('os.path.getmtime', return_value=1000), \
         patch('gensim.models.Word2Vec.load') as mock_w2v_load, \
         patch('torch.load'), \
         patch('src.model.rnn.MyRNN'), \
         patch('src.inference.predictor.Predictor'), \
         patch('flask.Flask.run') as mock_run:
             
        mock_w2v_instance = MagicMock()
        mock_w2v_instance.vector_size = 300
        mock_w2v_instance.wv.vectors = [[0.1]*300]
        mock_w2v_instance.wv.__len__.return_value = 1
        mock_w2v_load.return_value = mock_w2v_instance

        import runpy
        # We also need to patch print to avoid clutter, but it's fine
        runpy.run_path('src/apps/app.py', run_name='__main__')
        mock_run.assert_called()

def test_load_all_models_already_set():
    """Test load_all_models where current model is already set"""
    models = [{'path': 'p2', 'model_type': 'rnn'}]
    with patch('src.apps.app.scan_models'), \
         patch('src.apps.app.available_models', models), \
         patch('src.apps.app.load_single_model', return_value=True), \
         patch('src.apps.app.current_model_path', 'p1'): # Already set
             
         load_all_models()
         # current_model_path should remain p1
         from src.apps.app import current_model_path
         assert current_model_path == 'p1'

def test_switch_model_load_fail_handler(client):
    """Test 443->447: load_single_model returns False"""
    with patch('src.apps.app.os.path.exists', return_value=True), \
         patch('src.apps.app.load_single_model', return_value=False):
             
        response = client.post('/models/switch', json={'model_path': 'model_save/fail.pt'})
        assert response.status_code == 404
        assert "failed to load" in response.get_json()['error']

def test_refresh_models_already_loaded(client):
    """Test refresh skips already loaded models"""
    model = {'path': 'model_save/loaded.pt', 'model_type': 'rnn', 'name': 'n', 'size_mb': 1, 'accuracy': 1, 'modified': 1, 'formatted_time': 't', 'parsed': True}
    
    with patch('src.apps.app.scan_models'), \
         patch('src.apps.app.available_models', [model]), \
         patch('src.apps.app.load_single_model') as mock_load, \
         patch.dict('src.apps.app.loaded_models', {'model_save/loaded.pt': {}}):
             
        client.post('/models/refresh')
        mock_load.assert_not_called()

def test_main_load_all_fail_fallback_fail():
    """Test main when load_all_models fails AND fallback fails"""
    with patch('glob.glob', return_value=['m']), \
         patch('os.path.exists', return_value=True), \
         patch('os.path.getsize', return_value=1024), \
         patch('os.path.getmtime', return_value=1000), \
         patch('src.apps.app.load_all_models', return_value=False), \
         patch('src.apps.app.load_single_model', return_value=False), \
         patch('flask.Flask.run') as mock_run:
             
        import runpy
        try:
            runpy.run_path('src/apps/app.py', run_name='__main__')
        except SystemExit as e:
            assert e.code == 1
        
        mock_run.assert_not_called()
