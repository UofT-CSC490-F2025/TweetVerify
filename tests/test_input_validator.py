import pytest
from marshmallow import ValidationError
from flask import Flask, request, jsonify
import io
from unittest.mock import patch
from src.security.input_validator import (
    sanitize_text,
    is_suspicious_content,
    sanitize_sql_input,
    sanitize_filename,
    PredictionSchema,
    BatchPredictionSchema,
    LoginSchema,
    RegistrationSchema,
    ModelSwitchSchema,
    validate_file_upload,
    validate_and_sanitize,
    validate_request,
    setup_validation,
    MAX_TEXT_LENGTH,
    MAX_BATCH_SIZE
)

# --- Fixtures ---
@pytest.fixture
def app():
    app = Flask(__name__)
    app.config['TESTING'] = True
    # Ensure config is clean or set explicitly
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
    setup_validation(app)
    return app

@pytest.fixture
def client(app):
    return app.test_client()

# --- Sanitization Tests ---

def test_sanitize_text():
    # Basic
    assert sanitize_text("Hello World") == "Hello World"
    # Whitespace
    assert sanitize_text("  Hello   World  ") == "Hello World"
    # HTML escaping
    assert sanitize_text("<script>alert('xss')</script>") == "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"
    # Null bytes
    assert sanitize_text("Hello\x00World") == "HelloWorld"
    # Control chars
    assert sanitize_text("Hello\x01World") == "HelloWorld"
    assert sanitize_text("Line\nBreak") == "Line Break"

def test_is_suspicious_content():
    # Safe
    is_susp, reason = is_suspicious_content("This is a normal tweet.")
    assert not is_susp
    assert reason is None

    # Script injection
    is_susp, reason = is_suspicious_content("<script>alert(1)</script>")
    assert is_susp
    assert "script content" in reason
    
    # SQL injection pattern
    is_susp, reason = is_suspicious_content("' OR '1'='1")
    assert is_susp
    assert "SQL injection" in reason

    # Excessive repetition (DOS)
    is_susp, reason = is_suspicious_content("a" * 200)
    assert is_susp
    assert "repetition" in reason
    
    # Valid repetition
    long_text = "This is a sentence. " * 10
    is_susp, reason = is_suspicious_content(long_text)
    assert not is_susp

    # Invalid Unicode (surrogates)
    is_susp, reason = is_suspicious_content("\ud800")
    # Python 3 strings handle surrogates, but encode('utf-8') fails with strict error handling
    assert is_susp
    assert "Invalid Unicode" in reason

def test_sanitize_sql_input():
    assert sanitize_sql_input("admin' --") == "admin "
    assert sanitize_sql_input("SELECT * FROM users") == "SELECT * FROM users"
    assert sanitize_sql_input("xp_cmdshell") == "cmdshell"

def test_sanitize_filename():
    assert sanitize_filename("test.txt") == "test.txt"
    assert sanitize_filename("../test.txt") == "_test.txt"
    assert sanitize_filename("test/file.txt") == "test_file.txt"
    assert sanitize_filename("test\\file.txt") == "test_file.txt"
    # Length limit
    long_name = "a" * 300 + ".txt"
    sanitized = sanitize_filename(long_name)
    assert len(sanitized) <= 255
    assert sanitized.endswith(".txt")

# --- Schema Tests ---

def test_prediction_schema():
    schema = PredictionSchema()
    
    # Valid
    data = schema.load({"text": "Valid tweet"})
    assert data["text"] == "Valid tweet"
    
    # Empty
    with pytest.raises(ValidationError) as exc:
        schema.load({"text": "   "})
    assert "text" in exc.value.messages
    
    # Too long
    with pytest.raises(ValidationError):
        schema.load({"text": "a" * (MAX_TEXT_LENGTH + 1)})
        
    # Null bytes
    with pytest.raises(ValidationError):
        schema.load({"text": "Bad\x00Byte"})

    # Sanitization happens
    data = schema.load({"text": "<b>Bold</b>"})
    assert data["text"] == "&lt;b&gt;Bold&lt;/b&gt;"

def test_batch_prediction_schema():
    schema = BatchPredictionSchema()
    
    # Valid
    data = schema.load({"texts": ["Tweet 1", "Tweet 2"]})
    assert len(data["texts"]) == 2
    
    # Too many
    with pytest.raises(ValidationError):
        schema.load({"texts": ["t"] * (MAX_BATCH_SIZE + 1)})
        
    # Null bytes in batch
    with pytest.raises(ValidationError) as exc:
        schema.load({"texts": ["Valid", "Bad\x00"]})
    assert "null bytes" in str(exc.value)
    
    # Whitespace string (passes Length(min=1) but fails strip check)
    with pytest.raises(ValidationError) as exc:
        schema.load({"texts": ["Valid", "   "]})
    assert "cannot be empty" in str(exc.value)

def test_login_schema():
    schema = LoginSchema()
    
    # Valid
    data = schema.load({"username": "valid_user", "password": "SecurePassword123"})
    assert data["username"] == "valid_user"
    
    # Invalid char
    with pytest.raises(ValidationError):
        schema.load({"username": "user@name", "password": "pw"})
        
    # SQL Injection in username that passes regex (e.g. comment)
    # Regex allows hyphens: ^[a-zA-Z0-9_-]+$
    # SQL check looks for --
    with pytest.raises(ValidationError) as exc:
        schema.load({"username": "admin--", "password": "SecurePassword123"})
    assert "Invalid username format" in str(exc.value)

    # SQL Injection in password (no regex constraint)
    with pytest.raises(ValidationError) as exc:
        schema.load({"username": "valid_user", "password": "' OR '1'='1"})
    assert "Invalid password format" in str(exc.value)

def test_registration_schema():
    schema = RegistrationSchema()
    
    # Valid
    data = schema.load({"username": "new_user", "password": "SecurePassword1"})
    assert data["username"] == "new_user"
    
    # No uppercase
    with pytest.raises(ValidationError) as exc:
        schema.load({"username": "user", "password": "password1"})
    assert "uppercase" in str(exc.value)

    # No lowercase
    with pytest.raises(ValidationError) as exc:
        schema.load({"username": "user", "password": "PASSWORD1"})
    assert "lowercase" in str(exc.value)
    
    # No number
    with pytest.raises(ValidationError) as exc:
        schema.load({"username": "user", "password": "PasswordOnly"})
    assert "number" in str(exc.value)
        
    # Common password - Mock the list to include a complex password
    # The check uses password.lower() so list should have lowercase or handle it
    with patch.object(RegistrationSchema, 'WEAK_PASSWORDS', ['weakpass1']):
        with pytest.raises(ValidationError) as exc:
            schema.load({"username": "user", "password": "WeakPass1"})
        assert "too common" in str(exc.value)

def test_model_switch_schema():
    schema = ModelSwitchSchema()
    
    # Valid
    data = schema.load({"model_path": "model_save/bert.pt", "model_type": "bert"})
    assert data["model_path"] == "model_save/bert.pt"
    
    # Path traversal
    with pytest.raises(ValidationError):
        schema.load({"model_path": "../etc/passwd"})
        
    # Invalid directory
    with pytest.raises(ValidationError):
        schema.load({"model_path": "other_dir/model.pt"})

# --- Helper Tests ---

def test_validate_and_sanitize():
    data = {"text": "Test"}
    result = validate_and_sanitize(data, PredictionSchema)
    assert result["text"] == "Test"
    
    with pytest.raises(ValidationError):
        validate_and_sanitize({}, PredictionSchema)

def test_validate_file_upload():
    class MockFile:
        def __init__(self, filename, size_bytes):
            self.filename = filename
            self.size = size_bytes
        
        def seek(self, pos, whence=0):
            pass
            
        def tell(self):
            return self.size

    # Valid
    file = MockFile("model.pt", 1024)
    valid, msg = validate_file_upload(file)
    assert valid
    assert msg is None
    
    # No file
    valid, msg = validate_file_upload(None)
    assert not valid
    
    # No extension
    file = MockFile("README", 1024)
    valid, msg = validate_file_upload(file)
    assert not valid
    assert "no extension" in msg

    # Invalid extension
    file = MockFile("script.sh", 1024)
    valid, msg = validate_file_upload(file)
    assert not valid
    
    # Dangerous filename
    file = MockFile("../model.pt", 1024)
    valid, msg = validate_file_upload(file)
    assert not valid
    
    # Too large
    file = MockFile("large.pt", 101 * 1024 * 1024)
    valid, msg = validate_file_upload(file, max_size_mb=100)
    assert not valid
    assert "too large" in msg

    # Custom extensions
    file = MockFile("data.csv", 1024)
    valid, msg = validate_file_upload(file, allowed_extensions={'csv'})
    assert valid
    
    file = MockFile("data.txt", 1024)
    valid, msg = validate_file_upload(file, allowed_extensions={'csv'})
    assert not valid

def test_validate_file_upload_exception():
    class BrokenFile:
        filename = "test.pt"
        def seek(self, *args):
            raise Exception("Disk error")
            
    file = BrokenFile()
    # Should catch exception and pass
    valid, msg = validate_file_upload(file)
    assert valid is True

def test_global_error_handler(app):
    # Trigger the global ValidationError handler configured in setup_validation
    # We can do this by raising ValidationError in a view
    @app.route('/fail')
    def fail():
        raise ValidationError("Global failure")
        
    client = app.test_client()
    resp = client.get('/fail')
    assert resp.status_code == 400
    assert resp.json['error'] == "Validation failed"
    assert resp.json['messages'] == ["Global failure"]

def test_setup_validation_default_config():
    app = Flask(__name__)
    app.config['TESTING'] = True
    # Do NOT set MAX_CONTENT_LENGTH
    setup_validation(app)
    assert app.config['MAX_CONTENT_LENGTH'] == 1 * 1024 * 1024

# --- Flask Integration Tests ---

def test_validate_request_decorator(app):
    @app.route('/predict', methods=['POST'])
    @validate_request(PredictionSchema)
    def predict():
        return jsonify({"status": "ok", "data": request.validated_data})

    client = app.test_client()
    
    # Success
    resp = client.post('/predict', json={"text": "Hello"})
    assert resp.status_code == 200
    assert resp.json['data']['text'] == "Hello"
    
    # Validation Error
    resp = client.post('/predict', json={"text": ""})
    assert resp.status_code == 400
    assert resp.json['error'] == "Validation failed"
    
    # Not JSON
    resp = client.post('/predict', data="not json")
    assert resp.status_code == 400
    assert "Request must be JSON" in resp.json['error']

def test_setup_validation_size_limit():
    # Create a fresh app with small limit
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.config['MAX_CONTENT_LENGTH'] = 1024  # 1KB
    setup_validation(app)
    client = app.test_client()
    
    @app.route('/upload', methods=['POST'])
    def upload():
        return "ok"
        
    # Small request
    resp = client.post('/upload', data="small")
    assert resp.status_code == 200
    
    # Large request
    # To reliably trigger 413 with test client, we need to actually send data
    # or rely on the before_request check which uses content_length header.
    # Werkzeug test client calculates Content-Length automatically.
    large_data = "x" * 2048
    resp = client.post('/upload', data=large_data)
    
    assert resp.status_code == 413
    assert resp.json['error'] == "Request too large"