"""
Input Validation Module for TweetVerify
Provides comprehensive input validation to prevent injection attacks and ensure data integrity
"""

from marshmallow import Schema, fields, validate, ValidationError, validates_schema
from flask import jsonify
from functools import wraps
import re
import html


# Configuration
MAX_TEXT_LENGTH = 10000
MAX_BATCH_SIZE = 50
MAX_USERNAME_LENGTH = 50
MAX_PASSWORD_LENGTH = 128
MIN_PASSWORD_LENGTH = 8


class PredictionSchema(Schema):
    """Schema for single prediction requests"""
    text = fields.Str(
        required=True,
        validate=[
            validate.Length(min=1, max=MAX_TEXT_LENGTH, 
                          error=f"Text must be between 1 and {MAX_TEXT_LENGTH} characters"),
        ]
    )
    
    @validates_schema
    def validate_text_content(self, data, **kwargs):
        """Additional validation for text content"""
        text = data.get('text', '')
        
        # Check for null bytes (potential attack vector)
        if '\x00' in text:
            raise ValidationError("Text contains invalid null bytes", "text")
        
        # Check for excessive whitespace (potential DOS)
        if len(text.strip()) == 0:
            raise ValidationError("Text cannot be empty or only whitespace", "text")
        
        # Sanitize text (remove or escape dangerous characters)
        data['text'] = sanitize_text(text)


class BatchPredictionSchema(Schema):
    """Schema for batch prediction requests"""
    texts = fields.List(
        fields.Str(validate=validate.Length(min=1, max=MAX_TEXT_LENGTH)),
        required=True,
        validate=validate.Length(
            min=1, max=MAX_BATCH_SIZE,
            error=f"Batch size must be between 1 and {MAX_BATCH_SIZE}"
        )
    )
    
    @validates_schema
    def validate_batch_content(self, data, **kwargs):
        """Validate batch content"""
        texts = data.get('texts', [])
        
        # Validate each text in batch
        sanitized_texts = []
        for i, text in enumerate(texts):
            if '\x00' in text:
                raise ValidationError(f"Text at index {i} contains invalid null bytes")
            
            if len(text.strip()) == 0:
                raise ValidationError(f"Text at index {i} cannot be empty")
            
            sanitized_texts.append(sanitize_text(text))
        
        data['texts'] = sanitized_texts


class LoginSchema(Schema):
    """Schema for login requests"""
    username = fields.Str(
        required=True,
        validate=[
            validate.Length(min=3, max=MAX_USERNAME_LENGTH),
            validate.Regexp(
                r'^[a-zA-Z0-9_-]+$',
                error="Username can only contain letters, numbers, underscores, and hyphens"
            )
        ]
    )
    password = fields.Str(
        required=True,
        validate=validate.Length(
            min=MIN_PASSWORD_LENGTH, max=MAX_PASSWORD_LENGTH,
            error=f"Password must be between {MIN_PASSWORD_LENGTH} and {MAX_PASSWORD_LENGTH} characters"
        )
    )
    
    @validates_schema
    def validate_credentials(self, data, **kwargs):
        """Additional validation for credentials"""
        username = data.get('username', '')
        password = data.get('password', '')
        
        # Check for SQL injection patterns
        sql_patterns = [
            r"('|(\\'))",  # Single quotes
            r"(;|--|#|\/\*|\*\/)",  # SQL comments
            r"(\bunion\b|\bselect\b|\binsert\b|\bupdate\b|\bdelete\b|\bdrop\b)",  # SQL keywords
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, username, re.IGNORECASE):
                raise ValidationError("Invalid username format", "username")
            if re.search(pattern, password, re.IGNORECASE):
                raise ValidationError("Invalid password format", "password")
        
        # Sanitize
        data['username'] = sanitize_sql_input(username)


class RegistrationSchema(LoginSchema):
    """Schema for registration requests (extends LoginSchema)"""
    
    WEAK_PASSWORDS = ['password', '12345678', 'qwerty123', 'admin123']

    @validates_schema
    def validate_password_strength(self, data, **kwargs):
        """Validate password strength"""
        password = data.get('password', '')
        
        # Check password complexity
        if not re.search(r'[A-Z]', password):
            raise ValidationError("Password must contain at least one uppercase letter", "password")
        
        if not re.search(r'[a-z]', password):
            raise ValidationError("Password must contain at least one lowercase letter", "password")
        
        if not re.search(r'[0-9]', password):
            raise ValidationError("Password must contain at least one number", "password")
        
        # Check for common weak passwords
        if password.lower() in self.WEAK_PASSWORDS:
            raise ValidationError("Password is too common", "password")


class ModelSwitchSchema(Schema):
    """Schema for model switching requests"""
    model_path = fields.Str(
        required=True,
        validate=validate.Length(min=1, max=500)
    )
    model_type = fields.Str(
        required=False,
        validate=validate.OneOf(['rnn', 'lstm', 'bert'], 
                               error="Model type must be one of: rnn, lstm, bert")
    )
    
    @validates_schema
    def validate_path_safety(self, data, **kwargs):
        """Validate model path to prevent path traversal attacks"""
        model_path = data.get('model_path', '')
        
        # Check for path traversal attempts
        dangerous_patterns = ['..', '~', '/etc/', '/root/', '\\', 'C:']
        for pattern in dangerous_patterns:
            if pattern in model_path:
                raise ValidationError("Invalid model path", "model_path")
        
        # Ensure path is within allowed directory
        if not model_path.startswith('model_save/') and not model_path.startswith('./model_save/'):
            raise ValidationError("Model path must be in model_save directory", "model_path")


def sanitize_text(text):
    """
    Sanitize text input to prevent XSS and other attacks.
    
    Args:
        text: Raw text input
    
    Returns:
        Sanitized text
    """
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Escape HTML entities (prevent XSS)
    text = html.escape(text)
    
    # Remove control characters (except common ones like newline, tab)
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # Limit consecutive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def sanitize_sql_input(text):
    """
    Sanitize input to prevent SQL injection.
    Note: This is a backup. Always use parameterized queries as primary defense.
    
    Edge Case Note: This function performs a single-pass replacement.
    Constructed inputs like 'xpxp__' will result in 'xp_' after sanitization.
    See tests/test_input_validator_depth.py::test_sanitize_sql_input_recursive_bypass
    
    Args:
        text: Raw text input
    
    Returns:
        Sanitized text
    """
    # Remove dangerous SQL characters
    dangerous_chars = ["'", '"', ';', '--', '/*', '*/', 'xp_', 'sp_']
    
    for char in dangerous_chars:
        text = text.replace(char, '')
    
    return text


def validate_request(schema_class):
    """
    Decorator to validate request data against a schema.
    
    Usage:
        @validate_request(PredictionSchema)
        def predict():
            data = request.get_json()
            # data is now validated and sanitized
            ...
    
    Args:
        schema_class: Marshmallow schema class to validate against
    
    Returns:
        Decorator function
    """
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            from flask import request
            
            # Get request data
            if request.is_json:
                data = request.get_json()
            else:
                return jsonify({"error": "Request must be JSON"}), 400
            
            # Validate
            schema = schema_class()
            try:
                validated_data = schema.load(data)
                
                # Replace request data with validated data
                request.validated_data = validated_data
                
                return f(*args, **kwargs)
                
            except ValidationError as err:
                return jsonify({
                    "error": "Validation failed",
                    "messages": err.messages
                }), 400
        
        return wrapped
    return decorator


def validate_file_upload(file, allowed_extensions=None, max_size_mb=None):
    """
    Validate file uploads.
    
    Args:
        file: FileStorage object from Flask
        allowed_extensions: Set of allowed file extensions
        max_size_mb: Maximum file size in MB
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if allowed_extensions is None:
        allowed_extensions = {'pt', 'pth', 'pkl', 'model'}
    
    if max_size_mb is None:
        max_size_mb = 100  # 100MB default
    
    # Check if file exists
    if not file or file.filename == '':
        return False, "No file provided"
    
    # Check extension
    if '.' not in file.filename:
        return False, "File has no extension"
    
    ext = file.filename.rsplit('.', 1)[1].lower()
    if ext not in allowed_extensions:
        return False, f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
    
    # Check filename for dangerous characters
    dangerous_patterns = ['..', '/', '\\', '\x00']
    for pattern in dangerous_patterns:
        if pattern in file.filename:
            return False, "Invalid filename"
    
    # Check file size (if possible)
    try:
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if size > max_size_mb * 1024 * 1024:
            return False, f"File too large. Maximum size: {max_size_mb}MB"
    except:
        pass  # Size check not possible for all file types
    
    return True, None


def sanitize_filename(filename):
    """
    Sanitize filename to prevent directory traversal and other attacks.
    
    Edge Case Note: This function does not filter Windows reserved filenames (CON, PRN, etc.).
    It relies on the OS to handle or reject them, or they are considered valid in this context.
    See tests/test_input_validator_depth.py::test_sanitize_filename_reserved_windows_names
    
    Args:
        filename: Original filename
    
    Returns:
        Safe filename
    """
    # Remove path components
    filename = filename.replace('\\', '_').replace('/', '_')
    
    # Remove dangerous characters
    filename = re.sub(r'[^\w\s.-]', '', filename)
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:250] + ('.' + ext if ext else '')
    
    return filename


# Content Security Validation

def is_suspicious_content(text):
    """
    Check if content contains suspicious patterns that might indicate an attack.
    
    Returns:
        tuple: (is_suspicious, reason)
    """
    # Check for excessive repetition (potential DOS)
    if len(text) > 100:
        unique_chars = len(set(text))
        if unique_chars < 10:
            return True, "Excessive character repetition detected"
    
    # Check for binary content
    try:
        text.encode('utf-8')
    except UnicodeEncodeError:
        return True, "Invalid Unicode characters"
    
    # Check for script tags (XSS)
    script_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'onerror\s*=',
        r'onload\s*=',
    ]
    for pattern in script_patterns:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            return True, "Potentially malicious script content"
    
    # Check for SQL injection patterns (should be caught earlier, but double-check)
    sql_injection_patterns = [
        r"('\s*OR\s*'1'\s*=\s*'1)",
        r"('\s*OR\s*1\s*=\s*1)",
        r"(;\s*DROP\s+TABLE)",
        r"(UNION\s+SELECT)",
    ]
    for pattern in sql_injection_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True, "Potential SQL injection attempt"
    
    return False, None


def validate_and_sanitize(data, schema_class):
    """
    Convenience function to validate and sanitize data.
    
    Args:
        data: Dictionary of data to validate
        schema_class: Marshmallow schema class
    
    Returns:
        Validated and sanitized data
    
    Raises:
        ValidationError: If validation fails
    """
    schema = schema_class()
    validated_data = schema.load(data)
    return validated_data


# Example integration with Flask app

def setup_validation(app):
    """
    Setup validation for Flask app.
    Configures error handlers and validation middleware.
    """
    
    @app.errorhandler(ValidationError)
    def handle_validation_error(error):
        """Global handler for validation errors"""
        return jsonify({
            "error": "Validation failed",
            "messages": error.messages
        }), 400
    
    @app.before_request
    def validate_request_size():
        """Validate request size before processing"""
        from flask import request
        
        # Check content length
        if request.content_length and request.content_length > app.config.get('MAX_CONTENT_LENGTH', 1024 * 1024):
            return jsonify({
                "error": "Request too large",
                "max_size": app.config.get('MAX_CONTENT_LENGTH')
            }), 413
    
    # Set default max content length if not set
    if app.config.get('MAX_CONTENT_LENGTH') is None:
        app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB
    
    return app
