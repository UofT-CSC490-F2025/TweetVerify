import pytest
from marshmallow import ValidationError
from src.security.input_validator import (
    sanitize_text,
    is_suspicious_content,
    sanitize_sql_input,
    sanitize_filename
)

def test_sanitize_text():
    # Test basic sanitization
    assert sanitize_text("Hello World") == "Hello World"
    assert sanitize_text("  Hello   World  ") == "Hello World"
    
    # Test HTML escaping
    assert sanitize_text("<script>alert('xss')</script>") == "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"
    
    # Test null byte removal
    assert sanitize_text("Hello\x00World") == "HelloWorld"

def test_is_suspicious_content():
    # Test safe content
    is_susp, reason = is_suspicious_content("This is a normal tweet.")
    assert not is_susp
    assert reason is None

    # Test script injection
    is_susp, reason = is_suspicious_content("<script>alert(1)</script>")
    assert is_susp
    assert "script content" in reason
    
    # Test SQL injection pattern
    is_susp, reason = is_suspicious_content("' OR '1'='1")
    assert is_susp
    assert "SQL injection" in reason

def test_sanitize_sql_input():
    assert sanitize_sql_input("admin' --") == "admin "
    assert sanitize_sql_input("SELECT * FROM users") == "SELECT * FROM users"
    
def test_sanitize_filename():
    assert sanitize_filename("test.txt") == "test.txt"
    assert sanitize_filename("../test.txt") == "_test.txt"
    assert sanitize_filename("test/file.txt") == "test_file.txt"
    assert sanitize_filename("test\\file.txt") == "test_file.txt"

