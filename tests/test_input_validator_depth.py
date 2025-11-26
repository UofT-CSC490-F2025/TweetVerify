"""
Part 3: Test Coverage Depth for Input Validator Module
This module contains unit tests targeting edge cases, failure modes, and security bypass attempts
for src/security/input_validator.py.
"""

import pytest
from marshmallow import ValidationError
from unittest.mock import MagicMock
import os

from src.security.input_validator import (
    sanitize_text,
    sanitize_filename,
    sanitize_sql_input,
    is_suspicious_content,
    validate_file_upload,
    LoginSchema,
    RegistrationSchema,
    PredictionSchema,
    BatchPredictionSchema
)

class TestInputValidatorDepth:
    
    # --- Sanitize Text Edge Cases ---
    
    def test_sanitize_text_recursive_html(self):
        """
        Edge Case: Recursive HTML tags.
        Some sanitizers might strip tags, allowing nested tags to become valid.
        sanitize_text uses html.escape, so it should safely escape everything without recursive issues.
        Input: <scr<script>ipt>
        Expected: &lt;scr&lt;script&gt;ipt&gt;
        """
        text = "<scr<script>ipt>"
        sanitized = sanitize_text(text)
        assert sanitized == "&lt;scr&lt;script&gt;ipt&gt;"
        assert "<script>" not in sanitized

    def test_sanitize_text_invisible_characters(self):
        """
        Edge Case: Invisible characters (e.g., zero-width space \u200b).
        These can be used to bypass keyword filters.
        sanitize_text removes control characters (ord < 32), but \u200b is ord 8203.
        This tests the behavior for Unicode format characters.
        """
        text = "Bad\u200bWord"
        sanitized = sanitize_text(text)
        # Logic: ''.join(char for char in text if ord(char) >= 32 ...)
        # 8203 >= 32, so it is preserved.
        assert "\u200b" in sanitized
        
    def test_sanitize_text_unicode_homoglyphs(self):
        """
        Edge Case: Unicode Homoglyphs.
        Cyrillic 'a' (U+0430) looks like Latin 'a' (U+0061).
        This tests if the sanitizer preserves unicode characters that look suspicious.
        """
        cyrillic_a = "\u0430"
        text = f"admin{cyrillic_a}"
        sanitized = sanitize_text(text)
        assert cyrillic_a in sanitized
        # Note: This is 'correct' behavior for this function (it doesn't normalize), 
        # but good to verify as a security property.

    # --- Sanitize SQL Input Failure Modes ---

    def test_sanitize_sql_input_recursive_bypass(self):
        """
        Failure Mode: Single-pass sanitization.
        The function iterates through the blacklist once.
        If removing a token creates a new forbidden token, it's a vulnerability.
        Example: 'xp_' is forbidden. Input 'xpxp__' -> removes inner 'xp_' -> results in 'xp_'.
        """
        forbidden = "xp_"
        bypass_attempt = "xpxp__"
        result = sanitize_sql_input(bypass_attempt)
        # If vulnerable, result will be 'xp_'
        # If secure (recursive/regex), it should be empty or safe.
        # Based on code analysis, we expect this to fail (return dangerous content)
        # So we assert checking for this failure mode.
        assert result == "xp_"  # This confirms the vulnerability/limitation exists

    def test_sanitize_sql_input_unicode_quotes(self):
        """
        Edge Case: Unicode quotes that might be normalized to SQL quotes by DB.
        U+02BC (ʼ) Modifier Letter Apostrophe.
        """
        text = "adminʼ OR 1=1"
        sanitized = sanitize_sql_input(text)
        # The sanitizer only removes ASCII single quote '.
        # Unicode quote should remain.
        assert "ʼ" in sanitized

    # --- Sanitize Filename Edge Cases ---

    def test_sanitize_filename_reserved_windows_names(self):
        """
        Edge Case: Windows Reserved Filenames (CON, PRN, NUL).
        These are valid alphanumeric strings but dangerous on Windows.
        The regex [^\\w\\s.-] allows them.
        """
        filename = "CON.txt"
        sanitized = sanitize_filename(filename)
        assert sanitized == "CON.txt"
        # This is a valid filename in terms of the function's logic,
        # but represents a cross-platform risk.

    def test_sanitize_filename_extreme_length(self):
        """
        Edge Case: Very long filename.
        Filesystem limits are usually 255 bytes.
        The function has a check > 255.
        """
        long_name = "a" * 300 + ".txt"
        sanitized = sanitize_filename(long_name)
        assert len(sanitized) <= 255
        assert sanitized.endswith(".txt")

    def test_sanitize_filename_dots_only(self):
        """
        Edge Case: Filename consisting of only dots/spaces.
        '...' or '   '
        The function does strip('. ').
        """
        filename = "..."
        sanitized = sanitize_filename(filename)
        assert sanitized == "" # Should become empty

    # --- Validation Logic Edge Cases ---

    def test_is_suspicious_content_polyglot(self):
        """
        Edge Case: Polyglot-like patterns.
        Text containing javascript: protocol but obscured.
        """
        text = "Visit java\tscript:alert(1)"
        # Tab is allowed in sanitize_text but logic here uses regex.
        # Regex r'javascript:' usually doesn't match if split by whitespace unless \s* is used.
        is_suspicious, _ = is_suspicious_content(text)
        # The pattern is r'javascript:', so 'java script:' won't match
        assert not is_suspicious 

    def test_file_upload_zip_bomb_size(self):
        """
        Edge Case: Large file check (seek/tell).
        Simulate a file object that claims to be huge.
        """
        mock_file = MagicMock()
        mock_file.filename = "bomb.pt"
        # Simulate file size > 100MB
        # seek(0, 2) goes to end, tell() returns position
        mock_file.tell.return_value = 101 * 1024 * 1024
        
        is_valid, error = validate_file_upload(mock_file, max_size_mb=100)
        assert not is_valid
        assert "File too large" in error

    def test_registration_weak_password_case_insensitivity(self):
        """
        Edge Case: Weak password check should be case-insensitive.
        'PASSWORD' vs 'password'.
        """
        schema = RegistrationSchema()
        
        # Monkey patch the class list for this test
        original_weak = RegistrationSchema.WEAK_PASSWORDS
        # The logic expects the list to contain lowercased passwords
        RegistrationSchema.WEAK_PASSWORDS = ['weakpass1']
        
        try:
            # 'WeakPass1' has Upper, Lower, Digit.
            # Input is mixed case, check converts to lower and compares with list
            data = {"username": "valid_user", "password": "WeakPass1"}
            with pytest.raises(ValidationError) as excinfo:
                schema.load(data)
            assert "Password is too common" in str(excinfo.value)
        finally:
            RegistrationSchema.WEAK_PASSWORDS = original_weak

    def test_prediction_schema_whitespace_only(self):
        """
        Edge Case: Text with only invisible whitespace characters.
        The schema check uses .strip().
        """
        schema = PredictionSchema()
        data = {"text": "   \t   \n"}
        with pytest.raises(ValidationError) as excinfo:
            schema.load(data)
        assert "Text cannot be empty" in str(excinfo.value)


