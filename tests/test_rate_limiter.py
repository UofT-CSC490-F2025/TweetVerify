import pytest
import time
from unittest.mock import patch, MagicMock
from flask import Flask, request
from src.security.rate_limiter import (
    SimpleRateLimiter,
    TokenBucketRateLimiter,
    AdaptiveRateLimiter,
    IPBlacklist,
    rate_limit,
    log_rate_limit_violation,
    setup_flask_limiter,
    apply_rate_limiting_to_app,
    RATE_LIMIT_CONFIG
)

# --- SimpleRateLimiter Tests ---

def test_simple_rate_limiter():
    limiter = SimpleRateLimiter()
    key = "test_ip"
    
    # Allowed
    assert limiter.is_allowed(key, 2, 10)
    assert limiter.is_allowed(key, 2, 10)
    
    # Denied
    assert not limiter.is_allowed(key, 2, 10)
    
    # Wait for retry
    # Should be blocked
    assert limiter.get_retry_after(key, 10) > 0
    
    # New key
    assert limiter.is_allowed("other_ip", 2, 10)

def test_simple_rate_limiter_window():
    limiter = SimpleRateLimiter()
    key = "test_ip"
    
    with patch('time.time') as mock_time:
        mock_time.return_value = 100
        assert limiter.is_allowed(key, 1, 10)
        assert not limiter.is_allowed(key, 1, 10)
        
        # Move forward in time past window
        mock_time.return_value = 111
        assert limiter.is_allowed(key, 1, 10)

def test_get_retry_after_empty():
    limiter = SimpleRateLimiter()
    assert limiter.get_retry_after("unknown", 10) == 0

# --- TokenBucketRateLimiter Tests ---

def test_token_bucket():
    limiter = TokenBucketRateLimiter(capacity=2, refill_rate=1)
    key = "user"
    
    # Consume initial tokens
    assert limiter.consume(key)
    assert limiter.consume(key)
    assert not limiter.consume(key)
    
    # Refill
    # Need to mock time to simulate refill
    with patch('time.time') as mock_time:
        # Set initial time
        mock_time.return_value = 1000
        limiter = TokenBucketRateLimiter(capacity=2, refill_rate=1)
        limiter.consume(key) # 1 token used, 1 left
        
        # Advance 1.5 seconds -> +1.5 tokens, capped at 2
        mock_time.return_value = 1001.5
        assert limiter.consume(key) # 2
        assert limiter.consume(key) # 1
        assert not limiter.consume(key) # 0

# --- AdaptiveRateLimiter Tests ---

def test_adaptive_rate_limiter():
    limiter = AdaptiveRateLimiter(base_limit=100, min_limit=10, max_limit=200)
    
    assert limiter.get_current_limit() == 100
    
    # High load
    # Response time in ms. To get high load, we need large response time or just rely on formula.
    # Formula: cpu*0.4 + mem*0.3 + min(resp/1000, 100)*0.3
    # To get > 0.8 (which is 80/100), we need score > 80.
    # Let's use max values.
    # CPU=100 -> 40
    # MEM=100 -> 30
    # Total 70. We need 10 more from response time.
    # 10 = min(resp/1000, 100) * 0.3 => min(...) = 33.33
    # resp/1000 = 33.33 => resp = 33333 ms
    limiter.update_system_load(cpu_percent=100, memory_percent=100, response_time_ms=40000)
    assert limiter.system_load > 0.8
    assert limiter.get_current_limit() < 100
    
    # Low load
    # Reset to known state
    limiter.current_limit = 100
    limiter.update_system_load(cpu_percent=10, memory_percent=10, response_time_ms=10)
    assert limiter.system_load < 0.5
    assert limiter.get_current_limit() > 100

# --- IPBlacklist Tests ---

def test_ip_blacklist():
    blacklist = IPBlacklist(violation_threshold=2, ban_duration=10)
    ip = "1.2.3.4"
    
    assert not blacklist.record_violation(ip) # 1
    assert blacklist.record_violation(ip)     # 2 -> Ban
    
    assert blacklist.is_banned(ip)
    
    # Unban manually
    blacklist.unban(ip)
    assert not blacklist.is_banned(ip)
    
    # Ban Expiration
    blacklist.record_violation(ip)
    blacklist.record_violation(ip) # Banned
    
    future_time = time.time() + 20
    with patch('time.time') as mock_time:
        mock_time.return_value = future_time
        assert not blacklist.is_banned(ip)

def test_ip_blacklist_no_ban():
    blacklist = IPBlacklist(violation_threshold=10)
    assert not blacklist.is_banned("1.1.1.1")
    blacklist.unban("1.1.1.1") # Should not error

# --- Flask Integration Tests ---

def test_rate_limit_decorator():
    app = Flask(__name__)
    app.config['TESTING'] = True
    
    @app.route('/test')
    @rate_limit(max_requests=2, window_seconds=60)
    def test_route():
        return "success"
        
    client = app.test_client()
    
    # Reset limiter for this test
    from src.security.rate_limiter import rate_limiter
    rate_limiter.requests.clear()
    
    assert client.get('/test').status_code == 200
    assert client.get('/test').status_code == 200
    resp = client.get('/test')
    assert resp.status_code == 429
    assert resp.json['error'] == "Rate limit exceeded"
    assert 'Retry-After' in resp.headers

def test_rate_limit_custom_key():
    app = Flask(__name__)
    
    @app.route('/custom')
    @rate_limit(max_requests=1, window_seconds=60, key_func=lambda: "static_key")
    def custom():
        return "ok"
        
    client = app.test_client()
    from src.security.rate_limiter import rate_limiter
    rate_limiter.requests.clear()
    
    assert client.get('/custom').status_code == 200
    assert client.get('/custom').status_code == 429

# --- Helper Tests ---

def test_log_rate_limit_violation():
    with patch('logging.getLogger') as mock_logger:
        log_rate_limit_violation("endpoint", "1.1.1.1", {"max_requests": 10, "window_seconds": 60})
        mock_logger.return_value.warning.assert_called_once()

def test_setup_flask_limiter():
    app = Flask(__name__)
    limiter = setup_flask_limiter(app)
    assert limiter is not None

def test_apply_rate_limiting_to_app():
    app = Flask(__name__)
    limiter = apply_rate_limiting_to_app(app)
    assert limiter is not None

