"""
Rate Limiting Implementation for TweetVerify
This module provides rate limiting functionality to prevent DOS attacks
"""

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
from flask import request, jsonify
import time
from collections import defaultdict
from threading import Lock


class SimpleRateLimiter:
    """
    A simple in-memory rate limiter for demonstration purposes.
    For production, use Redis-backed rate limiting.
    """
    
    def __init__(self):
        self.requests = defaultdict(list)
        self.lock = Lock()
    
    def is_allowed(self, key, max_requests, window_seconds):
        """
        Check if a request is allowed based on rate limits.
        
        Args:
            key: Unique identifier (e.g., IP address)
            max_requests: Maximum number of requests allowed
            window_seconds: Time window in seconds
        
        Returns:
            bool: True if request is allowed, False otherwise
        """
        with self.lock:
            now = time.time()
            cutoff = now - window_seconds
            
            # Remove old requests outside the window
            self.requests[key] = [req_time for req_time in self.requests[key] 
                                  if req_time > cutoff]
            
            # Check if under limit
            if len(self.requests[key]) < max_requests:
                self.requests[key].append(now)
                return True
            
            return False
    
    def get_retry_after(self, key, window_seconds):
        """Get the time to wait before retrying"""
        with self.lock:
            if not self.requests[key]:
                return 0
            
            oldest_request = min(self.requests[key])
            wait_time = max(0, window_seconds - (time.time() - oldest_request))
            return int(wait_time)


# Global rate limiter instance
rate_limiter = SimpleRateLimiter()


def rate_limit(max_requests, window_seconds, key_func=None):
    """
    Decorator for rate limiting endpoints.
    
    Usage:
        @rate_limit(max_requests=10, window_seconds=60)
        def my_endpoint():
            ...
    
    Args:
        max_requests: Maximum number of requests allowed in the window
        window_seconds: Time window in seconds
        key_func: Function to generate unique key (default: uses IP address)
    """
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Get identifier for rate limiting
            if key_func:
                key = key_func()
            else:
                key = request.remote_addr or request.environ.get('REMOTE_ADDR', 'unknown')
            
            # Check rate limit
            if not rate_limiter.is_allowed(key, max_requests, window_seconds):
                retry_after = rate_limiter.get_retry_after(key, window_seconds)
                
                response = jsonify({
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {max_requests} requests per {window_seconds} seconds",
                    "retry_after": retry_after
                })
                response.status_code = 429
                response.headers['Retry-After'] = str(retry_after)
                return response
            
            return f(*args, **kwargs)
        
        return wrapped
    return decorator


# Configuration for different endpoints
RATE_LIMIT_CONFIG = {
    'predict': {
        'max_requests': 100,
        'window_seconds': 60,
        'description': '100 predictions per minute per IP'
    },
    'batch_predict': {
        'max_requests': 50,
        'window_seconds': 60,
        'description': '50 batch predictions per minute per IP'
    },
    'login': {
        'max_requests': 100,
        'window_seconds': 60,
        'description': '100 login attempts per minute per IP'
    },
    'register': {
        'max_requests': 100,
        'window_seconds': 3600,
        'description': '100 registrations per hour per IP'
    },
    'models_switch': {
        'max_requests': 100,
        'window_seconds': 60,
        'description': '100 model switches per minute per IP'
    },
}


# Advanced rate limiting strategies

class TokenBucketRateLimiter:
    """
    Token bucket algorithm for more sophisticated rate limiting.
    Allows burst traffic while maintaining average rate.
    """
    
    def __init__(self, capacity, refill_rate):
        """
        Args:
            capacity: Maximum number of tokens (burst size)
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.buckets = {}
        self.lock = Lock()
    
    def consume(self, key, tokens=1):
        """
        Try to consume tokens from the bucket.
        
        Returns:
            bool: True if tokens were consumed, False if insufficient
        """
        with self.lock:
            now = time.time()
            
            if key not in self.buckets:
                self.buckets[key] = {
                    'tokens': self.capacity,
                    'last_update': now
                }
            
            bucket = self.buckets[key]
            
            # Refill tokens based on time elapsed
            time_passed = now - bucket['last_update']
            bucket['tokens'] = min(
                self.capacity,
                bucket['tokens'] + time_passed * self.refill_rate
            )
            bucket['last_update'] = now
            
            # Try to consume tokens
            if bucket['tokens'] >= tokens:
                bucket['tokens'] -= tokens
                return True
            
            return False


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts limits based on system load.
    Useful for automatic DOS protection.
    """
    
    def __init__(self, base_limit, min_limit=1, max_limit=100):
        self.base_limit = base_limit
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.current_limit = base_limit
        self.system_load = 0.0
    
    def update_system_load(self, cpu_percent, memory_percent, response_time_ms):
        """Update system load metrics"""
        # Simple heuristic for system load
        load_score = (
            cpu_percent * 0.4 +
            memory_percent * 0.3 +
            min(response_time_ms / 1000, 100) * 0.3
        )
        self.system_load = load_score / 100
        
        # Adjust rate limit based on load
        if self.system_load > 0.8:  # High load
            self.current_limit = max(self.min_limit, self.current_limit * 0.8)
        elif self.system_load < 0.5:  # Low load
            self.current_limit = min(self.max_limit, self.current_limit * 1.1)
    
    def get_current_limit(self):
        """Get the current dynamic rate limit"""
        return int(self.current_limit)


# Monitoring and logging

def log_rate_limit_violation(endpoint, ip_address, limit_config):
    """
    Log rate limit violations for security monitoring.
    In production, this should integrate with your logging/monitoring system.
    """
    import logging
    
    logger = logging.getLogger('rate_limiter')
    logger.warning(
        f"Rate limit exceeded: endpoint={endpoint}, "
        f"ip={ip_address}, "
        f"limit={limit_config['max_requests']}/{limit_config['window_seconds']}s"
    )
    
    # Could also:
    # - Send alert if same IP violates repeatedly
    # - Implement temporary IP bans
    # - Integrate with WAF or firewall


# IP-based blacklisting for severe abuse

class IPBlacklist:
    """
    Manage IP blacklist for severe rate limit violations.
    """
    
    def __init__(self, violation_threshold=10, ban_duration=3600):
        self.violations = defaultdict(int)
        self.banned = {}
        self.violation_threshold = violation_threshold
        self.ban_duration = ban_duration
        self.lock = Lock()
    
    def record_violation(self, ip):
        """Record a rate limit violation"""
        with self.lock:
            self.violations[ip] += 1
            
            if self.violations[ip] >= self.violation_threshold:
                self.banned[ip] = time.time() + self.ban_duration
                return True  # IP was banned
            
            return False  # IP not banned yet
    
    def is_banned(self, ip):
        """Check if an IP is currently banned"""
        with self.lock:
            if ip in self.banned:
                if time.time() < self.banned[ip]:
                    return True
                else:
                    # Ban expired
                    del self.banned[ip]
                    self.violations[ip] = 0
            
            return False
    
    def unban(self, ip):
        """Manually unban an IP"""
        with self.lock:
            if ip in self.banned:
                del self.banned[ip]
            self.violations[ip] = 0
