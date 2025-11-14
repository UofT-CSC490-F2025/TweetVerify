"""
Red Team Load Testing - Aggressive Attack Scenarios
This file contains aggressive load testing scenarios designed to break the application
"""

from locust import HttpUser, task, between, tag, events
import random
import string
import json
import time

# DOS Attack Payloads
HUGE_PAYLOAD = "A" * 1024 * 1024  # 1MB of text
MASSIVE_PAYLOAD = "B" * 10 * 1024 * 1024  # 10MB of text
UNICODE_BOMB = "" * 100000  # Unicode characters can cause issues
NESTED_JSON = json.dumps({"text": {"nested": {"very": {"deep": "structure" * 1000}}}})

# SQL Injection Payloads
SQL_INJECTIONS = [
    "' OR '1'='1",
    "admin'--",
    "' UNION SELECT NULL--",
    "1'; DROP TABLE users--",
    "' OR 1=1--",
    "admin' OR '1'='1' /*",
    "') OR ('1'='1",
]

# XSS Payloads
XSS_PAYLOADS = [
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert('XSS')>",
    "javascript:alert('XSS')",
    "<svg onload=alert('XSS')>",
]


class ResourceExhaustionAttacker(HttpUser):
    """
    Red Team: Resource Exhaustion Attacks
    Attempts to exhaust server resources through various means
    """
    wait_time = between(0.1, 0.5)  # Very aggressive timing
    host = "http://18.208.150.135:5000"
    
    @task(5)
    @tag('redteam', 'dos', 'memory')
    def memory_exhaustion_massive_batch(self):
        """Attempt to exhaust memory with massive batch requests"""
        tweets = ["Test tweet for memory exhaustion"] * 1000  # 1000 tweets at once
        payload = {"texts": tweets}
        
        with self.client.post(
            "/batch_predict",
            json=payload,
            catch_response=True,
            name="RED: Massive batch (1000 tweets)"
        ) as response:
            # We expect this to potentially fail
            if response.status_code in [200, 413, 500, 503]:
                response.success()
            else:
                response.failure(f"Unexpected: {response.status_code}")
    
    @task(3)
    @tag('redteam', 'dos', 'memory')
    def memory_exhaustion_huge_text(self):
        """Attempt to exhaust memory with huge text payload"""
        payload = {"text": HUGE_PAYLOAD}
        
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="RED: Huge payload (1MB)"
        ) as response:
            if response.status_code in [200, 413, 400, 500, 503]:
                response.success()
            else:
                response.failure(f"Unexpected: {response.status_code}")
    
    @task(2)
    @tag('redteam', 'dos', 'memory')
    def memory_exhaustion_massive_text(self):
        """Attempt to exhaust memory with massive text payload"""
        payload = {"text": MASSIVE_PAYLOAD}
        
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="RED: Massive payload (10MB)",
            timeout=30
        ) as response:
            if response.status_code in [200, 413, 400, 500, 503]:
                response.success()
            else:
                response.failure(f"Unexpected: {response.status_code}")
    
    @task(3)
    @tag('redteam', 'dos', 'cpu')
    def cpu_exhaustion_rapid_fire(self):
        """Rapid-fire requests to exhaust CPU"""
        for i in range(10):  # Send 10 requests rapidly
            payload = {"text": "CPU exhaustion test " * 100}
            self.client.post(
                "/predict",
                json=payload,
                catch_response=True,
                name="RED: Rapid fire prediction"
            )
    
    @task(2)
    @tag('redteam', 'dos', 'connection')
    def connection_exhaustion(self):
        """Attempt to exhaust connection pool"""
        # Keep connections open
        payload = {"text": "Connection test"}
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="RED: Connection exhaustion"
        ) as response:
            if response.status_code == 200:
                response.success()
            time.sleep(5)  # Hold connection
    
    @task(1)
    @tag('redteam', 'dos', 'unicode')
    def unicode_bomb_attack(self):
        """Test with unicode bomb"""
        payload = {"text": UNICODE_BOMB}
        
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="RED: Unicode bomb"
        ) as response:
            if response.status_code in [200, 413, 400, 500]:
                response.success()
            else:
                response.failure(f"Unexpected: {response.status_code}")


class DatabaseAttacker(HttpUser):
    """
    Red Team: Database Attack Scenarios
    Attempts to exploit database vulnerabilities
    """
    wait_time = between(0.5, 1)
    host = "http://18.208.150.135:5001"
    
    @task(5)
    @tag('redteam', 'sql', 'injection')
    def sql_injection_login(self):
        """Test SQL injection in login"""
        injection = random.choice(SQL_INJECTIONS)
        payload = {
            "username": injection,
            "password": injection
        }
        
        with self.client.post(
            "/login",
            json=payload,
            catch_response=True,
            name="RED: SQL injection login"
        ) as response:
            if response.status_code == 200:
                response.failure("CRITICAL: SQL injection successful!")
            elif response.status_code in [400, 401, 500]:
                response.success()
            else:
                response.failure(f"Unexpected: {response.status_code}")
    
    @task(3)
    @tag('redteam', 'sql', 'injection')
    def sql_injection_register(self):
        """Test SQL injection in registration"""
        injection = random.choice(SQL_INJECTIONS)
        payload = {
            "username": injection,
            "password": "test123"
        }
        
        with self.client.post(
            "/register",
            json=payload,
            catch_response=True,
            name="RED: SQL injection register"
        ) as response:
            if response.status_code in [200, 400, 500]:
                response.success()
            else:
                response.failure(f"Unexpected: {response.status_code}")
    
    @task(4)
    @tag('redteam', 'db', 'dos')
    def database_flooding(self):
        """Flood database with registration attempts"""
        username = ''.join(random.choices(string.ascii_letters, k=20))
        payload = {
            "username": username,
            "password": "password123"
        }
        
        with self.client.post(
            "/register",
            json=payload,
            catch_response=True,
            name="RED: DB flood register"
        ) as response:
            if response.status_code in [200, 400, 500, 503]:
                response.success()
            else:
                response.failure(f"Unexpected: {response.status_code}")
    
    @task(2)
    @tag('redteam', 'db', 'connection')
    def connection_pool_exhaustion(self):
        """Attempt to exhaust database connection pool"""
        for i in range(20):  # Rapid requests
            payload = {
                "username": f"test_user_{random.randint(1, 1000)}",
                "password": "test"
            }
            self.client.post(
                "/login",
                json=payload,
                catch_response=True,
                name="RED: DB connection pool"
            )


class ModelSwitchingAttacker(HttpUser):
    """
    Red Team: Model Switching Attack
    Attempts to cause issues with rapid model switching
    """
    wait_time = between(0.2, 0.5)
    host = "http://18.208.150.135:5000"
    
    @task(5)
    @tag('redteam', 'model', 'switching')
    def rapid_model_refresh(self):
        """Rapidly refresh models list"""
        with self.client.post(
            "/models/refresh",
            json={},
            catch_response=True,
            name="RED: Rapid model refresh"
        ) as response:
            if response.status_code in [200, 500, 503]:
                response.success()
            else:
                response.failure(f"Unexpected: {response.status_code}")
    
    @task(3)
    @tag('redteam', 'model', 'prediction')
    def predict_during_model_ops(self):
        """Make predictions while doing model operations"""
        payload = {"text": "Test prediction during model operations"}
        
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="RED: Predict during ops"
        ) as response:
            if response.status_code in [200, 500, 503]:
                response.success()
            else:
                response.failure(f"Unexpected: {response.status_code}")


class CombinedRedTeamAttacker(HttpUser):
    """
    Red Team: Combined Attack Scenarios
    Combines multiple attack vectors simultaneously
    """
    wait_time = between(0.1, 0.3)  # Very aggressive
    
    def on_start(self):
        self.prediction_host = "http://18.208.150.135:5000"
        self.auth_host = "http://18.208.150.135:5001"
    
    @task(10)
    @tag('redteam', 'combined', 'dos')
    def combined_attack_prediction(self):
        """Combined attack on prediction endpoint"""
        attack_type = random.choice(['huge', 'batch', 'rapid', 'unicode'])
        
        if attack_type == 'huge':
            payload = {"text": "X" * 500000}
        elif attack_type == 'batch':
            payload = {"texts": ["Attack " * 100] * 100}
            endpoint = "/batch_predict"
        elif attack_type == 'unicode':
            payload = {"text": "" * 10000}
        else:  # rapid
            payload = {"text": "Rapid attack"}
        
        endpoint = "/batch_predict" if attack_type == 'batch' else "/predict"
        
        with self.client.post(
            f"{self.prediction_host}{endpoint}",
            json=payload,
            catch_response=True,
            name=f"RED: Combined attack [{attack_type}]"
        ) as response:
            if response.status_code in [200, 400, 413, 500, 503]:
                response.success()
    
    @task(5)
    @tag('redteam', 'combined', 'auth')
    def combined_attack_auth(self):
        """Combined attack on auth endpoints"""
        attack_type = random.choice(['sql', 'flood', 'xss'])
        
        if attack_type == 'sql':
            username = random.choice(SQL_INJECTIONS)
        elif attack_type == 'xss':
            username = random.choice(XSS_PAYLOADS)
        else:  # flood
            username = ''.join(random.choices(string.ascii_letters, k=50))
        
        payload = {
            "username": username,
            "password": "attack123"
        }
        
        with self.client.post(
            f"{self.auth_host}/login",
            json=payload,
            catch_response=True,
            name=f"RED: Combined auth [{attack_type}]"
        ) as response:
            if response.status_code in [200, 400, 401, 500]:
                response.success()


# Event hooks for tracking attack success
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Track requests that might indicate successful attacks"""
    if exception:
        print(f"[ATTACK] Exception occurred: {name} - {exception}")
    elif response_time > 10000:  # More than 10 seconds
        print(f"[ATTACK SUCCESS?] Slow response: {name} took {response_time}ms")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print summary when test stops"""
    print("\n" + "="*60)
    print("RED TEAM ATTACK SUMMARY")
    print("="*60)
    print(f"Total requests: {environment.stats.total.num_requests}")
    print(f"Total failures: {environment.stats.total.num_failures}")
    print(f"Average response time: {environment.stats.total.avg_response_time:.2f}ms")
    print(f"Max response time: {environment.stats.total.max_response_time:.2f}ms")
    print("="*60 + "\n")

