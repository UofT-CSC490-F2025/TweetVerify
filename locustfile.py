"""
Comprehensive Load Testing for TweetVerify Application
This file contains load tests for both the prediction API (port 5000) and auth API (port 5001)
"""

from locust import HttpUser, task, between, tag, TaskSet
import random
import json

# Sample tweet texts for testing
SAMPLE_TWEETS = [
    "Just finished a great workout at the gym! Feeling energized and ready to tackle the day. #fitness #motivation",
    "The new AI technology is revolutionizing how we approach complex problems in healthcare and medicine.",
    "Breaking: Major announcement expected from tech industry leaders regarding artificial intelligence regulations.",
    "Can't believe how beautiful the sunset is tonight. Nature never ceases to amaze me. 🌅",
    "Important update: Our team has been working tirelessly to improve user experience and security features.",
    "Just tried the new restaurant downtown. The food was amazing! Highly recommend the pasta dishes.",
    "Climate change is one of the most pressing issues of our time. We need to act now before it's too late.",
    "Excited to announce that our project has reached 1 million users! Thank you all for your support!",
    "The economic implications of recent policy changes remain unclear. Experts are divided on potential outcomes.",
    "Coffee is life. Can't start my day without it. ☕ #coffeeaddict #mondaymood",
    "New research suggests that machine learning models can predict disease outcomes with 95% accuracy.",
    "Just deployed our latest feature update. Let us know what you think! Feedback is always appreciated.",
    "The intersection of technology and society raises important ethical questions we must address.",
    "Game night with friends was so much fun! Nothing beats quality time with good people. 🎮",
    "Our quarterly results exceeded expectations. Proud of the team's hard work and dedication.",
]

# Generate longer texts for stress testing
LONG_TWEET = " ".join(["This is a very long tweet text that is designed to test the system's handling of large inputs."] * 50)
VERY_LONG_TWEET = " ".join(["Testing maximum input length with repeated text."] * 200)


class PredictionTasks(TaskSet):
    """Tasks for testing the prediction API endpoints"""
    
    @task(10)
    @tag('prediction', 'single')
    def predict_single_tweet(self):
        """Test single tweet prediction - most common use case"""
        tweet = random.choice(SAMPLE_TWEETS)
        payload = {"text": tweet}
        
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="/predict [normal]"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "prediction" in data and "confidence" in data:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(5)
    @tag('prediction', 'batch')
    def batch_predict_small(self):
        """Test small batch predictions (5 tweets)"""
        tweets = random.sample(SAMPLE_TWEETS, min(5, len(SAMPLE_TWEETS)))
        payload = {"texts": tweets}
        
        with self.client.post(
            "/batch_predict",
            json=payload,
            catch_response=True,
            name="/batch_predict [5 tweets]"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "results" in data and len(data["results"]) == len(tweets):
                        response.success()
                    else:
                        response.failure("Invalid batch response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    @tag('prediction', 'batch', 'stress')
    def batch_predict_medium(self):
        """Test medium batch predictions (20 tweets)"""
        tweets = [random.choice(SAMPLE_TWEETS) for _ in range(20)]
        payload = {"texts": tweets}
        
        with self.client.post(
            "/batch_predict",
            json=payload,
            catch_response=True,
            name="/batch_predict [20 tweets]"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    @tag('prediction', 'batch', 'stress', 'redteam')
    def batch_predict_large(self):
        """Test large batch predictions (100 tweets) - potential DOS vector"""
        tweets = [random.choice(SAMPLE_TWEETS) for _ in range(100)]
        payload = {"texts": tweets}
        
        with self.client.post(
            "/batch_predict",
            json=payload,
            catch_response=True,
            name="/batch_predict [100 tweets]"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 413:
                response.success()  # Expected for large payloads
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    @tag('prediction', 'stress', 'redteam')
    def predict_long_text(self):
        """Test prediction with very long text - potential DOS vector"""
        payload = {"text": LONG_TWEET}
        
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="/predict [long text]"
        ) as response:
            if response.status_code in [200, 413, 400]:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(3)
    @tag('health', 'monitoring')
    def health_check(self):
        """Test health check endpoint"""
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    @tag('models', 'monitoring')
    def get_models(self):
        """Test models listing endpoint"""
        with self.client.get(
            "/models",
            catch_response=True,
            name="/models [GET]"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    @tag('prediction', 'edge', 'redteam')
    def predict_empty_text(self):
        """Test edge case: empty text"""
        payload = {"text": ""}
        
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="/predict [empty]"
        ) as response:
            if response.status_code in [400, 200]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")
    
    @task(1)
    @tag('prediction', 'edge', 'redteam')
    def predict_invalid_payload(self):
        """Test edge case: invalid payload format"""
        payload = {"invalid_key": "test"}
        
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="/predict [invalid]"
        ) as response:
            if response.status_code in [400, 500]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")
    
    @task(1)
    @tag('prediction', 'stress', 'redteam')
    def predict_maximum_length(self):
        """Test prediction with maximum length text - DOS attack vector"""
        payload = {"text": VERY_LONG_TWEET}
        
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="/predict [max length]"
        ) as response:
            if response.status_code in [200, 413, 400, 500]:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


class AuthTasks(TaskSet):
    """Tasks for testing authentication API endpoints"""
    
    def on_start(self):
        """Initialize with random user credentials"""
        self.username = f"testuser_{random.randint(1000, 9999)}"
        self.password = f"testpass_{random.randint(1000, 9999)}"
        self.registered = False
    
    @task(3)
    @tag('auth', 'register')
    def register_user(self):
        """Test user registration"""
        if not self.registered:
            payload = {
                "username": self.username,
                "password": self.password
            }
            
            with self.client.post(
                "/register",
                json=payload,
                catch_response=True,
                name="/register"
            ) as response:
                if response.status_code in [200, 400]:  # 400 if already exists
                    response.success()
                    self.registered = True
                else:
                    response.failure(f"Status code: {response.status_code}")
    
    @task(5)
    @tag('auth', 'login')
    def login_user(self):
        """Test user login"""
        if self.registered:
            payload = {
                "username": self.username,
                "password": self.password
            }
            
            with self.client.post(
                "/login",
                json=payload,
                catch_response=True,
                name="/login [valid]"
            ) as response:
                if response.status_code in [200, 401]:
                    response.success()
                else:
                    response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    @tag('auth', 'login', 'redteam')
    def login_invalid_credentials(self):
        """Test login with invalid credentials"""
        payload = {
            "username": "invalid_user",
            "password": "wrong_password"
        }
        
        with self.client.post(
            "/login",
            json=payload,
            catch_response=True,
            name="/login [invalid]"
        ) as response:
            if response.status_code == 401:
                response.success()
            else:
                response.failure(f"Expected 401, got: {response.status_code}")
    
    @task(1)
    @tag('auth', 'monitoring')
    def check_status(self):
        """Test status check endpoint"""
        with self.client.get(
            "/status",
            catch_response=True,
            name="/status"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    @tag('auth', 'models')
    def get_models_api(self):
        """Test models API endpoint (requires auth)"""
        with self.client.get(
            "/api/models",
            catch_response=True,
            name="/api/models"
        ) as response:
            if response.status_code in [200, 401]:  # 401 if not authenticated
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    @tag('auth', 'stress', 'redteam')
    def rapid_registration_attempts(self):
        """Test rapid registration attempts - potential DOS vector"""
        random_username = f"user_{random.randint(10000, 99999)}"
        payload = {
            "username": random_username,
            "password": "password123"
        }
        
        with self.client.post(
            "/register",
            json=payload,
            catch_response=True,
            name="/register [rapid]"
        ) as response:
            if response.status_code in [200, 400, 500]:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    @tag('auth', 'edge', 'redteam')
    def sql_injection_attempt(self):
        """Test SQL injection vulnerability"""
        payload = {
            "username": "admin' OR '1'='1",
            "password": "password' OR '1'='1"
        }
        
        with self.client.post(
            "/login",
            json=payload,
            catch_response=True,
            name="/login [sql injection]"
        ) as response:
            if response.status_code in [400, 401]:
                response.success()
            elif response.status_code == 200:
                response.failure("SECURITY VULNERABILITY: SQL injection successful!")
            else:
                response.failure(f"Status code: {response.status_code}")


class PredictionAPIUser(HttpUser):
    """User that tests the prediction API (port 5000)"""
    tasks = [PredictionTasks]
    wait_time = between(1, 3)
    host = "http://18.208.150.135:5000"


class AuthAPIUser(HttpUser):
    """User that tests the authentication API (port 5001)"""
    tasks = [AuthTasks]
    wait_time = between(1, 3)
    host = "http://18.208.150.135:5001"


# For mixed workload testing
class MixedWorkloadUser(HttpUser):
    """User that tests both APIs with mixed workload"""
    wait_time = between(1, 2)
    
    def on_start(self):
        """Initialize user"""
        self.prediction_host = "http://18.208.150.135:5000"
        self.auth_host = "http://18.208.150.135:5001"
    
    @task(7)
    @tag('mixed', 'prediction')
    def test_prediction_api(self):
        """Test prediction API"""
        tweet = random.choice(SAMPLE_TWEETS)
        payload = {"text": tweet}
        
        with self.client.post(
            f"{self.prediction_host}/predict",
            json=payload,
            catch_response=True,
            name="Mixed: /predict"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(3)
    @tag('mixed', 'auth')
    def test_auth_api(self):
        """Test auth API"""
        with self.client.get(
            f"{self.auth_host}/status",
            catch_response=True,
            name="Mixed: /status"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

