# TweetVerify 🐦

A deep learning-based Twitter tweet authenticity verification system supporting multiple model training and real-time prediction.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation Guide](#installation-guide)
- [Configuration](#configuration)
- [User Guide](#user-guide)
- [API Documentation](#api-documentation)
- [Deployment Guide](#deployment-guide)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

TweetVerify is a comprehensive machine learning platform specifically designed for verifying the authenticity of Twitter tweets. The system supports multiple deep learning models (RNN, LSTM, BERT), provides a web interface for model training, management, and prediction, and integrates with AWS SageMaker for cloud-based training.

### Core Values

- **Accuracy**: Support for multiple advanced deep learning models
- **Usability**: Intuitive web interface requiring no programming knowledge
- **Scalability**: Support for cloud training and local deployment
- **Real-time**: Real-time log monitoring and model prediction

## ✨ Features

### 🤖 Model Management

- **Multi-Model Support**: RNN, LSTM, and BERT deep learning models
- **Model Training**: Support for both local and AWS SageMaker cloud training
- **Model Storage**: Automatic saving of trained models to local storage
- **Model Upload**: Support for manual upload of pre-trained models
- **Model Deletion**: One-click deletion of unwanted models
- **Model Information**: Display detailed information including model type, accuracy, training time, etc.

### 🎛️ Training Management

- **Parameter Configuration**: Adjustable training epochs, learning rate, batch size
- **Real-time Monitoring**: Real-time display of training logs and progress
- **Log Filtering**: Automatic filtering of HTTP request logs, showing only training-related information
- **Status Tracking**: Real-time tracking of training status (starting, running, completed, failed)
- **Auto Refresh**: Training logs automatically refresh every 2 seconds
- **Cloud Training**: Integration with AWS SageMaker for cloud GPU training

### 🔍 Prediction Features

- **Real-time Prediction**: Input tweet content for real-time authenticity prediction
- **Confidence Display**: Show confidence level of prediction results
- **Batch Prediction**: Support for batch tweet prediction
- **History Records**: Save prediction history records

### 👤 User Management

- **User Registration**: Support for new user registration
- **User Login**: Secure user authentication system
- **Session Management**: Automatic session management and timeout handling
- **Access Control**: User identity-based access control

### 📊 Data Management

- **Dataset Support**: Support for multiple formats of training data
- **Data Preprocessing**: Automatic data cleaning and preprocessing
- **Data Visualization**: Training process data visualization
- **Data Export**: Support for prediction result export

## 🛠️ Technology Stack

### Backend Technologies

- **Python 3.8+**: Primary programming language
- **Flask**: Web framework
- **PyTorch**: Deep learning framework
- **Transformers**: BERT model support
- **PostgreSQL**: Database
- **psycopg2**: Database connector
- **Werkzeug**: Password encryption and security

### Frontend Technologies

- **HTML5**: Page structure
- **CSS3**: Styling design
- **JavaScript (ES6+)**: Interactive logic
- **Responsive Design**: Support for mobile and desktop

### Machine Learning

- **RNN**: Recurrent Neural Network
- **LSTM**: Long Short-Term Memory Network
- **BERT**: Bidirectional Encoder Representations from Transformers
- **scikit-learn**: Machine learning tools
- **pandas**: Data processing
- **numpy**: Numerical computation

### Cloud Services

- **AWS SageMaker**: Cloud model training
- **AWS S3**: Model storage
- **AWS IAM**: Identity and Access Management
- **boto3**: AWS SDK

### Development Tools

- **Git**: Version control
- **Docker**: Containerized deployment
- **Terraform**: Infrastructure as Code
- **pytest**: Unit testing

## 📦 Installation Guide

### System Requirements

- Python 3.8 or higher
- PostgreSQL 12 or higher
- At least 4GB RAM
- At least 10GB available disk space
- Network connection (for AWS services)

### 1. Clone Repository

```bash
git clone https://github.com/your-username/TweetVerify.git
cd TweetVerify
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Database Setup

```bash
# Create PostgreSQL database
createdb tweetverify

# Set environment variables
export DB_HOST=localhost
export DB_NAME=tweetverify
export DB_USER=your_username
export DB_PASS=your_password
```

### 5. AWS Configuration (Optional)

```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-2
export AWS_ROLE_ARN=your_sagemaker_role_arn
```

### 6. Start Application

```bash
python src/auth_app.py
```

The application will start at `http://localhost:5001`.

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DB_HOST` | Database host | `localhost` | Yes |
| `DB_NAME` | Database name | `tweetverify` | Yes |
| `DB_USER` | Database username | `postgres` | Yes |
| `DB_PASS` | Database password | - | Yes |
| `AWS_ACCESS_KEY_ID` | AWS access key | - | No* |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | - | No* |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-2` | No* |
| `AWS_ROLE_ARN` | SageMaker role ARN | - | No* |

*Required only when using AWS SageMaker training

### Configuration Files

#### `requirements.txt`
Contains all Python dependency packages and their versions.

#### `src/config.py`
Application configuration file containing default settings and constants.

## 📖 User Guide

### 1. User Registration and Login

1. Visit `http://localhost:5001`
2. Click "Register" to create a new account
3. Login with username and password

### 2. Model Training

#### Local Training

1. After login, go to the "Training" page
2. Select model type (RNN, LSTM, or BERT)
3. Set training parameters:
   - **Epochs**: Recommended 100-500 epochs
   - **Learning Rate**: Recommended 0.0001-0.001
   - **Batch Size**: Recommended 32-512
4. Click "Start Training"
5. Monitor training progress in the log area

#### AWS SageMaker Training

1. Ensure AWS credentials are configured
2. Select "AWS SageMaker Training"
3. Set training parameters
4. Click "Start AWS Training"
5. Training will occur in the cloud, model automatically downloaded upon completion

### 3. Model Management

#### View Models

1. Go to the "Models" page
2. View all available models
3. Model information includes:
   - Model type
   - Accuracy
   - Training time
   - File size

#### Upload Model

1. Click "Upload Model" button
2. Select model file (.pt, .pth, .pkl formats)
3. System automatically parses model information

#### Delete Model

1. Find the model to delete in the model list
2. Click "Delete" button
3. Confirm deletion operation

### 4. Tweet Prediction

1. Go to the "Prediction" page
2. Select the model to use
3. Input tweet content
4. Click "Predict"
5. View prediction results and confidence

### 5. Real-time Log Monitoring

- **Auto Refresh**: Logs automatically update every 2 seconds
- **HTTP Filtering**: Automatically filters out HTTP request logs
- **Smart Classification**: Automatic coloring based on log content
- **Manual Control**: Support for manual refresh and clear

## 🔌 API Documentation

### Authentication

All API requests require user login through session authentication.

### User Management API

#### Register User
```http
POST /register
Content-Type: application/json

{
  "username": "string",
  "password": "string"
}
```

#### User Login
```http
POST /login
Content-Type: application/json

{
  "username": "string",
  "password": "string"
}
```

#### Check Login Status
```http
GET /status
```

#### User Logout
```http
GET /logout
```

### Model Management API

#### Get Model List
```http
GET /api/models
```

#### Delete Model
```http
POST /api/models/delete
Content-Type: application/json

{
  "model_path": "string"
}
```

#### Upload Model
```http
POST /api/models/upload
Content-Type: multipart/form-data

file: [model file]
```

### Training Management API

#### Start Training
```http
POST /api/training/start
Content-Type: application/json

{
  "model_type": "rnn|lstm|bert",
  "epochs": 100,
  "learning_rate": 0.0001,
  "batch_size": 314
}
```

#### Get Training Status
```http
GET /api/training/status/<training_id>
```

#### Stop Training
```http
POST /api/training/stop/<training_id>
```

#### Get Training List
```http
GET /api/training/list
```

#### Get Training Logs
```http
GET /api/training/logs/live/<training_id>
```

### Prediction API

#### Single Tweet Prediction
```http
POST /api/predict
Content-Type: application/json

{
  "text": "string",
  "model_path": "string"
}
```

#### Batch Prediction
```http
POST /api/predict/batch
Content-Type: application/json

{
  "texts": ["string1", "string2", ...],
  "model_path": "string"
}
```

## 🚀 Deployment Guide

### Docker Deployment

#### 1. Build Image

```bash
docker build -t tweetverify .
```

#### 2. Run Container

```bash
docker run -d \
  --name tweetverify \
  -p 5001:5001 \
  -e DB_HOST=your_db_host \
  -e DB_NAME=tweetverify \
  -e DB_USER=your_user \
  -e DB_PASS=your_password \
  tweetverify
```

### Production Environment Deployment

#### 1. Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 src.auth_app:app
```

#### 2. Using Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### AWS Deployment

#### 1. EC2 Deployment

1. Launch EC2 instance
2. Install dependencies
3. Configure environment variables
4. Start application

#### 2. ECS Deployment

1. Create Docker image
2. Push to ECR
3. Create ECS task definition
4. Deploy service

## 🔧 Troubleshooting

### Common Issues

#### 1. Database Connection Failed

**Symptoms**: Database connection error when starting application

**Solutions**:
- Check if PostgreSQL service is running
- Verify database connection parameters
- Confirm database user permissions

#### 2. AWS SageMaker Training Failed

**Symptoms**: Cloud training returns error

**Solutions**:
- Check AWS credential configuration
- Verify IAM role permissions
- Confirm SageMaker service availability

#### 3. Model Loading Failed

**Symptoms**: Cannot load model during prediction

**Solutions**:
- Check model file integrity
- Verify model format compatibility
- Confirm file permissions

#### 4. Insufficient Memory

**Symptoms**: Memory error during training process

**Solutions**:
- Reduce batch size
- Use smaller models
- Increase system memory

### Log Debugging

#### View Application Logs

```bash
tail -f logs/training_*.log
```

#### View Error Logs

```bash
grep -i error logs/training_*.log
```

#### Debug Mode

```bash
export FLASK_DEBUG=1
python src/auth_app.py
```

### Performance Optimization

#### 1. Database Optimization

- Create appropriate indexes
- Regularly clean old data
- Use connection pooling

#### 2. Model Optimization

- Use model quantization
- Implement model caching
- Batch process predictions

#### 3. Frontend Optimization

- Enable Gzip compression
- Use CDN
- Implement lazy loading

## 🤝 Contributing

### Development Environment Setup

1. Fork the project repository
2. Clone your fork
3. Create feature branch
4. Install development dependencies

```bash
pip install -r requirements-dev.txt
```

### Code Standards

- Use PEP 8 code style
- Write unit tests
- Add docstrings
- Run tests before committing

### Commit Standards

Use semantic commit messages:

```
feat: add new feature
fix: fix bug
docs: update documentation
style: code formatting
refactor: code refactoring
test: add tests
chore: build process or auxiliary tool changes
```

### Testing

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/test_models.py

# Generate coverage report
pytest --cov=src
```

### Pull Request

1. Ensure all tests pass
2. Update relevant documentation
3. Submit Pull Request
4. Wait for code review

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📞 Support & Contact

### Get Help

- 📧 Email: support@tweetverify.com
- 💬 Discussions: [GitHub Discussions](https://github.com/your-username/TweetVerify/discussions)
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/TweetVerify/issues)

### Community

- 🌟 Star the project
- 🍴 Fork the project
- 📢 Share with others

## 🙏 Acknowledgments

Thanks to the following open source projects and services:

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [Transformers](https://huggingface.co/transformers/) - BERT models
- [AWS SageMaker](https://aws.amazon.com/sagemaker/) - Cloud training service
- [PostgreSQL](https://www.postgresql.org/) - Database

---

**TweetVerify** - Making tweet verification simpler and more accurate! 🐦✨