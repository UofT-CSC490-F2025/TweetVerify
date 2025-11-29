# TweetVerify ![Code Coverage](./coverage.svg)

A deep learning-based Twitter tweet authenticity verification system supporting multiple model training and real-time prediction.

## Project Overview

TweetVerify is a comprehensive machine learning platform specifically designed for verifying the authenticity of Twitter tweets focused on political posts. The system supports multiple deep learning models, provides a modern web interface for model training, management, and prediction, and integrates with AWS SageMaker for cloud-based training.

### Core Values

- **Accuracy**: Support for multiple advanced deep learning models (RNN, LSTM, BERT)
- **Usability**: Intuitive, modern web interface requiring no programming knowledge
- **Scalability**: Support for cloud training and local deployment
- **Security**: Built-in rate limiting and input validation
- **Real-time**: Real-time log monitoring and model prediction

## Features

### Model Support
- **RNN (Recurrent Neural Network)**: Bidirectional RNN with Word2Vec embeddings (baseline model)
- **LSTM (Long Short-Term Memory)**: 2-layer bidirectional LSTM with dropout (baseline model)
- **BERT**: Pre-trained BERT-based classifier with fine-tuning support
- More SOTA models to be added...

### Data Ingestion
- Twitter tweet scraping via API
- LLM-generated content integration
- Automated data preprocessing and cleaning
- Data lake architecture with parquet storage

### Web Interface
- **Modern UI**: Redesigned responsive interface with glassmorphism aesthetics
- **Interactive Dashboard**: Real-time model performance monitoring
- **Prediction Interface**: Single text and batch upload support
- **Model Management**: Easy model switching, comparison, and file management
- **User System**: Secure login and registration

### Security
- **Rate Limiting**: Protection against API abuse
- **Input Validation**: Strict schema validation for all API endpoints
- **Secure Auth**: Password hashing and session management

### Cloud Integration
- **AWS SageMaker**: Distributed training support
- **Terraform**: Infrastructure as Code (IaC)
- **PostgreSQL**: Scalable database backend
- **Automated Deployment**: EC2 and RDS provisioning

## Technology Stack

### Machine Learning
- **PyTorch**: Deep learning framework
- **Transformers**: BERT model integration
- **Word2Vec (Gensim)**: Word embeddings
- **scikit-learn**: Evaluation metrics
- **pandas, numpy**: Data manipulation

### Web Framework
- **Flask**: Python web server
- **HTML5/CSS3**: Modern frontend with CSS variables
- **JavaScript**: Dynamic client-side interactions

### Data & Infrastructure
- **PostgreSQL**: Relational database
- **Terraform**: Infrastructure provisioning
- **AWS (SageMaker, EC2, S3, RDS)**: Cloud services
- **Tweepy**: Twitter API client

## Installation Guide

### Prerequisites
- Python 3.8+
- PostgreSQL 14+
- AWS Account (optional, for cloud features)
- Twitter API credentials

### Testing
Ensure all dependencies are installed and the python path is set:
```bash
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:.
```

Run the full test suite with coverage:
```bash
pytest --cov=src tests/
```

### Quick Start with Terraform

TweetVerify uses Terraform to automatically provision AWS infrastructure including EC2 instances, RDS database, and SageMaker endpoints.

1. **Clone the repository**
```bash
git clone https://github.com/UofT-CSC490-F2025/TweetVerify.git
cd TweetVerify
```

2. **Set up AWS credentials**
Configure your AWS credentials using one of the following methods:
```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

3. **Install Terraform**
```bash
# Linux/Mac
brew install terraform

# Or download from https://www.terraform.io/downloads
```

4. **Deploy infrastructure**
```bash
cd terraform
terraform init
terraform plan  # Review changes
terraform apply  # Deploy infrastructure
```

5. **Access web interface**
After deployment completes, you'll receive the EC2 instance IP in the Terraform outputs:
- **Prediction Interface**: `http://YOUR_EC2_IP:5000`
- **Admin Dashboard**: `http://YOUR_EC2_IP:5001`

6. **Tear down infrastructure** (when done)
```bash
terraform destroy
```

## Configuration

### Terraform Variables
Edit `terraform/variables.tf` to customize:
- **Instance types**: EC2 and RDS instance sizes
- **Security groups**: Firewall rules for SSH (port 22) and web apps (ports 5000, 5001)
- **Region**: AWS region for deployment (default: us-east-1)
- **Database credentials**: PostgreSQL username and password
- **Your IP CIDR**: Allowed IP for SSH access

### Model Configuration
Model training parameters can be adjusted in:
- `src/train.py`: Local training configuration
- `src/trainer/train_aws_sagemaker.py`: SageMaker training configuration

Key parameters:
- Batch size: `--batch_size` (default: 314)
- Learning rate: `--learning_rate` (default: 0.0001)
- Epochs: `--epochs` (default: 100)

## User Guide

### Using the Web Interface

1. **Login**: Access the login page (Port 5001) and authenticate
2. **Dashboard**: View model performance and statistics
3. **Train Models**: Configure and start training jobs
4. **Make Predictions**: Enter text to verify authenticity (Port 5000)
5. **Model Management**: View, compare, and select models

## Project Structure

```
TweetVerify/
├── src/
│   ├── apps/                # Web applications
│   │   ├── app.py           # Public prediction app
│   │   └── auth_app.py      # Admin dashboard app
│   ├── data_ingestion/      # Data collection and processing
│   │   ├── twitter_scrape.py
│   │   ├── twitter_db.py
│   │   └── llm_db.py
│   ├── data_preprocessing/  # Data cleaning and preparation
│   ├── dataloader/          # Dataset loaders
│   ├── model/               # Model architectures
│   │   ├── rnn.py
│   │   ├── lstm.py
│   │   ├── bert.py
│   │   └── roberta.py
│   ├── trainer/             # Training logic
│   │   ├── trainer.py
│   │   └── train_aws_sagemaker.py
│   ├── evaluator/           # Model evaluation
│   ├── inference/           # Prediction logic
│   ├── plotter/             # Visualization tools
│   ├── web/                 # Frontend templates
│   ├── utils/               # Utility functions
│   ├── security/            # Security modules (Rate limit, Validation)
│   ├── train.py             # Training script
│   └── train_model.py       # Training entry point
├── tests/                   # Unit tests
├── terraform/               # Infrastructure as code
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── model_save/              # Saved models
├── datalake/                # Data storage
│   ├── curated/
│   └── dataset/
├── datasets/                # Training datasets
├── requirements.txt
└── README.md
```

## AWS Integration

### Infrastructure Components

TweetVerify deploys the following AWS resources:
- **EC2 Instance**: Hosts the Flask web application
- **RDS PostgreSQL Database**: Stores training data and model metadata
- **Security Groups**: Manages network access (SSH on port 22, web apps on 5000/5001)
- **S3 Bucket**: Stores model artifacts and training datasets

### Terraform Workflow

The infrastructure is managed entirely through Terraform. See [Configuration](#configuration) section for customizing deployment parameters.