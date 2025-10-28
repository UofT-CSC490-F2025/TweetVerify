# TweetVerify 🐦

A deep learning-based Twitter tweet authenticity verification system supporting multiple model training and real-time prediction.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Technology Stack](#️-technology-stack)
- [Installation Guide](#-installation-guide)
- [Configuration](#-configuration)
- [User Guide](#-user-guide)
- [Project Structure](#-project-structure)
- [AWS Integration](#️-aws-integration)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Project Overview

TweetVerify is a comprehensive machine learning platform specifically designed for verifying the authenticity of Twitter tweets focused on political posts. The system supports multiple deep learning models, provides a web interface for model training, management, and prediction, and integrates with AWS SageMaker for cloud-based training.

### Core Values

- **Accuracy**: Support for multiple advanced deep learning models
- **Usability**: Intuitive web interface requiring no programming knowledge
- **Scalability**: Support for cloud training and local deployment
- **Real-time**: Real-time log monitoring and model prediction

## ✨ Features

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
- Interactive dashboard for model management
- Real-time training progress monitoring
- Model comparison and selection
- Live tweet prediction interface
- User authentication system

### Cloud Integration
- AWS SageMaker integration for distributed training
- Terraform infrastructure as code
- PostgreSQL database support
- EC2 and RDS deployment automation

## 🛠️ Technology Stack

### Machine Learning
- **PyTorch**: Deep learning framework
- **Transformers**: BERT model support
- **Word2Vec (Gensim)**: Word embeddings for RNN/LSTM
- **scikit-learn**: Model evaluation and data splitting
- **pandas, numpy**: Data processing

### Web Framework
- **Flask**: Backend web framework
- **HTML/CSS/JavaScript**: Frontend interface

### Data & Database
- **PostgreSQL**: Primary database
- **PyArrow/Parquet**: Efficient data storage
- **Tweepy**: Twitter API integration

### Cloud & Infrastructure
- **AWS SageMaker**: Cloud training platform
- **boto3**: AWS SDK for Python
- **Terraform**: Infrastructure as code

## 📦 Installation Guide

### Prerequisites
- Python 3.8+
- PostgreSQL 17.4+
- AWS Account (for cloud training)
- Twitter API credentials

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

# Option 3: AWS credentials file
mkdir -p ~/.aws
# Edit ~/.aws/credentials with your credentials
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
## ⚙️ Configuration

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
- `src/train_aws_sagemaker.py`: SageMaker training configuration

Key parameters:
- Batch size: `--batch_size` (default: 314)
- Learning rate: `--learning_rate` (default: 0.0001)
- Epochs: `--epochs` (default: 100)

## 🚀 User Guide

### Using the Web Interface

1. **Login**: Access the login page and authenticate
2. **Dashboard**: View model performance and statistics
3. **Train Models**: Configure and start training jobs
4. **Make Predictions**: Enter text to verify authenticity
5. **Model Management**: View, compare, and select models


## 📂 Project Structure

```
TweetVerify/
├── src/
│   ├── data_ingestion/     # Data collection and processing
│   │   ├── twitter_scrape.py
│   │   ├── twitter_db.py
│   │   └── llm_db.py
│   ├── data_preprocessing/  # Data cleaning and preparation
│   ├── model/               # Model architectures
│   │   ├── rnn.py
│   │   ├── lstm.py
│   │   └── bert.py
│   ├── trainer/             # Training logic
│   ├── evaluator/           # Model evaluation
│   ├── inference/           # Prediction logic
│   ├── web/                 # Frontend templates
│   ├── utils/               # Utility functions
│   ├── train.py             # Training script
│   ├── app.py               # Flask application
│   └── train_aws_sagemaker.py
├── terraform/               # Infrastructure as code
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── model_save/              # Saved models
├── datalake/                # Data storage
│   ├── curated/
│   └── processed/
├── datasets/                # Training datasets
├── requirements.txt
└── README.md
```

## ☁️ AWS Integration

### Infrastructure Components

TweetVerify deploys the following AWS resources:
- **EC2 Instance**: Hosts the Flask web application
- **RDS PostgreSQL Database**: Stores training data and model metadata
- **Security Groups**: Manages network access (SSH on port 22, web apps on 5000/5001)
- **S3 Bucket**: Stores model artifacts and training datasets

### Terraform Workflow

The infrastructure is managed entirely through Terraform. See [Configuration](#-configuration) section for customizing deployment parameters.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

**Course**: CSC490 - Engineering Capstone  
**Institution**: University of Toronto  
**Year**: 2025