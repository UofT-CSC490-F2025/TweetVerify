# TweetVerify ![Code Coverage](./coverage.svg)

A deep learning-based Twitter tweet authenticity verification system supporting multiple model training and real-time prediction.

## Project Overview

TweetVerify is a comprehensive machine learning platform specifically designed for verifying the authenticity of Twitter tweets focused on political posts. The system supports multiple deep learning models, provides a modern web interface for model training, management, and prediction, and integrates with AWS SageMaker for cloud-based training.

## Project Structure

```
TweetVerify/
├── src/                                    # Main source code directory
│   ├── apps/                               # Web application entry points
│   │   ├── app.py                          # Public prediction interface (Flask app on port 5000)
│   │   └── auth_app.py                     # Admin dashboard with authentication (Flask app on port 5001)
│   │
│   ├── data_ingestion/                     # Data collection and ingestion modules
│   │   ├── twitter_scrape.py               # Twitter API scraping functionality
│   │   ├── twitter_db.py                   # Database operations for Twitter data
│   │   ├── llm_db.py                       # Database operations for LLM-generated data
│   │   ├── llm_generate.py                 # LLM content generation scripts
│   │   ├── llm_synthesis.py                # LLM content synthesis utilities
│   │   └── main_db.py                      # Main database connection and operations
│   │
│   ├── data_preprocessing/                  # Data cleaning and preprocessing scripts
│   │   ├── processor.py                    # Main data processing pipeline
│   │   ├── cleaning_political_tweets.py   # Political tweet cleaning utilities
│   │   ├── filter_good_tweets.py           # Quality filtering for tweets using OpenAI batch API
│   │   ├── filter_tweets_through_post.py   # Async tweet filtering via OpenAI API
│   │   ├── creating_ai_tweets.py          # AI tweet variant generation
│   │   ├── reformat_human_response_high_quality.py  # Human tweet data reformatting
│   │   ├── reformat_and_filter_ai_response.py       # AI response filtering and reformatting
│   │   ├── cancel_batch.py                 # Batch job cancellation utilities
│   │   └── testing_cache.py                # Cache testing utilities
│   │
│   ├── dataloader/                         # PyTorch dataset loaders
│   │   ├── bertdataset.py                  # Dataset class for BERT/RoBERTa/DeBERTa models
│   │   └── featuredataset.py               # Dataset class with extra handcrafted features
│   │
│   ├── model/                              # Neural network model architectures
│   │   ├── rnn.py                          # Bidirectional RNN with Word2Vec embeddings
│   │   ├── lstm.py                         # 2-layer bidirectional LSTM with dropout
│   │   ├── bert.py                         # BERT-based binary classifier
│   │   ├── roberta.py                      # RoBERTa-based binary classifier
│   │   ├── roberta_extra.py               # RoBERTa with handcrafted features (perplexity, caps ratio, etc.)
│   │   ├── deberta.py                      # DeBERTa-v3-based binary classifier
│   │   ├── train_judge_qwen.py            # Qwen2.5 model training with LoRA and GRPO
│   │   ├── prompt_only_llm.py              # Prompt-only LLM classification (Qwen2.5-7B)
│   │   └── test.py                         # Model testing utilities
│   │
│   ├── trainer/                            # Model training modules
│   │   ├── trainer.py                      # Core training loop and optimization logic
│   │   ├── train_aws_sagemaker.py          # AWS SageMaker training integration
│   │   └── aws_training_manager.py         # AWS training job management
│   │
│   ├── evaluator/                          # Model evaluation and metrics
│   │   └── evaluator.py                    # Accuracy, F1, and AUC-ROC computation
│   │
│   ├── inference/                          # Prediction and inference logic
│   │   └── predictor.py                   # Model inference and prediction interface
│   │
│   ├── plotter/                            # Visualization and plotting utilities
│   │   └── plotter.py                      # Model performance visualization tools
│   │
│   ├── web/                                # Frontend templates and static files
│   │   └── templates/                      # HTML templates for web interface
│   │       ├── index.html                  # Public prediction page
│   │       ├── login.html                  # Admin login page
│   │       ├── dashboard.html              # Admin dashboard
│   │       ├── models.html                 # Model management page
│   │       └── training.html               # Training configuration page
│   │
│   ├── utils/                              # Utility functions and helpers
│   │   ├── collate_batch.py                # Custom collate function for RNN/LSTM batching
│   │   ├── convert_indices.py             # Word2Vec index conversion utilities
│   │   ├── extract_features.py            # Handcrafted feature extraction (perplexity, etc.)
│   │   ├── seed.py                         # Random seed setting for reproducibility
│   │   ├── canonical_id.py                # Canonical ID generation utilities
│   │   ├── benchmarking.py                # Model benchmarking and performance testing
│   │   └── get_from_s3.py                 # S3 data retrieval utilities
│   │
│   ├── security/                           # Security and validation modules
│   │   ├── __init__.py                     # Security package initialization
│   │   ├── rate_limiter.py                 # API rate limiting implementation
│   │   └── input_validator.py              # Input validation and sanitization
│   │
│   ├── outputs/                            # Generated outputs and predictions
│   │
│   ├── train.py                            # Main training script entry point
│   ├── run.py                              # Application runner script
│   └── app_wrapper.py                      # Application wrapper utilities
│
├── tests/                                  # Unit and integration tests
│   ├── test_app.py                         # Web application tests
│   ├── test_models.py                      # Model architecture tests
│   ├── test_train.py                       # Training pipeline tests
│   ├── test_trainer_evaluator.py          # Trainer and evaluator tests
│   ├── test_data_modules.py                # Data loading and preprocessing tests
│   ├── test_inference_plotter.py           # Inference and plotting tests
│   ├── test_input_validator.py             # Input validation tests
│   ├── test_rate_limiter.py                # Rate limiting tests
│   ├── test_scripts.py                     # Script execution tests
│   └── test_utils.py                       # Utility function tests
│
├── terraform/                             # Infrastructure as Code (IaC)
│   ├── main.tf                             # Main Terraform configuration
│   ├── variables.tf                        # Terraform variable definitions
│   └── outputs.tf                          # Terraform output definitions
│
├── .github/                                 # GitHub configuration
│   └── workflows/                          # GitHub Actions workflows
│       └── coverage.yml                    # Code coverage CI/CD workflow
│
├── .coveragerc                             # Coverage.py configuration
├── .gitignore                              # Git ignore patterns
├── requirements.txt                        # Python dependencies
├── coverage.svg                            # Code coverage badge
└── README.md                               # Project documentation
```

### Key Directories Explained

#### `src/apps/`
Contains the two main Flask applications:
- **`app.py`**: Public-facing prediction interface (port 5000) for end users to verify tweets
- **`auth_app.py`**: Admin dashboard (port 5001) with authentication for model management and training

#### `src/data_ingestion/`
Handles data collection from multiple sources:
- Twitter API scraping and database operations
- LLM-generated content integration and synthesis
- Database connection management

#### `src/data_preprocessing/`
Data cleaning and quality control scripts:
- Tweet cleaning and filtering (removes spam, low-quality content)
- AI tweet variant generation for training data augmentation
- Data reformatting for different model requirements

#### `src/model/`
Neural network architectures:
- **Baseline models**: RNN, LSTM (with Word2Vec embeddings)
- **Transformer models**: BERT, RoBERTa, DeBERTa-v3
- **Advanced models**: RoBERTa with handcrafted features, Qwen2.5 with LoRA/GRPO

#### `src/trainer/`
Training infrastructures:
- Local training with PyTorch
- AWS SageMaker integration for distributed cloud training
- Training job management and monitoring

#### `src/utils/`
Supporting utilities:
- Data batching and collation
- Feature extraction (perplexity, capitalization ratio, etc.)
- S3 integration for cloud storage
- Reproducibility tools (seed setting)

#### `src/security/`
Security and validation:
- Rate limiting to prevent API abuse
- Input validation and sanitization
- Secure authentication mechanisms

#### `tests/`
Comprehensive test suite covering:
- Model architectures and training pipelines
- Web application functionality
- Data processing and validation
- Security features

## Installation Guide

### Prerequisites
- Python 3.8 - 3.13
- PostgreSQL 14+
- AWS Account (optional, for cloud features)
- Twitter API credentials

### Data & Models Setup
Due to file size limits, the trained models and datasets are hosted externally. Please download them before running the project:

1. Download the `models_and_datasets` folder from [Google Drive](https://drive.google.com/file/d/1h3byaeFWFJWLdNWiP4S2LugwPwTiTNr8/view?usp=sharing).
2. Extract the contents.
3. Place the contents into the project root directory so that you have:
   - `datasets/` (containing `w2vmodel.model`, `*.csv`)
   - `model_save/` (containing `*.pt` checkpoints)

Alternatively, you can modify the paths in the scripts or arguments to point to your download location.

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

### Benchmarking & Evaluation
To evaluate trained models and reproduce performance metrics across multiple seeds:

```bash
# Single model evaluation
python -m src.utils.benchmarking --model bert --model_dir path/to/checkpoints

# Ensemble evaluation
python -m src.utils.benchmarking --model ensemble --model_dir path/to/checkpoints
```

Supported models: `rnn`, `lstm`, `bert`, `roberta`, `deberta`, `roberta_extra`, `ensemble`.

## Running the Application

You can run TweetVerify either locally for development or deploy it to the cloud using Terraform.

### Option 1: Local Execution

**Prerequisites for Local Run:**
1.  **PostgreSQL Database**: You must have a local PostgreSQL database running for the Admin Dashboard.
2.  **Environment Variables**: Set the following variables before running:
    ```bash
    export DB_HOST=localhost
    export DB_NAME=your_db_name
    export DB_USER=your_username
    export DB_PASS=your_password
    ```
3.  **Port 5000 (macOS Users)**: AirPlay Receiver often uses port 5000. Turn it off in *System Settings > General > AirDrop & Handoff > AirPlay Receiver*.

**1. Start the Web Interface**
To launch both the Prediction Interface (Port 5000) and the Admin Dashboard (Port 5001):

```bash
python -m src.app_wrapper
```

*   **Prediction App**: `http://localhost:5000`
*   **Admin Dashboard**: `http://localhost:5001`

**2. Run Only Prediction Interface (No DB Required)**
If you only want to test the prediction model without setting up a database:

```bash
python -m src.apps.app
```

**3. Run Local Training**
To train models locally without the web interface:

```bash
# Run with default settings
python src/run.py

# Or with specific arguments
python -m src.train --model bert --epochs 5
```

### Option 2: Cloud Deployment (Terraform)

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

## Features

### Model Support
- **Baseline Models**: Bidirectional RNN and LSTM initialized with Word2Vec embeddings
- **Transformers**: BERT, RoBERTa, and DeBERTa-v3 classifiers with full fine-tuning
- **Hybrid Models**: RoBERTa augmented with handcrafted linguistic features (perplexity, styling metrics)
- **LLMs**: Experimental integration with Qwen2.5 (7B/14B) using parameter-efficient fine-tuning (LoRA/GRPO)

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
