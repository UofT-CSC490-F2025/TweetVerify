import boto3
import os


def safe_download(s3, bucket, key, local_path):
    """Download only if file does NOT exist."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.exists(local_path):
        print(f"[Skip] {local_path} already exists.")
        return

    print(f"[Download] Downloading {key} -> {local_path}")
    s3.download_file(bucket, key, local_path)


def download_dataset():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )

    bucket = "datasettweet"

    files = [
        ("ai_token.csv", "datasets/ai_token.csv"),
        ("human_token.csv", "datasets/human_token.csv"),
        ("w2vmodel.model", "datasets/w2vmodel.model"),
        ("ai_token_with_features.csv","datasets/ai_token_with_features.csv"),
        ("human_token_with_features.csv","datasets/human_token_with_features.csv"),
    ]

    for key, local_path in files:
        safe_download(s3, bucket, key, local_path)


def download_model():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )

    bucket = "datasettweet"

    model_files = [
        ("roberta_89.7_2025-11-26_23-49-45.pt", "model_save/roberta_89.7_2025-11-26_23-49-45.pt"),
        ("lstm_67.9_2025-11-13_15-22-41.pt", "model_save/lstm_67.9_2025-11-13_15-22-41.pt"),
        ("rnn_67.5_2025-11-13_15-21-26.pt", "model_save/rnn_67.5_2025-11-13_15-21-26.pt"),
        ("bert_76.0_2025-11-13_15-35-26.pt", "model_save/bert_76.0_2025-11-13_15-35-26.pt"),
        ("deberta_76.7_2025-11-26_23-41-39.pt", "model_save/deberta_76.7_2025-11-26_23-41-39.pt"),
    ]

    for key, local_path in model_files:
        safe_download(s3, bucket, key, local_path)