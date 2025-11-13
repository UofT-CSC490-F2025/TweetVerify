import boto3
import os


s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
)
bucket_name = "datasettweet"


def download_dataset():
    object_key = "ai_token.csv"
    local_path = "datasets/ai_token.csv"
    s3.download_file(bucket_name, object_key, local_path)
    object_key = "human_token.csv"
    local_path = "datasets/human_token.csv"
    s3.download_file(bucket_name, object_key, local_path)
    object_key = "w2vmodel.model"
    local_path = "datasets/w2vmodel.model"
    s3.download_file(bucket_name, object_key, local_path)


def download_model():
    object_key = "bert_99.2_2025-10-13_15-15-24.pt"
    local_path = "model_save/bert_99.2_2025-10-13_15-15-24.pt"
    s3.download_file(bucket_name, object_key, local_path)
