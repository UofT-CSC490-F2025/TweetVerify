#!/usr/bin/env python3
"""
AWS SageMaker training script for TweetVerify models
(Modified to log all outputs — print() + logging — into one file)
"""

import os
import sys
import json
import time
import tarfile
import tempfile
import shutil
import logging
from pathlib import Path
import boto3
import sagemaker
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.session import Session
from datetime import datetime
import uuid

class TeeOutput:
    def __init__(self, file_path):
        """Redirect stdout/stderr so that all prints go to both terminal and file."""
        self.terminal = sys.__stdout__ 
        self.log_file = open(file_path, 'a', encoding='utf-8')
        self.closed = False

    def write(self, message):
        if not self.closed:
            if self.terminal:
                self.terminal.write(message)
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        if not self.closed:
            if self.terminal:
                self.terminal.flush()
            self.log_file.flush()

    def close(self):
        if not self.closed:
            self.log_file.close()
            self.closed = True

def setup_logging(log_file_path):
    log_dir = Path(log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    tee = TeeOutput(log_file_path)
    sys.stdout = tee
    sys.stderr = tee
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)  
        ]
    )

    logging.info("Logging initialized. All output will go to console and file.")
    return tee


# -------------------------------------------------------------
# AWS SageMaker Training Manager
# -------------------------------------------------------------
class AWSTrainingManager:
    """Manages AWS SageMaker training jobs"""

    def __init__(self, log_file_path=None):
        if log_file_path is None:
            log_file_path = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Initialize unified logging
        self.tee = setup_logging(log_file_path)
        self.log_file_path = log_file_path

        # AWS Configuration
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region_name = os.getenv("AWS_DEFAULT_REGION", "us-east-2")
        self.role_arn = os.getenv("AWS_ROLE_ARN")

        self.boto_session = boto3.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name,
        )

        self.sagemaker_session = Session(boto_session=self.boto_session)
        self.s3_client = self.boto_session.client("s3")
        self.sagemaker_client = self.boto_session.client("sagemaker")

        # Track active training jobs
        self.active_jobs = {}

        logging.info("AWS Training Manager initialized")

    # ---------------------------------------------------------
    # Start a new training job
    # ---------------------------------------------------------
    def start_training_job(self, model_type, epochs=100, learning_rate=0.0001, batch_size=314):
        try:
            job_name = f"tweetverify-{model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"
            logging.info(f"Starting SageMaker training job: {job_name}")
            estimator = PyTorch(
                source_dir=".",
                entry_point="src/run.py",
                role=self.role_arn,
                framework_version="2.2.0",
                py_version="py310",
                instance_count=1,
                instance_type="ml.g4dn.xlarge",
                sagemaker_session=self.sagemaker_session,
                requirements_file="requirements.txt",
                hyperparameters={
                    "model": str(model_type),
                    "epochs": str(epochs),
                    "learning_rate": str(learning_rate),
                    "batch_size": str(batch_size),
                },
                output_path=f"s3://sagemaker-{self.region_name}-993399330675/tweetverify-models/",
                job_name=job_name,
            )

            estimator.fit()

            self.active_jobs[job_name] = {
                "job_name": job_name,
                "model_type": model_type,
                "status": "completed",
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
            }

            msg = f"Training job {job_name} completed successfully"
            logging.info(msg)
            return {"success": True, "job_name": job_name, "message": msg}

        except Exception as e:
            error_msg = f"❌ Failed to start SageMaker training job: {str(e)}"
            logging.error(error_msg)
            return {"success": False, "error": error_msg}

    # ---------------------------------------------------------
    # Cleanup resources
    # ---------------------------------------------------------
    def cleanup(self):
        if hasattr(self, "tee"):
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            self.tee.close()
        print("AWS Training Manager resources cleaned up")


# -------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Start AWS SageMaker training job")
    parser.add_argument("model_type", choices=["rnn", "lstm", "bert"], help="Model type")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=314, help="Batch size")
    parser.add_argument("--log-file", type=str, help="Custom log file path")

    args = parser.parse_args()

    manager = AWSTrainingManager(log_file_path=args.log_file)

    try:
        result = manager.start_training_job(
            model_type=args.model_type,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )

        if result["success"]:
            sys.exit(0)
        else:
            sys.exit(1)

    finally:
        manager.cleanup()


if __name__ == "__main__":
    main()
