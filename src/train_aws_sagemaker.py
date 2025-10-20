#!/usr/bin/env python3
"""
AWS SageMaker training script for TweetVerify models
"""

import os
import sys
import json
import time
import tarfile
import tempfile
import shutil
from pathlib import Path
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from datetime import datetime
import uuid


class AWSTrainingManager:
    """Manages AWS SageMaker training jobs"""

    def __init__(self):
        # AWS Configuration
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region_name = os.getenv("AWS_DEFAULT_REGION", "us-east-2")
        self.role_arn = os.getenv("AWS_ROLE_ARN")  # 可选，如果需要 role_arn

        # 初始化 boto3 session（自动读取环境变量）
        self.boto_session = boto3.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name,
        )

        self.sagemaker_session = sagemaker.Session(boto_session=self.boto_session)
        self.s3_client = self.boto_session.client("s3")
        self.sagemaker_client = self.boto_session.client("sagemaker")

        # Training jobs tracking
        self.active_jobs = {}

    def start_training_job(
        self, model_type, epochs=100, learning_rate=0.0001, batch_size=314
    ):
        """Start a new SageMaker training job"""
        try:
            # Generate unique job name
            job_name = f"tweetverify-{model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"

            print(f"Starting SageMaker training job: {job_name}")

            # Create PyTorch estimator
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
                    "model": model_type,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                },
                output_path=f"s3://sagemaker-{self.region_name}-993399330675/tweetverify-models/",
                job_name=job_name,
            )

            # Start training
            estimator.fit()

            # Store job info
            self.active_jobs[job_name] = {
                "job_name": job_name,
                "model_type": model_type,
                "status": "completed",
                "estimator": estimator,
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
            }

            print(f"Training job {job_name} completed successfully")

            # Download model
            downloaded_files = self.download_model(job_name, estimator)

            return {
                "success": True,
                "job_name": job_name,
                "message": f"SageMaker training job {job_name} completed successfully",
                "downloaded_files": downloaded_files,
            }

        except Exception as e:
            error_msg = f"Failed to start SageMaker training job: {str(e)}"
            print(error_msg)
            return {"success": False, "error": error_msg}

    def download_model(self, job_name, estimator):
        """Download trained model from SageMaker to local model_save directory"""
        try:
            print(f"Downloading model for job: {job_name}")

            # Get model artifact S3 path
            model_artifacts = estimator.model_data

            # Prepare local save path
            model_save_dir = Path("model_save")
            model_save_dir.mkdir(parents=True, exist_ok=True)

            # Parse S3 URI
            if not model_artifacts.startswith("s3://"):
                print(f"Unexpected model artifacts URI: {model_artifacts}")
                return []
            parts = model_artifacts.replace("s3://", "").split("/", 1)
            bucket_name = parts[0]
            key = parts[1] if len(parts) > 1 else ""

            print(f"Downloading artifact from s3://{bucket_name}/{key}")

            downloaded_files = []
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_tar_path = Path(tmp_dir) / "model_artifacts.tar.gz"

                # Download model tar.gz
                self.s3_client.download_file(bucket_name, key, str(tmp_tar_path))
                print(f"Downloaded artifact to {tmp_tar_path}")

                # Extract tarball
                extract_dir = Path(tmp_dir) / "extracted"
                extract_dir.mkdir(parents=True, exist_ok=True)
                try:
                    with tarfile.open(tmp_tar_path, "r:*") as tar:
                        tar.extractall(path=extract_dir)
                    print(f"Extracted artifacts to {extract_dir}")
                except tarfile.ReadError:
                    # Not a tarball - copy directly
                    print("Artifact is not a tar archive; copying as-is")
                    dest_path = model_save_dir / Path(key).name
                    shutil.copyfile(tmp_tar_path, dest_path)
                    downloaded_files.append(str(dest_path))
                    return downloaded_files

                # Collect all extracted files
                for root, _, files in os.walk(extract_dir):
                    for fname in files:
                        src_path = Path(root) / fname
                        dest_path = model_save_dir / fname
                        shutil.copyfile(src_path, dest_path)
                        downloaded_files.append(str(dest_path))
                        print(f"Saved model file: {dest_path}")

            if downloaded_files:
                print(
                    f"Successfully saved {len(downloaded_files)} file(s) to {model_save_dir}"
                )
            else:
                print("No files were extracted from the artifact")

            return downloaded_files

        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            return []

    def get_training_job_status(self, job_name):
        """Get status of a training job"""
        try:
            response = self.sagemaker_client.describe_training_job(
                TrainingJobName=job_name
            )

            status = {
                "job_name": job_name,
                "status": response["TrainingJobStatus"],
                "creation_time": response["CreationTime"].isoformat(),
                "end_time": (
                    response.get("TrainingEndTime", "").isoformat()
                    if response.get("TrainingEndTime")
                    else None
                ),
                "model_artifacts": response.get("ModelArtifacts", {}).get(
                    "S3ModelArtifacts", ""
                ),
                "failure_reason": response.get("FailureReason", ""),
            }

            return status

        except Exception as e:
            return {"error": f"Failed to get job status: {str(e)}"}

    def list_training_jobs(self, max_results=10):
        """List recent training jobs"""
        try:
            response = self.sagemaker_client.list_training_jobs(
                NameContains="tweetverify",
                SortBy="CreationTime",
                SortOrder="Descending",
                MaxResults=max_results,
            )

            jobs = []
            for job in response["TrainingJobSummaries"]:
                jobs.append(
                    {
                        "job_name": job["TrainingJobName"],
                        "status": job["TrainingJobStatus"],
                        "creation_time": job["CreationTime"].isoformat(),
                        "end_time": (
                            job.get("TrainingEndTime", "").isoformat()
                            if job.get("TrainingEndTime")
                            else None
                        ),
                    }
                )

            return jobs

        except Exception as e:
            print(f"Error listing training jobs: {str(e)}")
            return []

    def stop_training_job(self, job_name):
        """Stop a running training job"""
        try:
            self.sagemaker_client.stop_training_job(TrainingJobName=job_name)
            print(f"Stopped training job: {job_name}")
            return True
        except Exception as e:
            print(f"Error stopping training job: {str(e)}")
            return False


def main():
    """Main function for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Start AWS SageMaker training job")
    parser.add_argument(
        "model_type", choices=["rnn", "lstm", "bert"], help="Model type"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=314, help="Batch size")

    args = parser.parse_args()

    # Initialize training manager
    manager = AWSTrainingManager()

    # Start training
    result = manager.start_training_job(
        model_type=args.model_type,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )

    if result["success"]:
        print(f"✅ {result['message']}")
        sys.exit(0)
    else:
        print(f"❌ {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
