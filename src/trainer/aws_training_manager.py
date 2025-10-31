#!/usr/bin/env python3
"""
AWS Training Manager for Web Interface
"""

import os
import re
from datetime import datetime
from typing import Dict, List, Optional
from src.trainer.train_aws_sagemaker import AWSTrainingManager


class WebAWSTrainingManager:
    """Manages AWS SageMaker training jobs for web interface"""

    def __init__(self):
        self.current_training: Optional[Dict] = None
        self.aws_manager = AWSTrainingManager()

        # Load existing completed trainings from model_save directory
        self._load_existing_trainings()

    def _load_existing_trainings(self):
        """Load existing completed trainings from model_save directory"""
        try:
            model_save_dir = "model_save"
            if os.path.exists(model_save_dir):
                model_files = [
                    f for f in os.listdir(model_save_dir) if f.endswith(".pt")
                ]
                if model_files:
                    # Get the most recent model file
                    latest_file = max(
                        model_files,
                        key=lambda x: os.path.getctime(os.path.join(model_save_dir, x)),
                    )

                    # Parse model filename
                    parsed_info = self._parse_model_filename(latest_file)

                    # Store as completed training
                    self.current_training = {
                        "training_id": f"completed_{latest_file.replace('.pt', '')}",
                        "model_type": parsed_info.get("model_type", "UNKNOWN").lower(),
                        "epochs": 100,  # Default value
                        "learning_rate": 0.0001,  # Default value
                        "batch_size": 314,  # Default value
                        "start_time": datetime.now(),
                        "status": "completed",
                        "end_time": datetime.now(),
                        "model_file": latest_file,
                        "model_path": os.path.join(model_save_dir, latest_file),
                        "test_accuracy": (
                            parsed_info.get("accuracy", 0.0) / 100.0
                            if parsed_info.get("accuracy")
                            else None
                        ),
                    }

        except Exception as e:
            print(f"Error loading existing trainings: {e}")

    def _get_active_training(self) -> Optional[Dict]:
        """Get the currently active training (not completed)"""
        if self.current_training and self.current_training["status"] in [
            "starting",
            "running",
        ]:
            return self.current_training
        return None

    def start_training(
        self,
        training_id: str,
        model_type: str,
        epochs: int = 100,
        learning_rate: float = 0.0001,
        batch_size: int = 314,
    ) -> bool:
        """Start a new AWS SageMaker training job"""
        try:
            # Check if there's already an active training (not completed)
            active_training = self._get_active_training()
            if active_training is not None:
                return False  # Another training is already running

            # Store training info
            self.current_training = {
                "training_id": training_id,
                "model_type": model_type,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "start_time": datetime.now(),
                "status": "starting",
            }

            # Start the actual SageMaker training (blocking)
            result = self.aws_manager.start_training_job(
                model_type=model_type,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
            )

            if result["success"]:
                # Training completed successfully
                self.current_training["status"] = "completed"
                self.current_training["end_time"] = datetime.now()
                self.current_training["job_name"] = result["job_name"]

                # Load results
                self._load_training_results()

                return True
            else:
                # Training failed
                self.current_training["status"] = "failed"
                self.current_training["end_time"] = datetime.now()
                return False

        except Exception as e:
            print(f"Error starting AWS training: {e}")
            if self.current_training:
                self.current_training["status"] = "error"
                self.current_training["end_time"] = datetime.now()
            return False

    def _load_training_results(self):
        """Load training results"""
        try:
            # Check if new model files were downloaded
            model_save_dir = "model_save"
            if os.path.exists(model_save_dir):
                model_files = [
                    f for f in os.listdir(model_save_dir) if f.endswith(".pt")
                ]
                if model_files:
                    # Get the most recent model file
                    latest_file = max(
                        model_files,
                        key=lambda x: os.path.getctime(os.path.join(model_save_dir, x)),
                    )
                    latest_path = os.path.join(model_save_dir, latest_file)

                    # Parse model filename to extract accuracy
                    parsed_info = self._parse_model_filename(latest_file)

                    # Update current training with results
                    if self.current_training:
                        self.current_training["model_file"] = latest_file
                        self.current_training["model_path"] = latest_path
                        self.current_training["test_accuracy"] = (
                            parsed_info.get("accuracy", 0.0) / 100.0
                            if parsed_info.get("accuracy")
                            else None
                        )
                        self.current_training["completed_time"] = (
                            datetime.now().isoformat()
                        )

        except Exception as e:
            print(f"Error loading training results: {e}")

    def get_training_status(self, training_id: Optional[str] = None) -> Optional[Dict]:
        """Get training status"""
        if not self.current_training:
            return None

        training = self.current_training
        status = {
            "training_id": training["training_id"],
            "model_type": training["model_type"],
            "epochs": training["epochs"],
            "learning_rate": training["learning_rate"],
            "batch_size": training["batch_size"],
            "start_time": training["start_time"].isoformat(),
            "status": training["status"],
        }

        # Add end time if available
        if "end_time" in training:
            status["end_time"] = training["end_time"].isoformat()

        # Add job name if available
        if "job_name" in training:
            status["job_name"] = training["job_name"]

        # Add results if available
        if "model_file" in training:
            status["model_file"] = training["model_file"]
        if "model_path" in training:
            status["model_path"] = training["model_path"]
        if "test_accuracy" in training:
            status["test_accuracy"] = training["test_accuracy"]
        if "completed_time" in training:
            status["completed_time"] = training["completed_time"]

        return status

    def stop_training(self, training_id: str) -> bool:
        """Stop a running training"""
        if (
            not self.current_training
            or self.current_training["training_id"] != training_id
        ):
            return False

        if self.current_training["status"] == "running":
            # Note: SageMaker training cannot be easily stopped once started
            # This is a placeholder for future implementation
            self.current_training["status"] = "stopped"
            return True

        return False

    def cleanup_training(self, training_id: str):
        """Clean up training resources"""
        if (
            self.current_training
            and self.current_training["training_id"] == training_id
        ):
            self.current_training = None

    def list_active_trainings(self) -> List[Dict]:
        """List all active trainings (including completed ones for display)"""
        if self.current_training:
            status = self.get_training_status()
            return [status] if status else []
        return []

    def list_running_trainings(self) -> List[Dict]:
        """List only running trainings (not completed)"""
        if self.current_training and self.current_training["status"] in [
            "starting",
            "running",
        ]:
            status = self.get_training_status()
            return [status] if status else []
        return []

    def get_training_log_path(self) -> Optional[str]:
        """Get the log file path - always returns the latest log file based on filename date"""
        try:
            training_logs_dir = "logs"
            if not os.path.exists(training_logs_dir):
                return None
            
            # Get all .log files
            log_files = []
            for filename in os.listdir(training_logs_dir):
                if filename.endswith('.log'):
                    log_files.append(os.path.join(training_logs_dir, filename))
            
            if not log_files:
                return None
            
            # Sort by filename date (extract date from filename)
            def extract_date_from_filename(filepath):
                filename = os.path.basename(filepath)
                try:
                    # Expected format: training_<uuid>_<YYYYMMDD>_<HHMMSS>.log
                    # Extract date part (YYYYMMDD_HHMMSS)
                    parts = filename.split('_')
                    if len(parts) >= 4:
                        date_part = parts[-2]  # YYYYMMDD
                        time_part = parts[-1].replace('.log', '')  # HHMMSS
                        return f"{date_part}_{time_part}"
                    else:
                        # Fallback to file modification time
                        return os.path.getmtime(filepath)
                except:
                    # Fallback to file modification time
                    return os.path.getmtime(filepath)
            
            # Sort by date extracted from filename (newest first)
            sorted_logs = sorted(log_files, key=extract_date_from_filename, reverse=True)
            
            # Return the most recent log file
            return sorted_logs[0]
            
        except Exception as e:
            print(f"Error getting training log path: {e}")
            return None

    def _parse_model_filename(self, filename):
        """Parse model filename to extract model type, accuracy, and timestamp"""
        # Pattern: {model_type}_{accuracy}_{date}_{time}.{ext}
        # Example: lstm_92.8_2025-10-12_18-23-37.pt
        pattern = r"^([a-zA-Z]+)_(\d+\.?\d*)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.(pt|pth|pkl|model)$"
        match = re.match(pattern, filename)

        if match:
            model_type = match.group(1).upper()
            accuracy = float(match.group(2))
            date_str = match.group(3)
            time_str = match.group(4)

            # Parse datetime
            try:
                datetime_str = f"{date_str} {time_str.replace('-', ':')}"
                timestamp = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
                formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                timestamp = None
                formatted_time = f"{date_str} {time_str}"

            return {
                "model_type": model_type,
                "accuracy": accuracy,
                "timestamp": timestamp,
                "formatted_time": formatted_time,
                "parsed": True,
            }
        else:
            # Fallback for files that don't match the pattern
            return {
                "model_type": "UNKNOWN",
                "accuracy": 0.0,
                "timestamp": None,
                "formatted_time": "Unknown",
                "parsed": False,
            }


# Global training manager instance
aws_training_manager = WebAWSTrainingManager()