"""
Base model wrapper for classification task.
"""
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from typing import List, Dict, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ClassificationModelWrapper:
    """
    Wrapper for transformer-based classification models.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        device: Optional[str] = None
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model_name: HuggingFace model identifier
            num_labels: Number of classification labels (2 for binary)
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
        
        print(f"[Model] Loaded {model_name} on {self.device}")
    
    def tokenize(self, texts: List[str], max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Tokenize texts for model input."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)
    
    def predict(self, texts: List[str], return_probs: bool = False) -> np.ndarray:
        """
        Predict labels for texts.
        
        Args:
            texts: List of input texts
            return_probs: If True, return probabilities; else return predictions
        
        Returns:
            Predictions or probabilities
        """
        self.model.eval()
        tokenized = self.tokenize(texts)
        
        with torch.no_grad():
            outputs = self.model(**tokenized)
            logits = outputs.logits
        
        if return_probs:
            probs = torch.softmax(logits, dim=-1)
            return probs.cpu().numpy()
        else:
            predictions = torch.argmax(logits, dim=-1)
            return predictions.cpu().numpy()
    
    def compute_reward(
        self,
        texts: List[str],
        true_labels: List[int],
        metric: str = "f1"
    ) -> float:
        """
        Compute reward based on classification performance.
        
        Args:
            texts: Input texts
            true_labels: Ground truth labels
            metric: Metric to use ('f1', 'accuracy', 'precision', 'recall')
        
        Returns:
            Reward value (0.0 to 1.0)
        """
        predictions = self.predict(texts)
        
        if metric == "f1":
            reward = f1_score(true_labels, predictions, average='binary')
        elif metric == "accuracy":
            reward = accuracy_score(true_labels, predictions)
        elif metric == "precision":
            reward = precision_score(true_labels, predictions, average='binary', zero_division=0)
        elif metric == "recall":
            reward = recall_score(true_labels, predictions, average='binary', zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return float(reward)
    
    def get_model(self):
        """Get the underlying model."""
        return self.model
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer
    
    def save(self, path: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[Model] Saved to {path}")
    
    def load(self, path: str):
        """Load model and tokenizer."""
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"[Model] Loaded from {path}")


