"""
Dataset class for RoBERTa with extra handcrafted features.

Handles tokenization and formatting for models that use both
text embeddings and handcrafted features.
"""

import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """
    PyTorch Dataset for models using text + handcrafted features.
    
    Args:
        texts: List of input text strings
        labels: List of corresponding labels
        features: Array of handcrafted features (shape: [n_samples, 5])
        tokenizer: HuggingFace tokenizer instance
        max_len: Maximum sequence length (default: 256)
    """
    def __init__(self, texts, labels, features, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.features = features
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, item):
        """
        Get a single sample from the dataset.
        
        Args:
            item: Sample index
            
        Returns:
            Dictionary containing:
                - input_ids: Tokenized input sequence
                - attention_mask: Attention mask
                - label: Ground truth label
                - extra_features: Handcrafted features tensor
        """
        text = str(self.texts[item])
        label = self.labels[item]
        feature = self.features[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
            "extra_features": torch.tensor(feature, dtype=torch.float),
        }
