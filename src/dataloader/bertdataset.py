"""
Dataset class for BERT-based models (BERT, RoBERTa, DeBERTa).

Handles tokenization and formatting for transformer-based models.
"""

from torch.utils.data import Dataset
import torch


class BertDataset(Dataset):
    """
    PyTorch Dataset for BERT-based models.
    
    Args:
        texts: List of input text strings
        labels: List of corresponding labels
        tokenizer: HuggingFace tokenizer instance
        max_len: Maximum sequence length (default: 256)
    """
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - input_ids: Tokenized input sequence
                - attention_mask: Attention mask
                - label: Ground truth label
        """
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
