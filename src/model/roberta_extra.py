"""
RoBERTa-based binary classifier with extra handcrafted features.

Combines RoBERTa's [CLS] token representation with handcrafted features
(perplexity, caps ratio, punctuation count, etc.) for enhanced classification.
"""

import torch
import torch.nn as nn
from transformers import RobertaPreTrainedModel, RobertaModel


class Roberta_Extra(RobertaPreTrainedModel):
    """
    RoBERTa-based binary classifier with extra handcrafted features.
    
    Combines RoBERTa embeddings with 5 handcrafted features:
    - log_mean_ppl: Logarithm of mean perplexity
    - log_max_ppl: Logarithm of max perplexity
    - caps_ratio: Ratio of capital letters
    - punc_count: Punctuation count
    - digit_ratio: Ratio of digits
    
    Args:
        config: RoBERTa model configuration
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2

        self.num_extra_features = 6

        self.roberta = RobertaModel(config)

        hidden_size = config.hidden_size

        # Combined size: RoBERTa hidden size + processed extra features (6 -> 32)
        combined_size = hidden_size + 32

        # Batch normalization commented out (can be enabled if needed)
        #self.batch_norm = nn.BatchNorm1d(self.num_extra_features)

        self.dropout = nn.Dropout(0.1)

        # Main classifier combining RoBERTa and feature representations
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_labels),
        )
        # Feature processing layer: 6 features -> 16 -> 32 dimensions
        self.linearlayer = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 32),
        )

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        extra_features=None,
        labels=None,
        **kwargs
    ):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input sequence
            attention_mask: Attention mask for input sequence
            extra_features: Optional tensor of handcrafted features (shape: [batch, 5])
            labels: Optional ground truth labels (not used in forward, for compatibility)
            **kwargs: Additional arguments passed to RoBERTa
            
        Returns:
            logits: Classification logits
        """
        outputs = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        cls_repr = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation

        if extra_features is not None:
            extra_features = extra_features.to(cls_repr.dtype)

            # Process extra features through linear layers
            # norm_features = self.batch_norm(extra_features)  # Optional batch norm
            extra_features = self.linearlayer(extra_features)
            x = torch.cat((cls_repr, extra_features), dim=1)
        else:
            # If no extra features provided, use zero padding
            zeros = torch.zeros(cls_repr.size(0), 32).to(
                cls_repr.device, dtype=cls_repr.dtype
            )
            x = torch.cat((cls_repr, zeros), dim=1)

        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

    def get_name(self):
        """Return model name identifier."""
        return "roberta_extra"
