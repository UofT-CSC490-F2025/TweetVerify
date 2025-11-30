"""
BERT-based binary classifier for tweet human/AI classification.

Uses BERT's pooled [CLS] token representation for classification.
"""

import torch
import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    """
    BERT-based binary classifier for tweet classification.
    
    Args:
        num_labels: Number of output classes (default: 2 for binary classification)
        dropout: Dropout probability (default: 0.3)
        freeze_bert: If True, freeze BERT parameters during training (default: False)
    """
    def __init__(self, num_labels=2, dropout=0.3, freeze_bert=False):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input sequence
            attention_mask: Attention mask for input sequence
            
        Returns:
            logits: Classification logits
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output     # [CLS] token representation
        out = self.dropout(pooled_output)
        logits = self.linear(out)
        return logits

    def get_name(self):
        """Return model name identifier."""
        return 'bert'
