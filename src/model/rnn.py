"""
RNN-based binary classifier for tweet human/AI classification.

Uses Word2Vec embeddings and bidirectional RNN layers.
"""

import torch
import torch.nn as nn
import numpy as np


class MyRNN(nn.Module):
    """
    Bidirectional RNN classifier with Word2Vec embeddings.
    
    Args:
        model_w2v: Pre-trained Word2Vec model
        hidden_size: Hidden dimension size for RNN
        num_classes: Number of output classes (default: 2 for binary classification)
    """
    def __init__(self, model_w2v, hidden_size, num_classes):
        super(MyRNN, self).__init__()
        self.vocab_size = len(model_w2v.wv)+1
        self.emb_size = model_w2v.vector_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.emb = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=0)
        # Add an extra zero-initialized vector as the padding vector
        self.emb.weight.data.copy_(torch.from_numpy(
            np.vstack((np.zeros((1, self.emb_size)), model_w2v.wv.vectors))))
        self.rnn = nn.RNN(self.emb_size, hidden_size,
                          bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, X):
        """
        Forward pass through the model.
        
        Args:
            X: Input token indices tensor
            
        Returns:
            output: Classification logits
        """
        X = X.long()
        embedded = self.emb(X)
        _, hidden = self.rnn(embedded)
        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        dropped = self.dropout(hidden)
        output = self.fc(dropped)
        return output

    def parameters(self):
        """
        Generator for trainable parameters (excluding embedding weights).
        Embedding weights are frozen to preserve Word2Vec embeddings.
        """
        for name, param in self.named_parameters():
            if name != 'emb.weight':
                yield param

    def get_name(self):
        """Return model name identifier."""
        return 'rnn'
