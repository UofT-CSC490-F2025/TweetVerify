import torch
from torch.nn.utils.rnn import pad_sequence


def collate_batch(batch):
    """
    Returns the input and target tensors for a batch of data

    Parameters:
        `batch` - An iterable data structure of tuples (emb, label),
                  where `emb` is a sequence of word embeddings, and
                  `label` is either 1 or 0.

    Returns: a tuple `(X, t)`, where
        - `X` is a PyTorch tensor of shape (batch_size, sequence_length)
        - `t` is a PyTorch tensor of shape (batch_size)
    where `sequence_length` is the length of the longest sequence in the batch
    """
    text_list = []
    label_list = []
    for text_indices, label in batch:
        text_list.append(torch.tensor(text_indices))
        label_list.append(label)

    X = pad_sequence(text_list, padding_value=0, batch_first=True)
    t = torch.tensor(label_list, dtype=torch.long)
    return X, t
