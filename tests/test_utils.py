import pytest
import torch
import os
import numpy as np
from unittest.mock import patch, MagicMock
from src.utils.seed import set_all_seeds
from src.utils.canonical_id import canonical_id
from src.utils.convert_indices import convert_indices
from src.utils.collate_batch import collate_batch
from src.utils.get_from_s3 import download_dataset, download_model, safe_download

# --- test_seed.py ---
def test_set_all_seeds():
    set_all_seeds(123)
    assert os.environ["PYTHONHASHSEED"] == "123"
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    # Verify reproducibility
    r1 = np.random.rand()
    set_all_seeds(123)
    r2 = np.random.rand()
    assert r1 == r2

# --- test_canonical_id.py ---
def test_canonical_id():
    cid1 = canonical_id("twitter", "12345")
    cid2 = canonical_id("twitter", "12345")
    assert cid1 == cid2
    assert len(cid1) == 40 # sha1 hexdigest length
    
    cid3 = canonical_id("llm", "12345")
    assert cid1 != cid3

# --- test_convert_indices.py ---
def test_convert_indices():
    # Mock Word2Vec model
    class MockKeyToIndex:
        def __init__(self):
            self.key_to_index = {"hello": 1, "world": 2}
            
    class MockWV:
        def __init__(self):
            self.wv = MockKeyToIndex()
            
    model_w2v = MockWV()
    
    data = [
        ("Hello world", 1),
        ("Unknown word", 0)
    ]
    
    result = convert_indices(data, model_w2v)
    
    # "hello" -> 1 + 1 = 2
    # "world" -> 2 + 1 = 3
    assert result[0][0] == [2, 3]
    assert result[0][1] == 1
    
    # "unknown" -> 0
    # "word" -> 0
    assert result[1][0] == [0, 0]
    assert result[1][1] == 0

# --- test_collate_batch.py ---
def test_collate_batch():
    # Batch of (indices, label)
    batch = [
        ([1, 2, 3], 1),
        ([4, 5], 0)
    ]
    
    X, t = collate_batch(batch)
    
    # Check padding (0 is padding value)
    assert X.shape == (2, 3)
    assert torch.equal(X[0], torch.tensor([1, 2, 3]))
    assert torch.equal(X[1], torch.tensor([4, 5, 0]))
    
    # Check labels
    assert torch.equal(t, torch.tensor([1, 0], dtype=torch.long))

# --- test_get_from_s3.py ---
def test_download_dataset():
    with patch('src.utils.get_from_s3.boto3.client') as mock_client_cls:
        mock_s3 = MagicMock()
        mock_client_cls.return_value = mock_s3
        
        download_dataset()
        
        assert mock_s3.download_file.call_count == 5
        # Check calls (partial check)
        calls = mock_s3.download_file.call_args_list
        assert calls[0][0] == ("datasettweet", "ai_token.csv", "datasets/ai_token.csv")
        assert calls[1][0] == ("datasettweet", "human_token.csv", "datasets/human_token.csv")
        assert calls[2][0] == ("datasettweet", "w2vmodel.model", "datasets/w2vmodel.model")

def test_download_model():
    with patch('src.utils.get_from_s3.boto3.client') as mock_client_cls:
        mock_s3 = MagicMock()
        mock_client_cls.return_value = mock_s3
        
        download_model()
        
        assert mock_s3.download_file.call_count == 5

def test_safe_download():
    mock_s3 = MagicMock()
    
    # Case 1: File exists
    with patch('os.path.exists', return_value=True), \
         patch('os.makedirs'):
        safe_download(mock_s3, 'bucket', 'key', 'dir/path')
        mock_s3.download_file.assert_not_called()
        
    # Case 2: File does not exist
    with patch('os.path.exists', return_value=False), \
         patch('os.makedirs'):
        safe_download(mock_s3, 'bucket', 'key', 'dir/path')
        mock_s3.download_file.assert_called_once()
