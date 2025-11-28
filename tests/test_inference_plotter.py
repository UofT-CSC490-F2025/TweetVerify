import pytest
import torch
import os
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from src.inference.predictor import Predictor
from src.plotter.plotter import plotter

# --- Predictor Tests ---

@pytest.fixture
def mock_w2v():
    class MockWV:
        def __init__(self):
            self.key_to_index = {"hello": 0, "world": 1}
            
    class MockWord2Vec:
        def __init__(self):
            self.wv = MockWV()
            
    return MockWord2Vec()

def test_predictor_rnn_lstm(mock_w2v):
    # Mock model
    mock_model = MagicMock()
    mock_model.get_name.return_value = "lstm"
    mock_model.eval.return_value = None
    # Mock output logits (batch_size=1, num_classes=2)
    mock_model.return_value = torch.tensor([[2.0, 1.0]]) # Class 0 higher
    
    device = torch.device("cpu")
    
    with patch('src.inference.predictor.Word2Vec.load') as mock_load:
        mock_load.return_value = mock_w2v
        
        predictor = Predictor(mock_model, device, tokenizer=None)
        
        # Test single prediction
        pred, conf = predictor.predict("hello world unknown")
        
        assert pred == 0 # Argmax of [2.0, 1.0] is 0
        assert isinstance(conf, float)
        
        # Verify tensor construction logic (hello->1, world->2, unknown->0)
        # We can't easily inspect the tensor passed to model because it's created inside,
        # but we can verify model was called
        mock_model.assert_called()
        
        # Test batch prediction
        # Adjust mock for batch of 2
        mock_model.return_value = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
        
        results = predictor.predict_batch(["hello", "world"])
        assert len(results) == 2
        assert isinstance(results[0], tuple)
        assert results[0][0] == 0 # First is class 0
        assert results[1][0] == 1 # Second is class 1

def test_predictor_rnn_unknown_word_edge_case(mock_w2v):
    # Test the edge case where word index retrieval logic takes the 'else' branch
    mock_model = MagicMock()
    mock_model.get_name.return_value = "rnn"
    # Mock return value as tensor
    mock_model.return_value = torch.tensor([[1.0, 2.0]])
    
    device = torch.device("cpu")
    
    with patch('src.inference.predictor.Word2Vec.load') as mock_load:
        # Create a weird dict-like object that claims to contain a key but returns None on get
        class WeirdDict:
            def __contains__(self, key):
                return True
            def get(self, key):
                return None
                
        mock_w2v.wv.key_to_index = WeirdDict()
        mock_load.return_value = mock_w2v
        
        predictor = Predictor(mock_model, device, tokenizer=None)
        
        # This should trigger the 'else' branch line 43
        predictor.predict("weird")
        
        # Verify model called with tensor containing 0 (padding)
        args = mock_model.call_args
        tensor = args[0][0]
        assert tensor[0][0] == 0

def test_predictor_bert():
    mock_model = MagicMock()
    mock_model.get_name.return_value = "bert"
    mock_model.eval.return_value = None
    # Mock output
    mock_model.return_value = torch.tensor([[1.0, 2.0]]) # Class 1 higher
    
    device = torch.device("cpu")
    mock_tokenizer = MagicMock()
    # Mock tokenizer output
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]])
    }
    
    with patch('src.inference.predictor.Word2Vec.load'):
        predictor = Predictor(mock_model, device, tokenizer=mock_tokenizer)
        
        # Test single prediction
        pred, conf = predictor.predict("hello world")
        
        assert pred == 1
        mock_tokenizer.assert_called()
        
        # Test batch prediction (uses BERT logic)
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([1, 2]), # Return tensor directly for list append
            "attention_mask": torch.tensor([1, 1])
        }
        # Adjust mock model output for batch of 2
        mock_model.return_value = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        
        results = predictor.predict_batch(["test1", "test2"])
        assert len(results) == 2
        assert results[0][0] == 1
        assert results[1][0] == 0

# --- Plotter Tests ---

def test_plotter(tmp_path):
    save_path = tmp_path / "plot.png"
    p = plotter(str(save_path))
    
    # Mock matplotlib.pyplot
    with patch('src.plotter.plotter.plt') as mock_plt:
        p.plot(
            num_epochs=5,
            train_loss=[0.5, 0.4, 0.3, 0.2, 0.1],
            train_acc=[0.5, 0.6, 0.7, 0.8, 0.9],
            val_acc=[0.5, 0.55, 0.6, 0.65, 0.7]
        )
        
        # Should create 2 figures
        assert mock_plt.figure.call_count == 2
        # Should save 2 times
        assert mock_plt.savefig.call_count == 2
        mock_plt.savefig.assert_called_with(str(save_path))
        # Should close 2 times
        assert mock_plt.close.call_count == 2
