import pytest
import torch
from unittest.mock import MagicMock, patch
import numpy as np
from src.model.bert import BertClassifier
from src.model.rnn import MyRNN
from src.model.lstm import MyLSTM
from src.model.roberta import MyRobertaForBinaryClassification
from src.model.deberta import DebertaV3

# --- Mock Word2Vec for RNN/LSTM ---
@pytest.fixture
def mock_w2v():
    class MockWV:
        def __init__(self):
            self.vector_size = 10
            self.vectors = np.random.rand(5, 10)
            self.key_to_index = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
        
        def __len__(self):
            return 5
            
    class MockModel:
        def __init__(self):
            self.wv = MockWV()
            self.vector_size = 10
            
    return MockModel()

# --- BERT Tests ---
def test_bert_classifier():
    with patch('src.model.bert.BertModel') as mock_bert_cls:
        mock_bert = MagicMock()
        mock_bert.config.hidden_size = 768
        # Mock output of BERT: (last_hidden_state, pooler_output)
        # We only use pooler_output which is accessible via .pooler_output attribute in output object
        mock_output = MagicMock()
        mock_output.pooler_output = torch.randn(2, 768)
        mock_bert.return_value = mock_output
        mock_bert_cls.from_pretrained.return_value = mock_bert
        
        # Test initialization
        model = BertClassifier(num_labels=2, dropout=0.3, freeze_bert=True)
        assert model.get_name() == 'bert'
        
        # Test forward
        input_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones((2, 10))
        logits = model(input_ids, attention_mask)
        
        assert logits.shape == (2, 2)
        mock_bert.assert_called_once()

# --- RNN Tests ---
def test_rnn_classifier(mock_w2v):
    model = MyRNN(mock_w2v, hidden_size=20, num_classes=2)
    assert model.get_name() == 'rnn'
    
    # Check parameters iteration (skips emb.weight)
    params = list(model.parameters())
    names = [name for name, _ in model.named_parameters()]
    assert 'emb.weight' in names
    # The custom parameters() method should yield fewer parameters than named_parameters()
    # because it filters out emb.weight
    assert len(params) < len(list(model.named_parameters()))
    
    # Forward pass
    # vocab size is 5+1=6. Indices should be < 6.
    inputs = torch.randint(0, 6, (4, 8)) # batch=4, seq=8
    output = model(inputs)
    
    assert output.shape == (4, 2)

# --- LSTM Tests ---
def test_lstm_classifier(mock_w2v):
    model = MyLSTM(mock_w2v, hidden_size=20, num_classes=2)
    assert model.get_name() == 'lstm'
    
    # Check parameters iteration
    params = list(model.parameters())
    assert len(params) < len(list(model.named_parameters()))
    
    # Forward pass
    inputs = torch.randint(0, 6, (4, 8))
    output = model(inputs)
    
    assert output.shape == (4, 2)

# --- RoBERTa Tests ---
def test_roberta_classifier():
    from transformers import PretrainedConfig
    
    # Mock config must be instance of PretrainedConfig
    class MockConfig(PretrainedConfig):
        pass
        
    mock_config = MockConfig()
    mock_config.hidden_size = 768
    mock_config.num_labels = 2
    mock_config.initializer_range = 0.02  # Add missing attribute
    
    with patch('src.model.roberta.RobertaModel') as mock_roberta_cls:
        mock_roberta = MagicMock()
        mock_output = MagicMock()
        # RoBERTa output: last_hidden_state is [batch, seq, hidden]
        mock_output.last_hidden_state = torch.randn(2, 10, 768)
        mock_roberta.return_value = mock_output
        mock_roberta_cls.return_value = mock_roberta
        
        model = MyRobertaForBinaryClassification(mock_config)
        assert model.get_name() == 'roberta'
        
        input_ids = torch.randint(0, 100, (2, 10))
        logits = model(input_ids)
        
        assert logits.shape == (2, 2)

# --- DeBERTa Tests ---
def test_deberta_classifier():
    from transformers import PretrainedConfig
    
    class MockConfig(PretrainedConfig):
        pass
        
    mock_config = MockConfig()
    mock_config.hidden_size = 768
    mock_config.hidden_dropout_prob = 0.1
    mock_config.initializer_range = 0.02  # Add missing attribute
    
    with patch('src.model.deberta.DebertaV2Model') as mock_deberta_cls:
        mock_deberta = MagicMock()
        mock_output = MagicMock()
        # DeBERTa output: last_hidden_state is [batch, seq, hidden]
        mock_output.last_hidden_state = torch.randn(2, 10, 768)
        mock_deberta.return_value = mock_output
        mock_deberta_cls.return_value = mock_deberta
        
        model = DebertaV3(mock_config)
        assert model.get_name() == 'deberta'
        
        input_ids = torch.randint(0, 100, (2, 10))
        logits = model(input_ids)
        
        assert logits.shape == (2, 2)

