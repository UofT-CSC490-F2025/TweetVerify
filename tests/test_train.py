import pytest
from unittest.mock import patch, MagicMock
import sys
from src.train import main
import numpy as np
import pandas as pd
from contextlib import ExitStack

@pytest.fixture
def mock_dependencies():
    with ExitStack() as stack:
        mock_parser = stack.enter_context(patch('src.train.argparse.ArgumentParser'))
        mock_makedirs = stack.enter_context(patch('src.train.os.makedirs'))
        mock_device = stack.enter_context(patch('src.train.torch.device'))
        mock_seeds = stack.enter_context(patch('src.train.set_all_seeds'))
        mock_read_csv = stack.enter_context(patch('src.train.pd.read_csv'))
        mock_concat = stack.enter_context(patch('src.train.pd.concat'))
        mock_w2v_load = stack.enter_context(patch('src.train.Word2Vec.load'))
        mock_tts = stack.enter_context(patch('src.train.train_test_split'))
        mock_rnn = stack.enter_context(patch('src.train.MyRNN'))
        mock_lstm = stack.enter_context(patch('src.train.MyLSTM'))
        mock_bert = stack.enter_context(patch('src.train.BertClassifier'))
        mock_bert_tokenizer = stack.enter_context(patch('src.train.BertTokenizer'))
        mock_bert_dataset = stack.enter_context(patch('src.train.BertDataset'))
        mock_feature_dataset = stack.enter_context(patch('src.train.FeatureDataset'))
        mock_trainer = stack.enter_context(patch('src.train.Trainer'))
        mock_evaluator = stack.enter_context(patch('src.train.Evaluator'))
        mock_convert = stack.enter_context(patch('src.train.convert_indices'))
        mock_deberta = stack.enter_context(patch('src.train.DebertaV3'))
        mock_autoconfig = stack.enter_context(patch('src.train.AutoConfig'))
        mock_autotokenizer = stack.enter_context(patch('src.train.AutoTokenizer'))
        mock_roberta = stack.enter_context(patch('src.train.MyRobertaForBinaryClassification'))
        mock_roberta_extra = stack.enter_context(patch('src.train.Roberta_Extra'))
        
        # Setup common mocks
        mock_args = MagicMock()
        mock_args.output_path = "dummy_path"
        mock_args.batch_size = 32
        mock_args.learning_rate = 0.001
        mock_args.epochs = 1
        mock_parser.return_value.parse_args.return_value = mock_args
        
        # Mock DataFrame returns
        mock_df = MagicMock()
        mock_df.__len__.return_value = 10
        # Mock __getitem__ for column access
        mock_df.__getitem__.return_value = mock_df
        # Mock fillna
        mock_df.fillna.return_value = mock_df
        # Mock values for numpy conversion
        mock_df.values = np.zeros((10, 1))
        
        mock_read_csv.return_value = mock_df
        mock_concat.return_value = mock_df
        
        # Mock Trainer return
        mock_trainer_instance = mock_trainer.return_value
        mock_trainer_instance.train_model.return_value = (0.1, 0.9)
        
        yield {
            'args': mock_args,
            'rnn': mock_rnn,
            'lstm': mock_lstm,
            'bert': mock_bert,
            'deberta': mock_deberta,
            'roberta': mock_roberta,
            'roberta_extra': mock_roberta_extra,
            'trainer': mock_trainer,
            'evaluator': mock_evaluator,
            'tts': mock_tts,
            'df': mock_df
        }

def test_train_rnn(mock_dependencies):
    mock_dependencies['args'].model = "rnn"
    # Configure TTS to return 4 items
    mock_dependencies['tts'].side_effect = [
        (mock_dependencies['df'], mock_dependencies['df'], mock_dependencies['df'], mock_dependencies['df']),
        (mock_dependencies['df'], mock_dependencies['df'], mock_dependencies['df'], mock_dependencies['df'])
    ]
    
    main()
    mock_dependencies['rnn'].assert_called_once()
    mock_dependencies['trainer'].assert_called_once()

def test_train_lstm(mock_dependencies):
    mock_dependencies['args'].model = "lstm"
    mock_dependencies['tts'].side_effect = [
        (mock_dependencies['df'], mock_dependencies['df'], mock_dependencies['df'], mock_dependencies['df']),
        (mock_dependencies['df'], mock_dependencies['df'], mock_dependencies['df'], mock_dependencies['df'])
    ]
    
    main()
    mock_dependencies['lstm'].assert_called_once()
    mock_dependencies['trainer'].assert_called_once()

def test_train_bert(mock_dependencies):
    mock_dependencies['args'].model = "bert"
    mock_dependencies['tts'].side_effect = [
        (mock_dependencies['df'], mock_dependencies['df'], mock_dependencies['df'], mock_dependencies['df']),
        (mock_dependencies['df'], mock_dependencies['df'], mock_dependencies['df'], mock_dependencies['df'])
    ]
    
    main()
    mock_dependencies['bert'].assert_called_once()
    mock_dependencies['trainer'].assert_called_once()

def test_train_deberta(mock_dependencies):
    mock_dependencies['args'].model = "deberta"
    mock_dependencies['tts'].side_effect = [
        (mock_dependencies['df'], mock_dependencies['df'], mock_dependencies['df'], mock_dependencies['df']),
        (mock_dependencies['df'], mock_dependencies['df'], mock_dependencies['df'], mock_dependencies['df'])
    ]
    
    main()
    mock_dependencies['deberta'].from_pretrained.assert_called_once()
    mock_dependencies['trainer'].assert_called_once()

def test_train_roberta(mock_dependencies):
    mock_dependencies['args'].model = "roberta"
    mock_dependencies['tts'].side_effect = [
        (mock_dependencies['df'], mock_dependencies['df'], mock_dependencies['df'], mock_dependencies['df']),
        (mock_dependencies['df'], mock_dependencies['df'], mock_dependencies['df'], mock_dependencies['df'])
    ]
    
    main()
    mock_dependencies['roberta'].from_pretrained.assert_called_once()
    mock_dependencies['trainer'].assert_called_once()

def test_train_roberta_extra(mock_dependencies):
    mock_dependencies['args'].model = "roberta_extra"
    
    # roberta_extra calls train_test_split 4 times total:
    d = mock_dependencies['df']
    mock_dependencies['tts'].side_effect = [
        (d, d, d, d),       # Common 1
        (d, d, d, d),       # Common 2
        (d, d, d, d, d, d), # Extra 1
        (d, d, d, d, d, d)  # Extra 2
    ]
    
    main()
    
    mock_dependencies['roberta_extra'].from_pretrained.assert_called_once()
    mock_dependencies['trainer'].assert_called_once()

def test_train_unknown_model(mock_dependencies):
    mock_dependencies['args'].model = "unknown"
    d = mock_dependencies['df']
    # Common split calls
    mock_dependencies['tts'].side_effect = [
        (d, d, d, d),
        (d, d, d, d)
    ]
    
    # Should raise UnboundLocalError because acc is not defined
    with pytest.raises(UnboundLocalError):
        main()
