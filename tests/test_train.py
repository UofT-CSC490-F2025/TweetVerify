import pytest
from unittest.mock import patch, MagicMock
import sys
from src.train import main

import pandas as pd

@pytest.fixture
def mock_dependencies():
    with patch('src.train.argparse.ArgumentParser') as mock_parser, \
         patch('src.train.os.makedirs') as mock_makedirs, \
         patch('src.train.torch.device') as mock_device, \
         patch('src.train.set_all_seeds') as mock_seeds, \
         patch('src.train.pd.read_csv') as mock_read_csv, \
         patch('src.train.pd.concat') as mock_concat, \
         patch('src.train.Word2Vec.load') as mock_w2v_load, \
         patch('src.train.train_test_split') as mock_tts, \
         patch('src.train.MyRNN') as mock_rnn, \
         patch('src.train.MyLSTM') as mock_lstm, \
         patch('src.train.BertClassifier') as mock_bert, \
         patch('src.train.BertTokenizer') as mock_bert_tokenizer, \
         patch('src.train.BertDataset') as mock_bert_dataset, \
         patch('src.train.Trainer') as mock_trainer, \
         patch('src.train.Evaluator') as mock_evaluator, \
         patch('src.train.convert_indices') as mock_convert, \
         patch('src.train.DebertaV3') as mock_deberta, \
         patch('src.train.AutoConfig') as mock_autoconfig, \
         patch('src.train.AutoTokenizer') as mock_autotokenizer, \
         patch('src.train.MyRobertaForBinaryClassification') as mock_roberta:
        
        # Setup common mocks
        mock_args = MagicMock()
        mock_args.output_path = "dummy_path"
        mock_args.batch_size = 32
        mock_args.learning_rate = 0.001
        mock_args.epochs = 1
        mock_parser.return_value.parse_args.return_value = mock_args
        
        # Mock DataFrame returns
        # We can mock concat to return a MagicMock that behaves enough like a DF, or patch it out
        # Since we patched pd.concat, we don't need real DFs for input, but we need concat to return something
        # that mock_tts can handle.
        
        mock_df = MagicMock()
        mock_df.__len__.return_value = 10
        # Mock __getitem__ for column access
        mock_df.__getitem__.return_value = mock_df
        
        mock_read_csv.return_value = mock_df
        mock_concat.return_value = mock_df
        
        # Mock split
        # train_test_split returns 4 objects
        mock_tts.return_value = (mock_df, mock_df, mock_df, mock_df)
        
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
            'trainer': mock_trainer,
            'evaluator': mock_evaluator
        }

def test_train_rnn(mock_dependencies):
    mock_dependencies['args'].model = "rnn"
    main()
    mock_dependencies['rnn'].assert_called_once()
    mock_dependencies['trainer'].assert_called_once()

def test_train_lstm(mock_dependencies):
    mock_dependencies['args'].model = "lstm"
    main()
    mock_dependencies['lstm'].assert_called_once()
    mock_dependencies['trainer'].assert_called_once()

def test_train_bert(mock_dependencies):
    mock_dependencies['args'].model = "bert"
    main()
    mock_dependencies['bert'].assert_called_once()
    mock_dependencies['trainer'].assert_called_once()

def test_train_deberta(mock_dependencies):
    mock_dependencies['args'].model = "deberta"
    main()
    mock_dependencies['deberta'].from_pretrained.assert_called_once()
    mock_dependencies['trainer'].assert_called_once()

def test_train_roberta(mock_dependencies):
    mock_dependencies['args'].model = "roberta"
    main()
    mock_dependencies['roberta'].from_pretrained.assert_called_once()
    mock_dependencies['trainer'].assert_called_once()

