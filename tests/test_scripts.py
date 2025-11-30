import pytest
import sys
import os
import torch
import subprocess
from unittest.mock import patch, MagicMock
from src.app_wrapper import main as app_wrapper_main
from src.run import main as run_main
import pandas as pd

def test_app_wrapper_main():
    with patch('src.app_wrapper.download_dataset') as mock_dd, \
         patch('src.app_wrapper.download_model') as mock_dm, \
         patch('subprocess.Popen') as mock_popen:
        
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        
        app_wrapper_main()
        
        mock_dd.assert_called_once()
        mock_dm.assert_called_once()
        assert mock_popen.call_count == 2

def test_run_main():
    with patch('subprocess.run') as mock_run, \
         patch('sys.argv', ['run.py', '--arg']):
        
        run_main()
        
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == sys.executable
        assert args[2] == 'src.train'
        assert args[3] == '--arg'

def test_run_script_execution():
    """Test executing src/run.py as main script"""
    with patch('subprocess.run') as mock_run, \
         patch('sys.argv', ['run.py', '--arg']):
        
        import runpy
        runpy.run_path('src/run.py', run_name='__main__')
        
        mock_run.assert_called_once()

def test_app_wrapper_script_execution():
    """Test executing src/app_wrapper.py as main script"""
    with patch('src.utils.get_from_s3.download_dataset') as mock_dd, \
         patch('src.utils.get_from_s3.download_model') as mock_dm, \
         patch('subprocess.Popen') as mock_popen:
        
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        
        import runpy
        runpy.run_path('src/app_wrapper.py', run_name='__main__')
        
        mock_dd.assert_called_once()
        mock_dm.assert_called_once()
        assert mock_popen.call_count == 2


def test_train_script_execution():
    """Test executing src/train.py as main script"""
    with patch('sys.argv', ['train.py', '--model', 'bert', '--epochs', '1']), \
         patch('src.train.argparse.ArgumentParser') as mock_parser, \
         patch('src.trainer.trainer.Trainer') as mock_trainer_cls, \
         patch('src.evaluator.evaluator.Evaluator'), \
         patch('src.train.pd.read_csv'), \
         patch('src.train.pd.concat'), \
         patch('sklearn.model_selection.train_test_split') as mock_tts, \
         patch('src.dataloader.bertdataset.BertDataset') as mock_dataset_cls, \
         patch('src.train.os.makedirs'), \
         patch.dict(os.environ, {'SM_MODEL_DIR': '/tmp/model_save'}), \
         patch('src.train.BertClassifier') as mock_bert_cls, \
         patch('src.train.BertTokenizer.from_pretrained') as mock_tokenizer_cls:
    
        mock_args = MagicMock()
        mock_args.model = 'bert'
        mock_args.batch_size = 32
        mock_args.learning_rate = 1e-3
        mock_args.epochs = 1
        mock_parser.return_value.parse_args.return_value = mock_args

        # Configure Trainer mock
        mock_trainer_instance = mock_trainer_cls.return_value
        mock_trainer_instance.train_model.return_value = ([], []) # Return empty lists for loss/acc

        
        mock_tts.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        
        # Configure mock dataset
        mock_dataset_instance = MagicMock()
        # Fix pickling error by allowing pickle
        mock_dataset_instance.__reduce__ = lambda: (MagicMock, ())
        mock_dataset_instance.__len__.return_value = 10
    
        # Make __getitem__ return valid tensors for collate_fn
        # Simulate a BATCHED return (batch_size=1)
        mock_dataset_instance.__getitem__.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]), # Batch size 1
            'attention_mask': torch.tensor([[1, 1, 1]]),
            'label': torch.tensor([1])
        }
    
        # Patch DataLoader to avoid multiprocessing (num_workers=0)
        with patch('src.trainer.trainer.DataLoader') as mock_dl_cls:
            # We need to make sure the mock behaves like an iterable for tqdm
            mock_dl_instance = MagicMock()
            mock_dl_instance.__iter__.return_value = [mock_dataset_instance.__getitem__(0)]
            mock_dl_instance.__len__.return_value = 1
            mock_dl_cls.return_value = mock_dl_instance
            
            mock_dataset_cls.return_value = mock_dataset_instance
    
            import runpy
            try:
                runpy.run_path('src/train.py', run_name='__main__')
            except SystemExit:
                pass
