import pytest
import sys
import subprocess
from unittest.mock import patch, MagicMock
from src.app_wrapper import main as app_wrapper_main
from src.run import main as run_main
from src.train_model import main as train_model_main
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

def test_train_model_main():
    with patch('src.train_model.pd.read_csv') as mock_read, \
         patch('src.train_model.pd.concat') as mock_concat, \
         patch('src.train_model.Word2Vec') as mock_w2v:
         
        mock_df = MagicMock()
        mock_df.__getitem__.return_value.dropna.return_value.astype.return_value.tolist.return_value = ["Hello world"]
        mock_read.return_value = mock_df
        mock_concat.return_value = mock_df
        
        train_model_main()
        
        mock_w2v.assert_called_once()
        mock_w2v.return_value.save.assert_called_once()

def test_run_script_execution():
    """Test executing src/run.py as main script"""
    with patch('subprocess.run') as mock_run, \
         patch('sys.argv', ['run.py', '--arg']):
        
        import runpy
        # We use run_module if installed or run_path
        # Since we are in root, run_path src/run.py works
        runpy.run_path('src/run.py', run_name='__main__')
        
        mock_run.assert_called_once()

def test_app_wrapper_script_execution():
    """Test executing src/app_wrapper.py as main script"""
    # Patch the source of the functions, not the imported name in the script
    # because runpy re-imports/re-executes
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

