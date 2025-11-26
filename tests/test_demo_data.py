import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.train import demo_prepare_data

def test_demo_prepare_data():
    with patch('src.train.scrape_user_tweets') as mock_scrape, \
         patch('src.train.TwitterDB') as mock_twitter_db_cls, \
         patch('src.train.LLMDB') as mock_llm_db_cls, \
         patch('src.train.MainDB') as mock_main_db_cls:
        
        # Setup mocks
        mock_scrape.return_value = [{"text": "test"}]
        
        mock_twitter_db = MagicMock()
        mock_twitter_db.process_twitter_dataset.return_value = Path("twitter_path")
        mock_twitter_db_cls.return_value = mock_twitter_db
        
        mock_llm_db = MagicMock()
        mock_llm_db.process_llm_dataset.return_value = Path("llm_path")
        mock_llm_db_cls.return_value = mock_llm_db
        
        mock_main_db = MagicMock()
        mock_main_db.merge_to_main.return_value = Path("main_path")
        mock_main_db_cls.return_value = mock_main_db
        
        # Execute
        result = demo_prepare_data()
        
        # Assertions
        mock_scrape.assert_called_with("NASA", max_results=100)
        mock_twitter_db_cls.assert_called_with([{"text": "test"}])
        mock_twitter_db.process_twitter_dataset.assert_called_once()
        
        mock_llm_db_cls.assert_called()
        args = mock_llm_db_cls.call_args[0][0]
        assert len(args) == 2
        assert args[0]['model'] == 'gpt-4'
        
        mock_main_db_cls.assert_called_with(Path("twitter_path"), Path("llm_path"))
        mock_main_db.merge_to_main.assert_called_once()
        
        assert result == Path("main_path")

