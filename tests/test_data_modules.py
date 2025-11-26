import pytest
import pandas as pd
import os
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.data_ingestion.llm_db import LLMDB
from src.data_ingestion.main_db import MainDB
from src.data_ingestion.twitter_db import TwitterDB
from src.data_ingestion.twitter_scrape import scrape_user_tweets, scrape_keyword_tweets, scrape_time_range_tweets
from src.data_preprocessing.processor import DataProcessor

# --- Test LLMDB ---
def test_llm_db_process():
    records = [
        {"text_id": "1", "text": " AI Generated ", "label": 1, "model": "gpt-4"},
        {"text_id": "2", "text": "", "label": 1}, # Empty, should be filtered
    ]
    
    with patch('src.data_ingestion.llm_db.pq.write_table') as mock_write, \
         patch('src.data_ingestion.llm_db.LLM_CURATED', Path('mock/path')):
        
        db = LLMDB(records)
        path = db.process_llm_dataset()
        
        assert "llm_curated" in str(path)
        assert mock_write.called
        
        # Verify DataFrame logic
        args = mock_write.call_args[0]
        pa_table = args[0]
        df = pa_table.to_pandas()
        
        assert len(df) == 1 # Empty text filtered
        assert df.iloc[0]['text'] == "AI Generated"
        assert df.iloc[0]['source'] == 'llm'

# --- Test TwitterDB ---
def test_twitter_db_process():
    records = [
        {"text_id": "1", "text": "Tweet 1 ", "label": 0},
        {"text_id": "2", "text": "Tweet 1 ", "label": 0}, # Duplicate
        {"text_id": "3", "text": "   ", "label": 0}, # Empty
    ]
    
    with patch('src.data_ingestion.twitter_db.pq.write_table') as mock_write, \
         patch('src.data_ingestion.twitter_db.TWITTER_CURATED', Path('mock/path')):
        
        db = TwitterDB(records)
        path = db.process_twitter_dataset()
        
        assert "twitter_curated" in str(path)
        assert mock_write.called
        
        df = mock_write.call_args[0][0].to_pandas()
        assert len(df) == 1 # Duplicate and empty removed
        assert df.iloc[0]['text'] == "Tweet 1"

def test_twitter_db_save_csv():
    records = [{"text_id": "1", "text": "Tweet"}]
    db = TwitterDB(records)
    
    # Mock process first to populate self.df
    with patch('src.data_ingestion.twitter_db.pq.write_table'):
        with patch('src.data_ingestion.twitter_db.TWITTER_CURATED', Path('mock')):
            db.process_twitter_dataset()
            
    with patch('pandas.DataFrame.to_csv') as mock_to_csv:
        db.save_to_csv("output.csv")
        mock_to_csv.assert_called_once()

# --- Test MainDB ---
def test_main_db_read_download():
    with patch('pandas.read_csv') as mock_read:
        mock_df = MagicMock()
        mock_read.return_value = mock_df
        
        # Mock getitem for column selection
        mock_df.__getitem__.return_value = mock_df
        
        db = MainDB("twitter.parquet", "llm.parquet")
        
        assert mock_read.call_count == 2 # Sentiment140 and DAIGT

def test_main_db_merge():
    with patch('pandas.read_csv') as mock_read_csv, \
         patch('pandas.read_parquet') as mock_read_parquet, \
         patch('src.data_ingestion.main_db.pq.write_table') as mock_write, \
         patch('src.data_ingestion.main_db.MAIN_CURATED', Path('mock/path')):
        
        # Setup mocks
        mock_df = pd.DataFrame({'text': ['t1'], 'label': [0]})
        mock_read_csv.return_value = mock_df
        mock_read_parquet.return_value = mock_df
        
        db = MainDB("twitter.parquet", "llm.parquet")
        path = db.merge_to_main()
        
        assert "main_curated" in str(path)
        assert mock_write.called
        
        df_result = mock_write.call_args[0][0].to_pandas()
        # 4 dataframes merged (twitter, llm, sentiment, daigt)
        # each has 1 row in our mock
        assert len(df_result) == 4 

# --- Test Twitter Scrape ---
def test_scrape_user_tweets():
    with patch('src.data_ingestion.twitter_scrape.tweepy.Client') as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        # Mock user
        mock_user = MagicMock()
        mock_user.data.id = "123"
        mock_user.data.username = "testuser"
        mock_client.get_user.return_value = mock_user
        
        # Mock tweets
        mock_tweet = MagicMock()
        mock_tweet.id = "t1"
        mock_tweet.text = "Hello"
        mock_tweet.created_at = "2024-01-01"
        
        mock_response = MagicMock()
        mock_response.data = [mock_tweet]
        mock_client.get_users_tweets.return_value = mock_response
        
        records = scrape_user_tweets("testuser")
        
        assert len(records) == 1
        assert records[0]['text'] == "Hello"
        assert records[0]['username'] == "testuser"

def test_scrape_keyword_tweets():
    with patch('src.data_ingestion.twitter_scrape.tweepy.Client') as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        # Mock tweets
        mock_tweet = MagicMock()
        mock_tweet.id = "t1"
        mock_tweet.text = "Hello #AI"
        mock_tweet.created_at = "2024-01-01"
        del mock_tweet.author_id # Test AttributeError handling
        
        mock_response = MagicMock()
        mock_response.data = [mock_tweet]
        mock_client.search_recent_tweets.return_value = mock_response
        
        records = scrape_keyword_tweets("#AI")
        
        assert len(records) == 1
        assert records[0]['text'] == "Hello #AI"
        assert records[0]['user_id'] is None

def test_scrape_time_range_tweets():
    with patch('src.data_ingestion.twitter_scrape.tweepy.Client') as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        mock_tweet = MagicMock()
        mock_tweet.id = "t1"
        mock_tweet.text = "Time tweet"
        mock_tweet.created_at = "2024-01-05"
        
        mock_response = MagicMock()
        mock_response.data = [mock_tweet]
        mock_client.search_recent_tweets.return_value = mock_response
        
        records = scrape_time_range_tweets("#AI")
        
        assert len(records) == 1
        assert records[0]['text'] == "Time tweet"

# --- Test DataProcessor ---
@pytest.fixture
def sample_data_parquet(tmp_path):
    df = pd.DataFrame({
        'text': ['Hello World! http://url.com @user #tag 😀', '  Bad Spacing  ', None, 'AI Text'],
        'label': [0, 0, 0, 1]
    })
    path = tmp_path / "test.parquet"
    df.to_parquet(path)
    return path

def test_data_processor_loading(sample_data_parquet):
    processor = DataProcessor(sample_data_parquet)
    assert len(processor.data) == 4

def test_clean_tweet_text(sample_data_parquet):
    processor = DataProcessor(sample_data_parquet)
    
    # Full cleaning
    text = "Hello World! http://url.com @user #tag 😀"
    cleaned = processor.clean_tweet_text(text)
    # lower, no url, no user, no hashtag, no emoji
    assert cleaned == "hello world!"
    
    # Partial cleaning
    cleaned = processor.clean_tweet_text(text, lower=False, remove_emoji=False)
    assert "😀" in cleaned
    assert "Hello" in cleaned
    
    # None handling
    assert processor.clean_tweet_text(None) == ''

def test_clean_data_full_flow(sample_data_parquet):
    processor = DataProcessor(sample_data_parquet)
    
    # Mock character removal logic for AI text to ensure it runs
    # We need overlapping chars logic to trigger
    
    cleaned_df = processor.clean_data()
    
    # 1. None row dropped -> 3 rows
    # 2. '  Bad Spacing  ' -> 'bad spacing'
    # 3. 'Hello ...' -> 'hello world!'
    assert len(cleaned_df) == 3
    
    texts = cleaned_df['text'].tolist()
    assert 'hello world!' in texts
    assert 'bad spacing' in texts
    # AI Text might be modified by char removal logic if AI chars are not in human chars
    # 'ai text' -> chars: a, i, t, e, x. Human chars: h, e, l, o, w, r, d, !, b, a, s, p, c, i, n, g
    # 'x' is not in human chars (oops, 'text' has 'x'). 'bad spacing' has no 'x'.
    # So 'x' might be removed from AI text.

def test_save_load_data(sample_data_parquet, tmp_path):
    processor = DataProcessor(sample_data_parquet)
    processor.clean_data()
    
    # Save Parquet
    out_pq = tmp_path / "out.parquet"
    processor.save_data(out_pq, 'parquet')
    assert out_pq.exists()
    
    # Save CSV
    out_csv = tmp_path / "out.csv"
    processor.save_data(out_csv, 'csv')
    assert out_csv.exists()
    
    # Invalid type
    with pytest.raises(ValueError):
        processor.save_data(out_csv, 'json')
        
    # Get data
    assert len(processor.get_data()) == 3

def test_processor_errors(sample_data_parquet):
    processor = DataProcessor(sample_data_parquet)
    # Not cleaned yet
    with pytest.raises(ValueError):
        processor.save_data("path")
    
    with pytest.raises(ValueError):
        processor.get_data()

