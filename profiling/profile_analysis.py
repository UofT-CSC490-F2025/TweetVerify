"""
Performance Profiling Script for TweetVerify
Uses Python's cProfile module to analyze critical functions
"""
import cProfile
import pstats
import io
import pandas as pd
import emoji
import re
import torch
from torch.nn.utils.rnn import pad_sequence
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.collate_batch import collate_batch
from src.data_preprocessing.processor import DataProcessor


def profile_function(func, *args, **kwargs):
    """
    Profile a function and return formatted statistics
    
    Args:
        func: Function to profile
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        tuple: (statistics string, function result)
    """
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    stats.print_stats(30)
    
    return s.getvalue(), result


def profile_collate_batch():
    """Profile collate_batch function - batch data collation for training"""
    print("\n" + "="*80)
    print("FUNCTION 1: collate_batch() - Batch Data Collation")
    print("="*80)
    
    batch = []
    for i in range(128):  # Typical batch size
        seq_len = 10 + (i % 50)  # Varying lengths
        indices = list(range(i, i + seq_len))
        label = i % 2
        batch.append((indices, label))
    
    def test_collate():
        for _ in range(200):  # Simulate 200 batches
            collate_batch(batch)
    
    stats, _ = profile_function(test_collate)
    print(stats)
    return stats


def profile_regex_cleaning():
    """Profile regex operations in text cleaning"""
    print("\n" + "="*80)
    print("FUNCTION 2: Regex Text Cleaning Operations")
    print("="*80)
    
    df = pd.read_csv("datalake/curated/twitter/high_quality_human.csv")
    texts = df["text"].head(1000).tolist()
    
    def clean_text(text):
        text = re.sub(r'http\S+', '', str(text))
        text = re.sub(r'@\w+', '', str(text))
        text = re.sub(r'#\w+', '', str(text))
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text)
        text = str(text).lower()
        return text
    
    def test_cleaning():
        cleaned = []
        for text in texts:
            cleaned.append(clean_text(text))
        return cleaned
    
    stats, _ = profile_function(test_cleaning)
    print(stats)
    return stats


def profile_emoji_removal():
    """Profile emoji removal operations"""
    print("\n" + "="*80)
    print("FUNCTION 3: Emoji Removal from Text")
    print("="*80)
    
    df = pd.read_csv("datalake/curated/twitter/high_quality_human.csv")
    texts = df["text"].head(1000).tolist()
    
    def test_emoji_removal():
        cleaned = []
        for text in texts:
            cleaned.append(emoji.replace_emoji(str(text), replace=''))
        return cleaned
    
    stats, _ = profile_function(test_emoji_removal)
    print(stats)
    return stats


def profile_dataframe_operations():
    """Profile pandas DataFrame operations"""
    print("\n" + "="*80)
    print("FUNCTION 4: DataFrame Processing Operations")
    print("="*80)
    
    def test_df_ops():
        human_df = pd.read_csv("datalake/curated/twitter/high_quality_human.csv")
        ai_df = pd.read_csv("datalake/curated/llm/ai_generated.csv")
        
        combined = pd.concat([human_df, ai_df], ignore_index=True)
        combined = combined.dropna(subset=['text'])
        combined = combined.drop_duplicates(subset=['text'])
        
        human_only = combined[combined['label'] == 0].copy()
        ai_only = combined[combined['label'] == 1].copy()
        
        # Slow operations using apply
        combined['text_length'] = combined['text'].apply(lambda x: len(str(x)))
        combined['word_count'] = combined['text'].apply(lambda x: len(str(x).split()))
        
        return combined
    
    stats, result = profile_function(test_df_ops)
    print(stats)
    return stats


def profile_data_processor():
    """Profile DataProcessor.clean_data() - complete pipeline"""
    print("\n" + "="*80)
    print("FUNCTION 5: DataProcessor.clean_data() - Complete Pipeline")
    print("="*80)
    
    human_df = pd.read_csv("datalake/curated/twitter/high_quality_human.csv")
    ai_df = pd.read_csv("datalake/curated/llm/ai_generated.csv")
    
    human_df = human_df.head(500)
    ai_df = ai_df.head(500)
    
    combined = pd.concat([human_df, ai_df], ignore_index=True)
    temp_path = "/tmp/temp_data_profile.parquet"
    combined.to_parquet(temp_path, index=False)
    
    def test_processor():
        processor = DataProcessor(temp_path)
        processor.clean_data()
        return processor
    
    stats, _ = profile_function(test_processor)
    print(stats)
    return stats


def main():
    """Run all profiling tests"""
    print("="*80)
    print("PYTHON PROFILING ANALYSIS - TweetVerify Codebase")
    print("="*80)
    print("Analyzing 5 critical functions for performance optimization\n")
    
    results = {}
    
    try:
        results['collate_batch'] = profile_collate_batch()
    except Exception as e:
        print(f"Error profiling collate_batch: {e}")
    
    try:
        results['regex_cleaning'] = profile_regex_cleaning()
    except Exception as e:
        print(f"Error profiling regex_cleaning: {e}")
    
    try:
        results['emoji_removal'] = profile_emoji_removal()
    except Exception as e:
        print(f"Error profiling emoji_removal: {e}")
    
    try:
        results['dataframe_ops'] = profile_dataframe_operations()
    except Exception as e:
        print(f"Error profiling dataframe_operations: {e}")
    
    try:
        results['data_processor'] = profile_data_processor()
    except Exception as e:
        print(f"Error profiling data_processor: {e}")
    
    print("\n" + "="*80)
    print("PROFILING ANALYSIS COMPLETE")
    print("="*80)
    print("\nResults have been generated successfully.")


if __name__ == "__main__":
    main()

