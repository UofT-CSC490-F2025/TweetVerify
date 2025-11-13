"""
Benchmark Script: Compare Original vs Optimized Implementations
Measures actual performance improvements from optimizations
"""
import time
import pandas as pd
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Original implementations
from src.utils.collate_batch import collate_batch
from src.data_preprocessing.processor import DataProcessor

# Optimized implementations
from profiling.optimizations import (
    collate_batch_optimized,
    TextCleanerOptimized,
    EmojiRemoverOptimized,
    process_dataframe_optimized,
    DataProcessorOptimized
)
import re
import emoji


def benchmark_function(func, iterations=10):
    """
    Benchmark a function over multiple iterations
    
    Args:
        func: Function to benchmark
        iterations: Number of times to run
        
    Returns:
        tuple: (average_time, std_dev)
    """
    times = []
    for _ in range(iterations):
        start = time.time()
        func()
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    return avg_time, std_dev


def benchmark_collate_batch():
    """Benchmark 1: Batch Collation"""
    print("\n" + "="*80)
    print("BENCHMARK 1: Batch Collation (collate_batch)")
    print("="*80)
    
    # Create test data
    batch = []
    for i in range(128):
        seq_len = 10 + (i % 50)
        indices = list(range(i, i + seq_len))
        label = i % 2
        batch.append((indices, label))
    
    # Original version
    def original():
        for _ in range(100):
            collate_batch(batch)
    
    # Optimized version
    def optimized():
        for _ in range(100):
            collate_batch_optimized(batch)
    
    print("\nRunning original implementation...")
    orig_time, orig_std = benchmark_function(original, iterations=5)
    
    print("Running optimized implementation...")
    opt_time, opt_std = benchmark_function(optimized, iterations=5)
    
    speedup = orig_time / opt_time
    improvement = ((orig_time - opt_time) / orig_time) * 100
    
    print(f"\nOriginal:  {orig_time:.4f}s ± {orig_std:.4f}s")
    print(f"Optimized: {opt_time:.4f}s ± {opt_std:.4f}s")
    print(f"Speedup:   {speedup:.2f}x")
    print(f"Improvement: {improvement:.1f}%")
    
    return {
        'function': 'collate_batch',
        'original_time': orig_time,
        'optimized_time': opt_time,
        'speedup': speedup,
        'improvement_pct': improvement
    }


def benchmark_regex_cleaning():
    """Benchmark 2: Regex Text Cleaning"""
    print("\n" + "="*80)
    print("BENCHMARK 2: Regex Text Cleaning")
    print("="*80)
    
    df = pd.read_csv("datalake/curated/twitter/high_quality_human.csv")
    texts = df["text"].head(2000).tolist()  # Use more texts for better measurement
    print(f"Testing with {len(texts)} tweets")
    
    # Original version - multiple passes, no pre-compilation
    def original():
        cleaned = []
        for text in texts:
            # Multiple regex passes
            text = re.sub(r'http\S+', '', str(text))
            text = re.sub(r'@\w+', '', str(text))
            text = re.sub(r'#\w+', '', str(text))
            text = str(text).strip()
            text = re.sub(r'\s+', ' ', text)
            text = str(text).lower()
            cleaned.append(text)
        return cleaned
    
    # Optimized version - single pass with pre-compiled patterns and optimized operations
    cleaner = TextCleanerOptimized()
    def optimized():
        return cleaner.clean_texts_batch(texts)
    
    print("\nRunning original implementation...")
    orig_time, orig_std = benchmark_function(original, iterations=10)
    
    print("Running optimized implementation...")
    opt_time, opt_std = benchmark_function(optimized, iterations=10)
    
    speedup = orig_time / opt_time
    improvement = ((orig_time - opt_time) / orig_time) * 100
    
    print(f"\nOriginal:  {orig_time:.4f}s ± {orig_std:.4f}s")
    print(f"Optimized: {opt_time:.4f}s ± {opt_std:.4f}s")
    print(f"Speedup:   {speedup:.2f}x")
    print(f"Improvement: {improvement:.1f}%")
    
    return {
        'function': 'regex_cleaning',
        'original_time': orig_time,
        'optimized_time': opt_time,
        'speedup': speedup,
        'improvement_pct': improvement
    }


def benchmark_emoji_removal():
    """Benchmark 3: Emoji Removal"""
    print("\n" + "="*80)
    print("BENCHMARK 3: Emoji Removal")
    print("="*80)
    
    df = pd.read_csv("datalake/curated/twitter/high_quality_human.csv")
    texts = df["text"].head(1000).tolist()
    
    # Original version - always uses emoji library
    def original():
        cleaned = []
        for text in texts:
            cleaned.append(emoji.replace_emoji(str(text), replace=''))
        return cleaned
    
    # Optimized version - fast-path check
    remover = EmojiRemoverOptimized()
    def optimized():
        return remover.remove_emoji_batch(texts)
    
    print("\nRunning original implementation...")
    orig_time, orig_std = benchmark_function(original, iterations=5)
    
    print("Running optimized implementation...")
    opt_time, opt_std = benchmark_function(optimized, iterations=5)
    
    speedup = orig_time / opt_time
    improvement = ((orig_time - opt_time) / orig_time) * 100
    
    print(f"\nOriginal:  {orig_time:.4f}s ± {orig_std:.4f}s")
    print(f"Optimized: {opt_time:.4f}s ± {opt_std:.4f}s")
    print(f"Speedup:   {speedup:.2f}x")
    print(f"Improvement: {improvement:.1f}%")
    
    return {
        'function': 'emoji_removal',
        'original_time': orig_time,
        'optimized_time': opt_time,
        'speedup': speedup,
        'improvement_pct': improvement
    }


def benchmark_dataframe_operations():
    """Benchmark 4: DataFrame Operations"""
    print("\n" + "="*80)
    print("BENCHMARK 4: DataFrame Operations (Processing Only)")
    print("="*80)
    
    # Load data once, outside the benchmark
    print("Loading data...")
    human_df = pd.read_csv("datalake/curated/twitter/high_quality_human.csv")
    ai_df = pd.read_csv("datalake/curated/llm/ai_generated.csv")
    print(f"Loaded {len(human_df)} human tweets and {len(ai_df)} AI tweets")
    
    # Original version - using apply
    def original():
        combined = pd.concat([human_df.copy(), ai_df.copy()], ignore_index=True)
        combined = combined.dropna(subset=['text'])
        combined = combined.drop_duplicates(subset=['text'])
        
        human_only = combined[combined['label'] == 0].copy()
        ai_only = combined[combined['label'] == 1].copy()
        
        # Slow apply operations
        combined['text_length'] = combined['text'].apply(lambda x: len(str(x)))
        combined['word_count'] = combined['text'].apply(lambda x: len(str(x).split()))
        
        return combined
    
    # Optimized version - hybrid approach
    def optimized():
        combined = pd.concat([human_df.copy(), ai_df.copy()], ignore_index=True)
        combined = combined.dropna(subset=['text'])
        combined = combined.drop_duplicates(subset=['text'])
        
        human_only = combined[combined['label'] == 0]
        ai_only = combined[combined['label'] == 1]
        
        # OPTIMIZED OPERATIONS
        # str.len() is fast and vectorized
        combined['text_length'] = combined['text'].str.len()
        # For word count, use faster method: str.count(' ') + 1
        # This avoids the expensive split operation
        combined['word_count'] = combined['text'].str.count(' ') + 1
        
        return combined
    
    print("\nRunning original implementation...")
    orig_time, orig_std = benchmark_function(original, iterations=5)
    
    print("Running optimized implementation...")
    opt_time, opt_std = benchmark_function(optimized, iterations=5)
    
    speedup = orig_time / opt_time
    improvement = ((orig_time - opt_time) / orig_time) * 100
    
    print(f"\nOriginal:  {orig_time:.4f}s ± {orig_std:.4f}s")
    print(f"Optimized: {opt_time:.4f}s ± {opt_std:.4f}s")
    print(f"Speedup:   {speedup:.2f}x")
    print(f"Improvement: {improvement:.1f}%")
    
    return {
        'function': 'dataframe_ops',
        'original_time': orig_time,
        'optimized_time': opt_time,
        'speedup': speedup,
        'improvement_pct': improvement
    }


def benchmark_data_processor():
    """Benchmark 5: Complete Data Processing Pipeline"""
    print("\n" + "="*80)
    print("BENCHMARK 5: Complete Data Processing Pipeline")
    print("="*80)
    
    # Prepare test data
    human_df = pd.read_csv("datalake/curated/twitter/high_quality_human.csv")
    ai_df = pd.read_csv("datalake/curated/llm/ai_generated.csv")
    human_df = human_df.head(500)
    ai_df = ai_df.head(500)
    combined = pd.concat([human_df, ai_df], ignore_index=True)
    
    temp_path_orig = "/tmp/temp_data_original.parquet"
    temp_path_opt = "/tmp/temp_data_optimized.parquet"
    combined.to_parquet(temp_path_orig, index=False)
    combined.to_parquet(temp_path_opt, index=False)
    
    # Original version
    def original():
        processor = DataProcessor(temp_path_orig)
        processor.clean_data()
        return processor
    
    # Optimized version
    def optimized():
        processor = DataProcessorOptimized(temp_path_opt)
        processor.clean_data_vectorized()
        return processor
    
    print("\nRunning original implementation...")
    orig_time, orig_std = benchmark_function(original, iterations=3)
    
    print("Running optimized implementation...")
    opt_time, opt_std = benchmark_function(optimized, iterations=3)
    
    speedup = orig_time / opt_time
    improvement = ((orig_time - opt_time) / orig_time) * 100
    
    print(f"\nOriginal:  {orig_time:.4f}s ± {orig_std:.4f}s")
    print(f"Optimized: {opt_time:.4f}s ± {opt_std:.4f}s")
    print(f"Speedup:   {speedup:.2f}x")
    print(f"Improvement: {improvement:.1f}%")
    
    return {
        'function': 'data_processor',
        'original_time': orig_time,
        'optimized_time': opt_time,
        'speedup': speedup,
        'improvement_pct': improvement
    }


def generate_summary_table(results):
    """Generate summary table of all benchmarks"""
    print("\n" + "="*80)
    print("SUMMARY: Performance Improvements")
    print("="*80)
    
    # Print table header
    print(f"\n{'Function':<25} {'Original':<15} {'Optimized':<15} {'Speedup':<12} {'Improvement':<12}")
    print("-" * 85)
    
    table_data = []
    for result in results:
        func_name = result['function']
        orig_time = f"{result['original_time']:.4f}s"
        opt_time = f"{result['optimized_time']:.4f}s"
        speedup = f"{result['speedup']:.2f}x"
        improvement = f"{result['improvement_pct']:.1f}%"
        
        print(f"{func_name:<25} {orig_time:<15} {opt_time:<15} {speedup:<12} {improvement:<12}")
        
        table_data.append([func_name, orig_time, opt_time, speedup, improvement])
    
    print("-" * 85)
    
    # Calculate overall statistics
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    avg_improvement = sum(r['improvement_pct'] for r in results) / len(results)
    
    print(f"\nAverage Speedup: {avg_speedup:.2f}x")
    print(f"Average Improvement: {avg_improvement:.1f}%")
    
    return table_data


def main():
    """Run all benchmarks"""
    print("="*80)
    print("BENCHMARK: Original vs Optimized Implementations")
    print("="*80)
    print("\nThis will compare the performance of original and optimized versions")
    print("of the 5 critical functions identified through profiling.\n")
    
    results = []
    
    try:
        results.append(benchmark_collate_batch())
    except Exception as e:
        print(f"Error in benchmark_collate_batch: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results.append(benchmark_regex_cleaning())
    except Exception as e:
        print(f"Error in benchmark_regex_cleaning: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results.append(benchmark_emoji_removal())
    except Exception as e:
        print(f"Error in benchmark_emoji_removal: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results.append(benchmark_dataframe_operations())
    except Exception as e:
        print(f"Error in benchmark_dataframe_operations: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results.append(benchmark_data_processor())
    except Exception as e:
        print(f"Error in benchmark_data_processor: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate summary
    if results:
        table_data = generate_summary_table(results)
        
        # Save results to file
        with open("benchmark_results.txt", "w") as f:
            f.write("BENCHMARK RESULTS: Original vs Optimized\n")
            f.write("="*80 + "\n\n")
            f.write(f"{'Function':<25} {'Original':<15} {'Optimized':<15} {'Speedup':<12} {'Improvement':<12}\n")
            f.write("-" * 85 + "\n")
            for row in table_data:
                f.write(f"{row[0]:<25} {row[1]:<15} {row[2]:<15} {row[3]:<12} {row[4]:<12}\n")
            f.write("-" * 85 + "\n")
            
            avg_speedup = sum(r['speedup'] for r in results) / len(results)
            avg_improvement = sum(r['improvement_pct'] for r in results) / len(results)
            f.write(f"\nAverage Speedup: {avg_speedup:.2f}x\n")
            f.write(f"Average Improvement: {avg_improvement:.1f}%\n")
        
        print("\nResults saved to benchmark_results.txt")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

