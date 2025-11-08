"""
Main script for baseline LLM classifier evaluation.
"""
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_loader import load_data, create_splits
from baseline_classifier import BaselineClassifier
from evaluator import Evaluator
import pandas as pd

# Conditional import for LLM classifier (OpenAI)
try:
    from llm_classifier import LLMClassifier
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMClassifier = None

# Conditional import for LLM classifier (Hugging Face - no API key needed)
try:
    from llm_classifier_hf import LLMClassifierHF
    LLM_HF_AVAILABLE = True
except ImportError:
    LLM_HF_AVAILABLE = False
    LLMClassifierHF = None


def run_llm_classifier(train_df, val_df, test_df, model_name="gpt-3.5-turbo", api_key=None, use_hf=False):
    """Run LLM classifier on all splits."""
    print("\n" + "="*60)
    print("Running LLM Classifier")
    print("="*60)
    
    if use_hf:
        if not LLM_HF_AVAILABLE:
            raise ImportError("Hugging Face transformers not available. Install with: pip install transformers torch")
        classifier = LLMClassifierHF(model_name=model_name)
    else:
        if not LLM_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        classifier = LLMClassifier(model=model_name, api_key=api_key)
    
    # Evaluate on validation set (skip for faster execution, or use small sample)
    print("\nEvaluating on validation set (first 50 samples for speed)...")
    val_sample = val_df.head(50)  # Sample for speed
    val_results = classifier.predict_df(val_sample)
    val_evaluator = Evaluator(
        val_results['label'].tolist(),
        val_results['prediction'].tolist(),
        val_results['text'].tolist(),
        f"LLM ({model_name}) - Validation"
    )
    val_evaluator.print_report()
    val_evaluator.print_error_analysis(top_n=5)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = classifier.predict_df(test_df)
    test_evaluator = Evaluator(
        test_results['label'].tolist(),
        test_results['prediction'].tolist(),
        test_results['text'].tolist(),
        f"LLM ({model_name}) - Test"
    )
    test_evaluator.print_report()
    test_evaluator.print_error_analysis(top_n=10)
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    model_safe_name = model_name.replace("/", "_")
    test_results.to_csv(output_dir / f"llm_{model_safe_name}_test_results.csv", index=False)
    test_evaluator.save_error_analysis(str(output_dir / f"llm_{model_safe_name}_errors.csv"))
    
    return test_evaluator


def run_baseline_classifier(train_df, val_df, test_df, use_word2vec=False):
    """Run baseline classifier on all splits."""
    print("\n" + "="*60)
    print("Running Baseline Classifier (Logistic Regression on Embeddings)")
    print("="*60)
    
    classifier = BaselineClassifier(use_word2vec=use_word2vec)
    
    # Train on training set
    print("\nTraining on training set...")
    classifier.fit(train_df['text'].tolist(), train_df['label'].tolist())
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_results = classifier.predict_df(val_df)
    val_evaluator = Evaluator(
        val_results['label'].tolist(),
        val_results['prediction'].tolist(),
        val_results['text'].tolist(),
        "Baseline (Logistic Regression) - Validation"
    )
    val_evaluator.print_report()
    val_evaluator.print_error_analysis(top_n=10)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = classifier.predict_df(test_df)
    test_evaluator = Evaluator(
        test_results['label'].tolist(),
        test_results['prediction'].tolist(),
        test_results['text'].tolist(),
        "Baseline (Logistic Regression) - Test"
    )
    test_evaluator.print_report()
    test_evaluator.print_error_analysis(top_n=10)
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    test_results.to_csv(output_dir / "baseline_test_results.csv", index=False)
    test_evaluator.save_error_analysis(str(output_dir / "baseline_errors.csv"))
    
    return test_evaluator


def compare_models(llm_evaluator, baseline_evaluator):
    """Compare LLM and baseline model results."""
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    
    llm_metrics = llm_evaluator.compute_metrics()
    baseline_metrics = baseline_evaluator.compute_metrics()
    
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'LLM': [
            llm_metrics['accuracy'],
            llm_metrics['precision'],
            llm_metrics['recall'],
            llm_metrics['f1']
        ],
        'Baseline': [
            baseline_metrics['accuracy'],
            baseline_metrics['precision'],
            baseline_metrics['recall'],
            baseline_metrics['f1']
        ]
    })
    
    comparison['Difference'] = comparison['LLM'] - comparison['Baseline']
    
    print("\n" + comparison.to_string(index=False))
    
    # Save comparison
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    print(f"\nComparison saved to {output_dir / 'model_comparison.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Baseline LLM Classifier")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per class (for faster testing)")
    parser.add_argument("--llm-model", type=str, default="gpt-3.5-turbo",
                       help="LLM model to use (OpenAI model name or Hugging Face model)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--use-hf", action="store_true",
                       help="Use Hugging Face model (no API key needed)")
    parser.add_argument("--skip-llm", action="store_true",
                       help="Skip LLM classifier (only run baseline)")
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip baseline classifier (only run LLM)")
    parser.add_argument("--use-word2vec", action="store_true",
                       help="Use Word2Vec instead of SentenceTransformer for baseline")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    df = load_data(max_samples=args.max_samples)
    
    # Create splits
    print("\nCreating train/val/test splits...")
    train_df, val_df, test_df = create_splits(df)
    
    # Run classifiers
    llm_evaluator = None
    baseline_evaluator = None
    
    if not args.skip_llm:
        if args.use_hf:
            if not LLM_HF_AVAILABLE:
                print("Hugging Face LLM classifier not available. Install with: pip install transformers torch")
            else:
                try:
                    llm_evaluator = run_llm_classifier(
                        train_df, val_df, test_df, 
                        model_name=args.llm_model,
                        use_hf=True
                    )
                except Exception as e:
                    print(f"Error running Hugging Face LLM classifier: {e}")
                    print("Skipping LLM classifier...")
        else:
            if not LLM_AVAILABLE:
                print("OpenAI LLM classifier not available (missing openai package). Skipping...")
                print("Tip: Use --use-hf to use Hugging Face models (no API key needed)")
            else:
                try:
                    llm_evaluator = run_llm_classifier(
                        train_df, val_df, test_df, 
                        model_name=args.llm_model,
                        api_key=args.api_key,
                        use_hf=False
                    )
                except Exception as e:
                    print(f"Error running LLM classifier: {e}")
                    print("Skipping LLM classifier...")
    
    if not args.skip_baseline:
        try:
            baseline_evaluator = run_baseline_classifier(
                train_df, val_df, test_df,
                use_word2vec=args.use_word2vec
            )
        except Exception as e:
            print(f"Error running baseline classifier: {e}")
            print("Skipping baseline classifier...")
    
    # Compare models
    if llm_evaluator and baseline_evaluator:
        compare_models(llm_evaluator, baseline_evaluator)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

