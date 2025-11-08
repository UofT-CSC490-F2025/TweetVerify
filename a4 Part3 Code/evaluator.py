"""
Evaluation metrics and error analysis.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator:
    """
    Evaluate classifier performance and analyze errors.
    """
    
    def __init__(self, true_labels: List[int], predictions: List[int], 
                 texts: List[str], model_name: str = "Model"):
        """
        Initialize evaluator.
        
        Args:
            true_labels: True labels (0=Human, 1=AI)
            predictions: Predicted labels (0=Human, 1=AI)
            texts: Original texts for error analysis
            model_name: Name of the model being evaluated
        """
        self.true_labels = np.array(true_labels)
        self.predictions = np.array(predictions)
        self.texts = texts
        self.model_name = model_name
        
        if len(true_labels) != len(predictions) or len(true_labels) != len(texts):
            raise ValueError("All inputs must have the same length")
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(self.true_labels, self.predictions),
            'precision': precision_score(self.true_labels, self.predictions, average='binary'),
            'recall': recall_score(self.true_labels, self.predictions, average='binary'),
            'f1': f1_score(self.true_labels, self.predictions, average='binary'),
        }
        
        # Per-class metrics
        cm = confusion_matrix(self.true_labels, self.predictions)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Precision and recall per class
        if tp + fp > 0:
            metrics['precision_ai'] = tp / (tp + fp)
        else:
            metrics['precision_ai'] = 0.0
        
        if tp + fn > 0:
            metrics['recall_ai'] = tp / (tp + fn)
        else:
            metrics['recall_ai'] = 0.0
        
        if tn + fn > 0:
            metrics['precision_human'] = tn / (tn + fn)
        else:
            metrics['precision_human'] = 0.0
        
        if tn + fp > 0:
            metrics['recall_human'] = tn / (tn + fp)
        else:
            metrics['recall_human'] = 0.0
        
        return metrics
    
    def print_report(self):
        """Print detailed classification report."""
        print(f"\n{'='*60}")
        print(f"Evaluation Report: {self.model_name}")
        print(f"{'='*60}")
        
        metrics = self.compute_metrics()
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(self.true_labels, self.predictions)
        print(f"                Predicted")
        print(f"              Human    AI")
        print(f"  Actual Human  {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"          AI    {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        print(f"\nPer-Class Metrics:")
        print(f"  AI Class:")
        print(f"    Precision: {metrics['precision_ai']:.4f}")
        print(f"    Recall:    {metrics['recall_ai']:.4f}")
        print(f"  Human Class:")
        print(f"    Precision: {metrics['precision_human']:.4f}")
        print(f"    Recall:    {metrics['recall_human']:.4f}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(self.true_labels, self.predictions, 
                                   target_names=['Human', 'AI']))
    
    def analyze_errors(self, top_n: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Analyze classification errors.
        
        Args:
            top_n: Number of examples to show for each error type
        
        Returns:
            Dictionary with error analysis dataframes
        """
        df = pd.DataFrame({
            'text': self.texts,
            'true_label': self.true_labels,
            'prediction': self.predictions,
            'correct': self.true_labels == self.predictions
        })
        
        # Add label names
        df['true_label_name'] = df['true_label'].map({0: 'Human', 1: 'AI'})
        df['pred_label_name'] = df['prediction'].map({0: 'Human', 1: 'AI'})
        
        # False Positives: Predicted AI but actually Human
        false_positives = df[(df['true_label'] == 0) & (df['prediction'] == 1)].copy()
        
        # False Negatives: Predicted Human but actually AI
        false_negatives = df[(df['true_label'] == 1) & (df['prediction'] == 0)].copy()
        
        # Correct predictions (for comparison)
        correct_ai = df[(df['true_label'] == 1) & (df['prediction'] == 1)].copy()
        correct_human = df[(df['true_label'] == 0) & (df['prediction'] == 0)].copy()
        
        results = {
            'false_positives': false_positives.head(top_n),
            'false_negatives': false_negatives.head(top_n),
            'correct_ai': correct_ai.head(top_n),
            'correct_human': correct_human.head(top_n),
            'error_summary': pd.DataFrame({
                'Error Type': ['False Positives (Human→AI)', 'False Negatives (AI→Human)'],
                'Count': [len(false_positives), len(false_negatives)],
                'Percentage': [
                    100 * len(false_positives) / len(df),
                    100 * len(false_negatives) / len(df)
                ]
            })
        }
        
        return results
    
    def print_error_analysis(self, top_n: int = 10):
        """Print detailed error analysis."""
        errors = self.analyze_errors(top_n=top_n)
        
        print(f"\n{'='*60}")
        print(f"Error Analysis: {self.model_name}")
        print(f"{'='*60}")
        
        print(f"\nError Summary:")
        print(errors['error_summary'].to_string(index=False))
        
        print(f"\n{'='*60}")
        print(f"False Positives (Human text incorrectly classified as AI) - Top {top_n}:")
        print(f"{'='*60}")
        for idx, row in errors['false_positives'].iterrows():
            print(f"\n[{idx}] {row['text'][:200]}...")
        
        print(f"\n{'='*60}")
        print(f"False Negatives (AI text incorrectly classified as Human) - Top {top_n}:")
        print(f"{'='*60}")
        for idx, row in errors['false_negatives'].iterrows():
            print(f"\n[{idx}] {row['text'][:200]}...")
        
        # Quantitative analysis
        print(f"\n{'='*60}")
        print(f"Quantitative Error Analysis:")
        print(f"{'='*60}")
        
        fp_texts = errors['false_positives']['text'].tolist()
        fn_texts = errors['false_negatives']['text'].tolist()
        
        # Average text length
        fp_avg_len = np.mean([len(str(t)) for t in fp_texts]) if fp_texts else 0
        fn_avg_len = np.mean([len(str(t)) for t in fn_texts]) if fn_texts else 0
        
        # Count URLs, hashtags, mentions
        fp_urls = sum([str(t).count('http') for t in fp_texts])
        fn_urls = sum([str(t).count('http') for t in fn_texts])
        fp_hashtags = sum([str(t).count('#') for t in fp_texts])
        fn_hashtags = sum([str(t).count('#') for t in fn_texts])
        fp_mentions = sum([str(t).count('@') for t in fp_texts])
        fn_mentions = sum([str(t).count('@') for t in fn_texts])
        
        print(f"\nFalse Positives (Human→AI):")
        print(f"  Average text length: {fp_avg_len:.1f} chars")
        print(f"  URLs per text: {fp_urls / len(fp_texts) if fp_texts else 0:.2f}")
        print(f"  Hashtags per text: {fp_hashtags / len(fp_texts) if fp_texts else 0:.2f}")
        print(f"  Mentions per text: {fp_mentions / len(fp_texts) if fp_texts else 0:.2f}")
        
        print(f"\nFalse Negatives (AI→Human):")
        print(f"  Average text length: {fn_avg_len:.1f} chars")
        print(f"  URLs per text: {fn_urls / len(fn_texts) if fn_texts else 0:.2f}")
        print(f"  Hashtags per text: {fn_hashtags / len(fn_texts) if fn_texts else 0:.2f}")
        print(f"  Mentions per text: {fn_mentions / len(fn_texts) if fn_texts else 0:.2f}")
    
    def save_error_analysis(self, output_path: str, top_n: int = 50):
        """Save error analysis to CSV."""
        errors = self.analyze_errors(top_n=top_n)
        
        # Save false positives
        if len(errors['false_positives']) > 0:
            fp_path = output_path.replace('.csv', '_false_positives.csv')
            errors['false_positives'][['text', 'true_label', 'prediction']].to_csv(fp_path, index=False)
            print(f"Saved false positives to {fp_path}")
        
        # Save false negatives
        if len(errors['false_negatives']) > 0:
            fn_path = output_path.replace('.csv', '_false_negatives.csv')
            errors['false_negatives'][['text', 'true_label', 'prediction']].to_csv(fn_path, index=False)
            print(f"Saved false negatives to {fn_path}")


