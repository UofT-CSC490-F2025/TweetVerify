"""
Data loading utilities for RLVR training.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Dataset paths
HUMAN_PATH = PROJECT_ROOT / "src/data_tokenize/datasets/high_quality_human.csv"
AI_PATH = PROJECT_ROOT / "src/data_tokenize/datasets/ai_generated.csv"


def load_data(max_samples: int = None) -> pd.DataFrame:
    """
    Load and combine human and AI datasets.
    
    Args:
        max_samples: Maximum number of samples per class (None for all)
    
    Returns:
        Combined dataframe with 'text' and 'label' columns
    """
    if not os.path.exists(HUMAN_PATH):
        raise FileNotFoundError(f"Missing file: {HUMAN_PATH}")
    if not os.path.exists(AI_PATH):
        raise FileNotFoundError(f"Missing file: {AI_PATH}")
    
    human_df = pd.read_csv(HUMAN_PATH)
    ai_df = pd.read_csv(AI_PATH)
    
    # Basic validation
    for name, df in [("human", human_df), ("ai", ai_df)]:
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{name} dataset must have columns ['text', 'label'].")
        df.dropna(subset=["text"], inplace=True)
        df["label"] = df["label"].astype(int)
    
    # Sample if needed
    if max_samples:
        human_df = human_df.sample(n=min(max_samples, len(human_df)), random_state=42)
        ai_df = ai_df.sample(n=min(max_samples, len(ai_df)), random_state=42)
    
    # Combine
    df_all = pd.concat([human_df[["text", "label"]], ai_df[["text", "label"]]], 
                       ignore_index=True)
    
    # Shuffle
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"[data_loader] Loaded {len(df_all)} samples "
          f"(Human: {len(human_df)}, AI: {len(ai_df)})")
    
    return df_all


def create_splits(df: pd.DataFrame, 
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train, validation, and test splits with stratified sampling.
    
    Args:
        df: Dataframe with 'text' and 'label' columns
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_state: Random seed
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(1 - train_ratio),
        stratify=df['label'],
        random_state=random_state
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        stratify=temp_df['label'],
        random_state=random_state
    )
    
    print(f"[data_loader] Splits created:")
    print(f"  Train: {len(train_df)} samples "
          f"(Human: {sum(train_df['label'] == 0)}, AI: {sum(train_df['label'] == 1)})")
    print(f"  Val: {len(val_df)} samples "
          f"(Human: {sum(val_df['label'] == 0)}, AI: {sum(val_df['label'] == 1)})")
    print(f"  Test: {len(test_df)} samples "
          f"(Human: {sum(test_df['label'] == 0)}, AI: {sum(test_df['label'] == 1)})")
    
    return train_df, val_df, test_df


