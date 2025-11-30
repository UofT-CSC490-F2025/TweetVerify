"""
Training entry script for tweet human/AI binary classification.

Supports:
- RNN / LSTM with Word2Vec embeddings
- BERT / RoBERTa / DeBERTa
- RoBERTa with extra handcrafted features (perplexity, caps ratio, etc.)
"""

# ===== Standard Library =====
import os
import re
import argparse

# ===== Third-Party Libraries =====
import numpy as np
import torch
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, AutoTokenizer, AutoConfig

# ===== Local Modules =====
from src.model.lstm import MyLSTM
from src.model.rnn import MyRNN
from src.model.bert import BertClassifier
from src.model.deberta import DebertaV3
from src.model.roberta import MyRobertaForBinaryClassification
from src.model.roberta_extra import Roberta_Extra
from src.dataloader.bertdataset import BertDataset
from src.dataloader.featuredataset import FeatureDataset
from src.trainer.trainer import Trainer
from src.evaluator.evaluator import Evaluator
from src.utils.convert_indices import convert_indices
from src.utils.seed import set_all_seeds


# ===== Text cleaning utilities =====
def clean_text(s: str) -> str:
    """
    Basic text cleaning for tweets:
    - Remove URLs
    - Remove long hex-like hashes (e.g., very long IDs)
    - Strip leading/trailing whitespace

    Parameters
    ----------
    s : str
        Input text.

    Returns
    -------
    str
        Cleaned text. If input is not a string, returns it unchanged.
    """
    if not isinstance(s, str):
        return s

    url_pattern = r"http\S+|www\.\S+"
    hash_pattern = r"\b[a-fA-F0-9]{20,}\b"

    # Remove URLs
    s = re.sub(url_pattern, "", s)
    # Remove long hash-like tokens
    s = re.sub(hash_pattern, "", s)

    return s.strip()


def main():
    """
    Main training pipeline.

    Steps:
        1. Parse command-line arguments.
        2. Set random seeds and device.
        3. Load Word2Vec model (for RNN / LSTM).
        4. Load and clean human + AI token datasets.
        5. Split into train / val / test.
        6. Build model and dataloaders based on --model.
        7. Train model, evaluate on validation & test sets.
    """
    parser = argparse.ArgumentParser(description="Train tweet human-vs-AI classifiers.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model type: rnn | lstm | bert | roberta | deberta | roberta_extra",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Directory to save trained model checkpoints",
    )
    parser.add_argument(
        "--batch_size", type=int, default=314, help="Mini-batch size for training"
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    print(
        f"model_type: {args.model} | batch_size: {args.batch_size} | "
        f"learning_rate: {args.learning_rate}",
        flush=True,
    )
    print(f"Model checkpoints will be saved to: {args.output_path}", flush=True)

    # Select device (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set all random seeds for reproducibility
    set_all_seeds()

    # Load pre-trained Word2Vec model for RNN/LSTM baselines
    model_w2v = Word2Vec.load("datasets/w2vmodel.model")

    # ===== Load and clean base datasets (without extra features) =====
    human_token = pd.read_csv("datasets/human_token.csv")
    ai_token = pd.read_csv("datasets/ai_token.csv")

    # Apply basic text cleaning
    ai_token["text"] = ai_token["text"].apply(clean_text)
    human_token["text"] = human_token["text"].apply(clean_text)

    # Combine human and AI data
    token = pd.concat([human_token, ai_token], ignore_index=True)

    # Train/val/test split (text + label)
    X_train, X_temp, y_train, y_temp = train_test_split(
        token["text"], token["label"], test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Reset indices to ensure continuous indexing for DataLoader
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Pack (text, label) pairs for RNN/LSTM
    train_data = list(zip(X_train, y_train))
    val_data = list(zip(X_val, y_val))
    test_data = list(zip(X_test, y_test))

    # ===== Model-specific branches =====
    if args.model == "rnn":
        # ---- RNN with Word2Vec embeddings ----
        train_data_indices = convert_indices(train_data, model_w2v)
        val_data_indices = convert_indices(val_data, model_w2v)
        test_data_indices = convert_indices(test_data, model_w2v)

        model = MyRNN(model_w2v, hidden_size=300, num_classes=2).to(device)

        trainer = Trainer(
            device=device,
            model=model,
            train_dataset=train_data_indices,
            val_dataset=val_data_indices,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            model_save_dir=args.output_path,
            batch_size=args.batch_size,
        )
        train_loss, val_acc = trainer.train_model()

        test_evaluator = Evaluator(model, test_data_indices, device)
        acc = test_evaluator.accuracy(args.batch_size)

    elif args.model == "lstm":
        # ---- LSTM with Word2Vec embeddings ----
        train_data_indices = convert_indices(train_data, model_w2v)
        val_data_indices = convert_indices(val_data, model_w2v)
        test_data_indices = convert_indices(test_data, model_w2v)

        model = MyLSTM(model_w2v, hidden_size=256, num_classes=2).to(device)

        trainer = Trainer(
            device=device,
            model=model,
            train_dataset=train_data_indices,
            val_dataset=val_data_indices,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            model_save_dir=args.output_path,
            batch_size=args.batch_size,
        )
        train_loss, val_acc = trainer.train_model()

        test_evaluator = Evaluator(model, test_data_indices, device)
        acc = test_evaluator.accuracy(args.batch_size)

    elif args.model == "bert":
        # ---- BERT (bert-base-uncased) ----
        model = BertClassifier(num_labels=2, dropout=0.3, freeze_bert=False).to(device)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        train_dataset = BertDataset(X_train, y_train, tokenizer)
        val_dataset = BertDataset(X_val, y_val, tokenizer)
        test_dataset = BertDataset(X_test, y_test, tokenizer)

        trainer = Trainer(
            device=device,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            model_save_dir=args.output_path,
            batch_size=args.batch_size,
        )
        train_loss, val_acc = trainer.train_model()

        test_evaluator = Evaluator(model, test_dataset, device)
        acc = test_evaluator.accuracy(args.batch_size)

    elif args.model == "deberta":
        # ---- DeBERTa-v3-large ----
        model_name = "microsoft/deberta-v3-large"
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 2

        model = DebertaV3.from_pretrained(model_name, config=config).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_dataset = BertDataset(X_train, y_train, tokenizer)
        val_dataset = BertDataset(X_val, y_val, tokenizer)
        test_dataset = BertDataset(X_test, y_test, tokenizer)

        trainer = Trainer(
            device=device,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            model_save_dir=args.output_path,
            batch_size=args.batch_size,
        )
        train_loss, val_acc = trainer.train_model()

        test_evaluator = Evaluator(model, test_dataset, device)
        acc = test_evaluator.accuracy(args.batch_size)

    elif args.model == "roberta":
        # ---- RoBERTa-large ----
        model_name = "FacebookAI/roberta-large"
        model = MyRobertaForBinaryClassification.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_dataset = BertDataset(X_train, y_train, tokenizer)
        val_dataset = BertDataset(X_val, y_val, tokenizer)
        test_dataset = BertDataset(X_test, y_test, tokenizer)

        trainer = Trainer(
            device=device,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            model_save_dir=args.output_path,
            batch_size=args.batch_size,
        )
        train_loss, val_acc = trainer.train_model()

        test_evaluator = Evaluator(model, test_dataset, device)
        acc = test_evaluator.accuracy(args.batch_size)

    elif args.model == "roberta_extra":
        # ===== Re-load dataset with additional handcrafted features =====
        human_token = pd.read_csv("datasets/human_token_with_features.csv")
        ai_token = pd.read_csv("datasets/ai_token_with_features.csv")

        ai_token["text"] = ai_token["text"].apply(clean_text)
        human_token["text"] = human_token["text"].apply(clean_text)

        token = pd.concat([human_token, ai_token], ignore_index=True)

        # Fill missing values in feature columns
        token["log_mean_ppl"] = token["log_mean_ppl"].fillna(0)
        token["log_max_ppl"] = token["log_max_ppl"].fillna(0)
        token["caps_ratio"] = token["caps_ratio"].fillna(0)
        token["punc_count"] = token["punc_count"].fillna(0)
        token["digit_ratio"] = token["digit_ratio"].fillna(0)

        # Build feature matrix: [log_mean_ppl, log_max_ppl, caps_ratio, punc_count, digit_ratio]
        feature_data = np.column_stack(
            (
                token["log_mean_ppl"].to_numpy(dtype=np.float64),
                token["log_max_ppl"].to_numpy(dtype=np.float64),
                token["caps_ratio"].to_numpy(dtype=np.float64),
                token["punc_count"].to_numpy(dtype=np.float64),
                token["digit_ratio"].to_numpy(dtype=np.float64),
            )
        )

        # Standardize features (z-score per column)
        feature_data = (feature_data - feature_data.mean(axis=0)) / (
            feature_data.std(axis=0) + 1e-6
        )

        # Jointly shuffle and split (text, label, features)
        X_train, X_temp, y_train, y_temp, feat_train, feat_temp = train_test_split(
            token["text"],
            token["label"],
            feature_data,
            test_size=0.3,
            random_state=42,
        )

        X_val, X_test, y_val, y_test, feat_val, feat_test = train_test_split(
            X_temp,
            y_temp,
            feat_temp,
            test_size=0.5,
            random_state=42,
        )

        # Reset indices for clean dataset construction
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        model_name = "FacebookAI/roberta-large"
        model = Roberta_Extra.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_dataset = FeatureDataset(X_train, y_train, feat_train, tokenizer)
        val_dataset = FeatureDataset(X_val, y_val, feat_val, tokenizer)
        test_dataset = FeatureDataset(X_test, y_test, feat_test, tokenizer)

        trainer = Trainer(
            device=device,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            model_save_dir=args.output_path,
            batch_size=args.batch_size,
        )
        train_loss, val_acc = trainer.train_model()

        test_evaluator = Evaluator(model, test_dataset, device)
        acc = test_evaluator.accuracy(args.batch_size)

    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    # NOTE: Evaluator.accuracy(...) may return (acc, f1, auc) depending on implementation.
    # Here we print whatever is returned as `acc` for backward compatibility.
    print(f"acc, f1, auc: {acc}", flush=True)


if __name__ == "__main__":
    main()
