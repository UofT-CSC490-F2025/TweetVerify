"""
Evaluation script for binary tweet classification models.

Supports:
- RNN / LSTM with Word2Vec features
- BERT / RoBERTa / DeBERTa
- RoBERTa with extra handcrafted features
- Voting ensemble over multiple transformer models
"""

# ===== Standard Library =====
import os
import argparse
import statistics

# ===== Third-Party Libraries =====
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import BertTokenizer, AutoTokenizer, AutoConfig

# ===== Local Modules =====
from src.utils.collate_batch import collate_batch
from src.model.lstm import MyLSTM
from src.model.rnn import MyRNN
from src.model.bert import BertClassifier
from src.model.deberta import DebertaV3
from src.model.roberta import MyRobertaForBinaryClassification
from src.model.roberta_extra import Roberta_Extra
from src.dataloader.bertdataset import BertDataset
from src.dataloader.featuredataset import FeatureDataset
from src.utils.convert_indices import convert_indices
from src.utils.seed import set_all_seeds

# ===== Paths to Best Checkpoints =====
# Defaults are relative to project root
DEFAULT_MODEL_DIR = "model_save"

def get_best_model_path(model_type, model_dir=DEFAULT_MODEL_DIR):
    """Find the best model checkpoint in the directory based on naming convention."""
    if not os.path.exists(model_dir):
        return None
    
    files = [f for f in os.listdir(model_dir) if f.endswith(".pt") and f.startswith(model_type)]
    if not files:
        return None
        
    # Format: modeltype_accuracy_timestamp.pt
    # Find the checkpoint with the highest accuracy.
    # If parsing fails, fallback to the latest file based on timestamp (lexicographical sort).
    
    best_file = None
    best_acc = -1.0
    
    for f in files:
        parts = f.split("_")
        try:
            # Attempt to find accuracy part. 
            # Standard format: rnn_67.5_2025...
            # parts[1] is usually accuracy
            acc = float(parts[1])
            if acc > best_acc:
                best_acc = acc
                best_file = f
        except (IndexError, ValueError):
            continue
            
    if best_file:
        return os.path.join(model_dir, best_file)
    
    # Fallback: just return the last one alphabetically (likely latest date)
    files.sort()
    return os.path.join(model_dir, files[-1])



class Evaluator:
    """
    Generic evaluator for binary classification models.

    Supports:
    - Sequence models (RNN / LSTM) with custom collate function
    - Transformer-based models (BERT / RoBERTa / DeBERTa)
    - Models with additional feature inputs (e.g., Roberta_Extra)
    """

    def __init__(
        self,
        model: torch.nn.Module | None,
        dataset: torch.utils.data.Dataset | None,
        device: torch.device,
        num_workers: int = 1,
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module or None
            Model to evaluate. Can be None when using `accuracy_voting`.
        dataset : Dataset or None
            Dataset to evaluate on. Can be None when using `accuracy_voting`.
        device : torch.device
            Device on which to run evaluation (CPU / CUDA).
        num_workers : int, optional
            Number of workers for DataLoader.
        """
        if model is not None:
            self.model = model.to(device)
        if dataset is not None:
            self.dataset = dataset
        self.device = device
        self.num_workers = num_workers

    def accuracy(self, batch_size: int = 64):
        """
        Compute accuracy, F1, and AUC for a single model and dataset.

        The predicted class is chosen via argmax over class probabilities.

        Parameters
        ----------
        batch_size : int, optional
            Batch size for evaluation DataLoader.

        Returns
        -------
        acc : float
            Accuracy in [0, 1].
        f1 : float
            F1 score (binary).
        auc : float
            ROC AUC score using positive-class probabilities.
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        # Use custom collate function for RNN/LSTM models
        if self.model.get_name() in ["lstm", "rnn"]:
            dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                collate_fn=collate_batch,
                pin_memory=True,
                num_workers=self.num_workers,
                shuffle=True,
            )
        else:
            dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

        with torch.no_grad():
            for batch in dataloader:
                if self.model.get_name() in ["lstm", "rnn"]:
                    # RNN / LSTM: batch = (sequence_tensor, labels)
                    x, t = batch
                    x, t = x.to(self.device), t.to(self.device)
                    z = self.model(x)
                    labels = t
                else:
                    # Transformer models: batch is a dict with input_ids, attention_mask, label
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)

                    if self.model.get_name() == "roberta_extra":
                        # Additional handcrafted features
                        extra_features = batch["extra_features"].to(self.device)
                        z = self.model(input_ids, attention_mask, extra_features)
                    else:
                        z = self.model(input_ids, attention_mask)

                y_pred = torch.argmax(z, dim=1)
                all_preds.append(y_pred.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_probs.append(torch.softmax(z, dim=1).cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs[:, 1])

        return acc, f1, auc

    def accuracy_ensemble(self, models, datasets, collate_fns=None, weights=None, batch_size=64):
        """
        Perform weighted averaging ensemble over multiple models.

        Each model can have its own dataset (e.g., different tokenization),
        and optionally its own collate function.

        Parameters
        ----------
        models : list[torch.nn.Module]
            List of models to ensemble.
        datasets : list[Dataset]
            List of datasets, one per model.
        collate_fns : list[callable] or None, optional
            Per-model collate functions. If None, uses default for each model.
        weights : list[float] or None, optional
            Per-model weights for averaging. If None, all weights are equal.
        batch_size : int, optional
            Batch size for evaluation.

        Returns
        -------
        acc : float
            Ensemble accuracy.
        f1 : float
            Ensemble F1 score.
        auc : float
            Ensemble ROC AUC score.
        """
        # ===== 0. Normalize averaging weights =====
        if weights is None:
            weights = [1.0] * len(models)
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()

        # If custom collate_fns not provided, default to using collate_batch for RNN and None otherwise
        if collate_fns is None:
            collate_fns = [None] * len(models)

        model_probs = []
        true_labels = None

        # ===== 1. Run each model separately, collect probabilities =====
        for idx, model in enumerate(models):
            model.eval()
            model.to(self.device)

            dataset = datasets[idx]
            collate_fn = collate_fns[idx]

            is_rnn = model.get_name() in ["lstm", "rnn"]
            if is_rnn and collate_fn is None:
                collate_fn = collate_batch

            if is_rnn:
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=self.num_workers,
                )
            else:
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )

            probs_list = []
            labels_list = []

            with torch.no_grad():
                for batch in dataloader:
                    if is_rnn:
                        # RNN / LSTM: batch = (sequence_tensor, labels)
                        x, t = batch
                        x = x.to(self.device)
                        labels = t.to(self.device)
                        z = model(x)
                    else:
                        # Transformer-based models
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        labels = batch["label"].to(self.device)

                        extra = batch.get("extra_features", None)
                        if extra is not None:
                            extra = extra.to(self.device)

                        if model.get_name() == "roberta_extra" and extra is not None:
                            z = model(input_ids, attention_mask, extra)
                        else:
                            z = model(input_ids, attention_mask)

                        # Softmax over logits to get probabilities
                    probs = torch.softmax(z, dim=1)

                    probs_list.append(probs.cpu().numpy())
                    labels_list.append(labels.cpu().numpy())

            probs_all = np.concatenate(probs_list)
            labels_all = np.concatenate(labels_list)

            model_probs.append(probs_all)

            # Take labels from the first model as ground truth (should be consistent across models)
            if true_labels is None:
                true_labels = labels_all

        # Stack probabilities: [num_models, N, 2]
        model_probs = np.stack(model_probs, axis=0)

        # ===== 2. Weighted soft averaging =====
        weights = weights.reshape(-1, 1, 1)  # [num_models, 1, 1]
        weighted_probs = (model_probs * weights).sum(axis=0)  # [N, 2]

        # Predicted label via argmax over ensemble probabilities
        y_pred = np.argmax(weighted_probs, axis=1)

        # ===== 3. Compute metrics =====
        acc = accuracy_score(true_labels, y_pred)
        f1 = f1_score(true_labels, y_pred)
        auc = roc_auc_score(true_labels, weighted_probs[:, 1])

        return acc, f1, auc


def prepared_data(seed: int):
    """
    Prepare train/val/test splits for text-only datasets.

    Uses a fixed random split for train/val, and a seed-dependent split
    for test to support repeated experiments with different seeds.

    Parameters
    ----------
    seed : int
        Random seed for the last split (controls which samples go into test).

    Returns
    -------
    X_test : pd.Series
        Test set texts.
    y_test : pd.Series
        Test set labels.
    test_data : list[tuple[str, int]]
        List of (text, label) pairs for the test set.
    """
    human_token = pd.read_csv("datasets/human_token.csv")
    ai_token = pd.read_csv("datasets/ai_token.csv")
    token = pd.concat([human_token, ai_token], ignore_index=True)

    X_train, X_temp, y_train, y_temp = train_test_split(
        token["text"], token["label"], test_size=0.3, random_state=42
    )
    X_val, X_test1, y_val, y_test1 = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test1, y_test1, test_size=0.6, random_state=seed
    )

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    test_data = list(zip(X_test, y_test))

    return X_test, y_test, test_data


if __name__ == "__main__":
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate different tweet detection models.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model type: rnn | lstm | bert | roberta | deberta | roberta_extra | ensemble")
    parser.add_argument("--model_dir", type=str, default="model_save",
                        help="Directory containing trained model checkpoints")
    args = parser.parse_args()

    # Load pre-trained Word2Vec model for RNN / LSTM baselines ONLY if needed
    model_w2v = None
    if args.model in ["rnn", "lstm"]:
        if not os.path.exists("datasets/w2vmodel.model"):
             print("Error: 'datasets/w2vmodel.model' not found. Please download it first.")
             exit(1)
        model_w2v = Word2Vec.load("datasets/w2vmodel.model")

    accs, f1s, aucs = [], [], []

    # Resolve model path dynamically
    if args.model != "ensemble":
        model_path = get_best_model_path(args.model, args.model_dir)
        if model_path is None:
            print(f"No checkpoint found for {args.model} in {args.model_dir}. Skipping.")
            exit(1)
        print(f"Evaluating model: {model_path}")


    # Run evaluation across multiple seeds to estimate mean ± std
    for i in range(5):
        seed = 42 + i
        set_all_seeds(seed)

        if args.model == "rnn":
            # RNN with Word2Vec embeddings
            _, _, test_data = prepared_data(seed)
            test_data_indices = convert_indices(test_data, model_w2v)
            model = MyRNN(model_w2v, hidden_size=300, num_classes=2).to(device)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

            test_evaluator = Evaluator(model, test_data_indices, device)
            acc, f1, auc = test_evaluator.accuracy()
            length = len(test_data)

        elif args.model == "lstm":
            # LSTM with Word2Vec embeddings
            _, _, test_data = prepared_data(seed)
            test_data_indices = convert_indices(test_data, model_w2v)
            model = MyLSTM(model_w2v, hidden_size=256, num_classes=2).to(device)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

            test_evaluator = Evaluator(model, test_data_indices, device)
            acc, f1, auc = test_evaluator.accuracy()
            length = len(test_data)

        elif args.model == "bert":
            # BERT classifier (bert-base-uncased)
            X_test, y_test, _ = prepared_data(seed)
            model = BertClassifier(num_labels=2, dropout=0.3, freeze_bert=False).to(device)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            test_dataset = BertDataset(X_test, y_test, tokenizer)

            test_evaluator = Evaluator(model, test_dataset, device)
            acc, f1, auc = test_evaluator.accuracy()
            length = len(test_dataset)

        elif args.model == "deberta":
            # DeBERTa-v3-large classifier
            X_test, y_test, _ = prepared_data(seed)
            model_name = "microsoft/deberta-v3-large"
            config = AutoConfig.from_pretrained(model_name)
            config.num_labels = 2

            model = DebertaV3.from_pretrained(model_name, config=config).to(device)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            test_dataset = BertDataset(X_test, y_test, tokenizer)

            test_evaluator = Evaluator(model, test_dataset, device)
            acc, f1, auc = test_evaluator.accuracy()
            length = len(test_dataset)

        elif args.model == "roberta":
            # RoBERTa-large classifier
            X_test, y_test, _ = prepared_data(seed)
            model_name = "FacebookAI/roberta-large"
            model = MyRobertaForBinaryClassification.from_pretrained(model_name).to(device)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            test_dataset = BertDataset(X_test, y_test, tokenizer)

            test_evaluator = Evaluator(model, test_dataset, device)
            acc, f1, auc = test_evaluator.accuracy()
            length = len(test_dataset)

        elif args.model == "roberta_extra":
            # RoBERTa-large with additional handcrafted features
            human_token = pd.read_csv("datasets/human_token_with_features.csv")
            ai_token = pd.read_csv("datasets/ai_token_with_features.csv")
            token = pd.concat([human_token, ai_token], ignore_index=True)

            # Fill missing feature values with 0
            token["log_mean_ppl"] = token["log_mean_ppl"].fillna(0)
            token["log_max_ppl"] = token["log_max_ppl"].fillna(0)
            token["caps_ratio"] = token["caps_ratio"].fillna(0)
            token["punc_count"] = token["punc_count"].fillna(0)
            token["emoji_count"] = token["emoji_count"].fillna(0)
            token["dash_count"] = token["dash_count"].fillna(0)

            # Build feature matrix: [log_mean_ppl, log_max_ppl, caps_ratio, punc_count, emoji_count, dash_count]
            feature_data = np.column_stack(
                (
                    token["log_mean_ppl"].to_numpy(dtype=np.float64),
                    token["log_max_ppl"].to_numpy(dtype=np.float64),
                    token["caps_ratio"].to_numpy(dtype=np.float64),
                    token["punc_count"].to_numpy(dtype=np.float64),
                    token["emoji_count"].to_numpy(dtype=np.float64),
                    token["dash_count"].to_numpy(dtype=np.float64),
                )
            )

            # Standardize features (per-feature z-score)
            feature_data = (feature_data - feature_data.mean(axis=0)) / (
                feature_data.std(axis=0) + 1e-6
            )

            # Shuffle and split text + labels + features together
            X_train, X_temp, y_train, y_temp, feat_train, feat_temp = train_test_split(
                token["text"],
                token["label"],
                feature_data,
                test_size=0.3,
                random_state=42,
            )

            X_val, X_test1, y_val, y_test1, feat_val, feat_test1 = train_test_split(
                X_temp,
                y_temp,
                feat_temp,
                test_size=0.5,
                random_state=42,
            )

            X_val, X_test, y_val, y_test, feat_val, feat_test = train_test_split(
                X_test1,
                y_test1,
                feat_test1,
                test_size=0.6,
                random_state=seed,
            )

            X_test = X_test.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

            model_name = "FacebookAI/roberta-large"
            model = Roberta_Extra.from_pretrained(model_name).to(device)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            test_dataset = FeatureDataset(X_test, y_test, feat_test, tokenizer)

            test_evaluator = Evaluator(model, test_dataset, device)
            acc, f1, auc = test_evaluator.accuracy()
            length = len(test_dataset)

        elif args.model == "ensemble":
            # ensemble over BERT + DeBERTa + RoBERTa
            X_test, y_test, _ = prepared_data(seed)

            path_bert = get_best_model_path("bert", args.model_dir)
            path_deberta = get_best_model_path("deberta", args.model_dir)
            path_roberta = get_best_model_path("roberta", args.model_dir)

            if not (path_bert and path_deberta and path_roberta):
                print("Missing models for ensemble. Need bert, deberta, roberta in model_dir.")
                exit(1)

            # BERT
            model_bert = BertClassifier(num_labels=2, dropout=0.3, freeze_bert=False).to(device)
            state_dict = torch.load(path_bert, map_location=device)
            model_bert.load_state_dict(state_dict)
            tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
            test_dataset_bert = BertDataset(X_test, y_test, tokenizer_bert)

            # DeBERTa
            model_name_deberta = "microsoft/deberta-v3-large"
            config = AutoConfig.from_pretrained(model_name_deberta)
            config.num_labels = 2
            model_deberta = DebertaV3.from_pretrained(model_name_deberta, config=config).to(device)
            state_dict = torch.load(path_deberta, map_location=device)
            model_deberta.load_state_dict(state_dict)
            tokenizer_deberta = AutoTokenizer.from_pretrained(model_name_deberta)
            test_dataset_deberta = BertDataset(X_test, y_test, tokenizer_deberta)

            # RoBERTa
            model_name_roberta = "FacebookAI/roberta-large"
            model_roberta = MyRobertaForBinaryClassification.from_pretrained(
                model_name_roberta
            ).to(device)
            state_dict = torch.load(path_roberta, map_location=device)
            model_roberta.load_state_dict(state_dict)
            tokenizer_roberta = AutoTokenizer.from_pretrained(model_name_roberta)
            test_dataset_roberta = BertDataset(X_test, y_test, tokenizer_roberta)

            models = [model_bert, model_deberta, model_roberta]
            datasets = [test_dataset_bert, test_dataset_deberta, test_dataset_roberta]
            weights = [0.3, 0.1, 0.9]  # Example: favor RoBERTa slightly more

            # For ensemble we instantiate Evaluator without a single model/dataset
            test_evaluator = Evaluator(None, None, device)
            acc, f1, auc = test_evaluator.accuracy_ensemble(
                models=models,
                datasets=datasets,
                weights=weights,
            )
            length = len(test_dataset_deberta)

        else:
            raise ValueError(f"Unsupported model type: {args.model}")

        accs.append(acc)
        f1s.append(f1)
        aucs.append(auc)

    # Aggregate results across seeds
    mean_accs = statistics.mean(accs)
    std_accs = statistics.stdev(accs)
    mean_f1s = statistics.mean(f1s)
    std_f1s = statistics.stdev(f1s)
    mean_aucs = statistics.mean(aucs)
    std_aucs = statistics.stdev(aucs)

    # Final summary
    print(f"{args.model} tested on {length} samples")
    print(f"Accuracy:  {mean_accs:.3f} ± {std_accs:.3f}")
    print(f"F1:        {mean_f1s:.3f} ± {std_f1s:.3f}")
    print(f"AUC:       {mean_aucs:.3f} ± {std_aucs:.3f}")
