# ===== Standard Library =====
import os
import argparse
import numpy as np
# ===== Third-Party Libraries =====
import torch
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, AutoTokenizer, AutoConfig

# ===== Local Modules =====
from src.data_ingestion.twitter_db import TwitterDB
from src.data_ingestion.llm_db import LLMDB
from src.data_ingestion.main_db import MainDB
from src.data_ingestion.twitter_scrape import scrape_user_tweets, scrape_keyword_tweets
from src.data_preprocessing.processor import DataProcessor
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Model type"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Learning rate"
    )
    parser.add_argument(
        "--output_path", help="model save path"
    )#default=os.environ["SM_MODEL_DIR"]
    parser.add_argument("--batch_size", type=int, default=314, help="model batch size")
    args = parser.parse_args()
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    print(f"model_type:{args.model} batch size:{args.batch_size} learning rate:{args.learning_rate}", flush=True)
    print(f"Model will be saved to: {args.output_path}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds()
    
    model_w2v = Word2Vec.load("datasets/w2vmodel.model")
    human_token = pd.read_csv("datasets/human_token.csv")
    ai_token = pd.read_csv("datasets/ai_token.csv")
    # Combine human token and ai token
    token = pd.concat([human_token, ai_token], ignore_index=True)
    # Shuffle and split the data
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
    train_data = list(zip(X_train, y_train))
    val_data = list(zip(X_val, y_val))
    test_data = list(zip(X_test, y_test))
    # Create model based on argument
    if args.model == "rnn":
        # Convert to indices
        train_data_indices = convert_indices(train_data, model_w2v)
        val_data_indices = convert_indices(val_data, model_w2v)
        test_data_indices = convert_indices(test_data, model_w2v)
        model = MyRNN(model_w2v, hidden_size=300, num_classes=2).to(device)
        trainer = Trainer(
            device,
            model,
            train_data_indices,
            val_data_indices,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            model_save_dir=args.output_path,
            batch_size=args.batch_size,
        )
        train_loss,  val_acc = trainer.train_model()
        test_evaluator = Evaluator(model, test_data_indices, device)
        acc = test_evaluator.accuracy(args.batch_size)
    elif args.model == "lstm":
        train_data_indices = convert_indices(train_data, model_w2v)
        val_data_indices = convert_indices(val_data, model_w2v)
        test_data_indices = convert_indices(test_data, model_w2v)
        model = MyLSTM(model_w2v, hidden_size=256, num_classes=2).to(device)
        trainer = Trainer(
            device,
            model,
            train_data_indices,
            val_data_indices,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            model_save_dir=args.output_path,
            batch_size=args.batch_size,
        )
        train_loss,  val_acc = trainer.train_model()
        test_evaluator = Evaluator(model, test_data_indices, device)
        acc = test_evaluator.accuracy(args.batch_size)
    elif args.model == "bert":
        model = BertClassifier(num_labels=2, dropout=0.3, freeze_bert=False).to(device)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        train_dataset = BertDataset(X_train, y_train, tokenizer)
        val_dataset = BertDataset(X_val, y_val, tokenizer)
        test_dataset = BertDataset(X_test, y_test, tokenizer)
        trainer = Trainer(
            device,
            model,
            train_dataset,
            val_dataset,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            model_save_dir=args.output_path,
            batch_size=args.batch_size,
        )
        train_loss,  val_acc = trainer.train_model()

        test_evaluator = Evaluator(model, test_dataset, device)
        acc = test_evaluator.accuracy(args.batch_size)
    elif args.model == "deberta":
        model_name = "microsoft/deberta-v3-large"
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 2

        model = DebertaV3.from_pretrained(
            model_name,
            config=config
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_dataset = BertDataset(X_train, y_train, tokenizer)
        val_dataset = BertDataset(X_val, y_val, tokenizer)
        test_dataset = BertDataset(X_test, y_test, tokenizer)

        trainer = Trainer(
            device,
            model,
            train_dataset,
            val_dataset,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            model_save_dir=args.output_path,
            batch_size=args.batch_size,
        )
        train_loss,  val_acc = trainer.train_model()

        test_evaluator = Evaluator(model, test_dataset, device)
        acc = test_evaluator.accuracy(args.batch_size)
    elif args.model == "roberta":
        model_name = "FacebookAI/roberta-large"
        model = MyRobertaForBinaryClassification.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_dataset = BertDataset(X_train, y_train, tokenizer)
        val_dataset = BertDataset(X_val, y_val, tokenizer)
        test_dataset = BertDataset(X_test, y_test, tokenizer)

        trainer = Trainer(
            device,
            model,
            train_dataset,
            val_dataset,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            model_save_dir=args.output_path,
            batch_size=args.batch_size,
        )
        train_loss, val_acc = trainer.train_model()

        test_evaluator = Evaluator(model, test_dataset, device)
        acc = test_evaluator.accuracy(args.batch_size)
    elif args.model=='roberta_extra':
        human_token = pd.read_csv("datasets/human_token_with_features.csv")
        ai_token = pd.read_csv("datasets/ai_token_with_features.csv")
        # Combine human token and ai token
        token = pd.concat([human_token, ai_token], ignore_index=True)
        token['log_mean_ppl'] = token['log_mean_ppl'].fillna(0)
        token['log_max_ppl'] = token['log_max_ppl'].fillna(0)
        token['caps_ratio'] = token['caps_ratio'].fillna(0)
        token['punc_count'] = token['punc_count'].fillna(0)
        token['digit_ratio'] = token['digit_ratio'].fillna(0)
        feature_data = np.column_stack((
            token['log_mean_ppl'].to_numpy(dtype=np.float64), 
            token['log_max_ppl'].to_numpy(dtype=np.float64), 
            token['caps_ratio'].to_numpy(dtype=np.float64), 
            token['punc_count'].to_numpy(dtype=np.float64), 
            token['digit_ratio'].to_numpy(dtype=np.float64)
        ))
        feature_data = (feature_data - feature_data.mean(axis=0)) / (feature_data.std(axis=0) + 1e-6)
        # Shuffle and split the data
        X_train, X_temp, y_train, y_temp, feat_train, feat_temp = train_test_split(
            token["text"], 
            token["label"], 
            feature_data,  
            test_size=0.3, 
            random_state=42
        )
        
        X_val, X_test, y_val, y_test, feat_val, feat_test = train_test_split(
            X_temp, 
            y_temp, 
            feat_temp,     
            test_size=0.5, 
            random_state=42
        )

        # Reset indices to ensure continuous indexing for DataLoader
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
            device,
            model,
            train_dataset,
            val_dataset,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            model_save_dir=args.output_path,
            batch_size=args.batch_size,
        )
        train_loss, val_acc = trainer.train_model()

        test_evaluator = Evaluator(model, test_dataset, device)
        acc = test_evaluator.accuracy(args.batch_size)

    print(f"test accuracy: {acc}", flush=True)



if __name__ == "__main__":
    main()
