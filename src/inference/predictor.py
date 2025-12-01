import torch
import numpy as np
from gensim.models import Word2Vec
import sys


class Predictor:
    def __init__(
        self,
        model: torch.nn.Module,
        device,
        tokenizer,
        word2vec_model_path: str = "datasets/w2vmodel.model",
    ):
        self.model = model
        self.device = device
        self.tokenizer=tokenizer
        self.model_w2v = Word2Vec.load(word2vec_model_path)
        self.model.eval()  # Set model to evaluation mode

    def predict(self, text):
        """
        Predict whether the text is human-written or AI-generated

        Args:
            text (str): Input text to classify (already cleaned)

        Returns:
            tuple: (prediction, confidence)
                - prediction: 0 for human-written, 1 for AI-generated
                - confidence: confidence score (0-1)
        """
        model_name = self.model.get_name()

        # ==== RNN / LSTM branch (Word2Vec indices) ====
        if model_name in ["lstm", "rnn"]:
            words = text.lower().split()
            indices = []
            for word in words:
                if word in self.model_w2v.wv.key_to_index:
                    # Shift indices up by one since the padding token is at index 0
                    word_index = self.model_w2v.wv.key_to_index.get(word)
                    if word_index is not None:
                        indices.append(word_index + 1)
                    else:
                        indices.append(0)
                else:
                    indices.append(0)  # Unknown word -> padding token

            # Convert to tensor and add batch dimension
            text_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(text_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1).item()
                confidence = torch.max(probabilities).item()
            return prediction, confidence

        # ==== Transformer branch (BERT / RoBERTa / DeBERTa / RoBERTa_Extra) ====
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        text_tensor = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        with torch.no_grad():
            # Models that accept extra handcrafted features (e.g., roberta_extra)
            # default to a zero-vector when extra_features=None, so we can safely
            # call them with only (input_ids, attention_mask) at inference time.
            outputs = self.model(text_tensor, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = torch.max(probabilities).item()
        return prediction, confidence

    def predict_batch(self, texts):
        """
        Predict multiple texts at once

        Args:
            texts (list): List of input texts to classify

        Returns:
            list: List of tuples (prediction, confidence) for each text
        """
        results = []
        model_name = self.model.get_name()

        with torch.no_grad():
            # ==== RNN / LSTM batch prediction (Word2Vec indices) ====
            if model_name in ["lstm", "rnn"]:
                device = self.device
                all_indices = []
                max_len = 128  # optional padding length, can adjust

                for text in texts:
                    words = text.lower().split()
                    indices = [
                        (
                            self.model_w2v.wv.key_to_index.get(w, -1) + 1
                            if w in self.model_w2v.wv.key_to_index
                            else 0
                        )
                        for w in words
                    ]
                    # pad or truncate
                    indices = indices[:max_len] + [0] * (max_len - len(indices))
                    all_indices.append(indices)

                text_tensor = torch.tensor(all_indices, dtype=torch.long).to(device)
                outputs = self.model(text_tensor)
                probs = torch.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, dim=1)
                results = list(zip(preds.cpu().tolist(), confs.cpu().tolist()))
                return results

            # ==== Transformer batch prediction (BERT / RoBERTa / DeBERTa / RoBERTa_Extra) ====
            encoding = self.tokenizer(
                list(texts),
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt",
            )
            text_tensor = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            outputs = self.model(text_tensor, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)
            results = list(zip(preds.cpu().tolist(), confs.cpu().tolist()))

        return results
