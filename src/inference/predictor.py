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
            text (str): Input text to classify

        Returns:
            tuple: (prediction, confidence)
                - prediction: 0 for AI-generated, 1 for human-written
                - confidence: confidence score (0-1)
        """
        if self.model.get_name() in ["lstm", "rnn"]:
            words = text.split()
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
                prediction = torch.argmax(outputs, dim=1).item()
                confidence = outputs[:,1]
            return prediction, confidence
        else:
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
                outputs = self.model(text_tensor, attention_mask)
                prediction = torch.argmax(outputs, dim=1).item()
                confidence = outputs[:,1]
            return prediction, confidence

    def predict_batch(self, texts):
        """
        Predict multiple texts at once

        Args:
            texts (list): List of input texts to classify
            tokenizer: Tokenizer for BERT models

        Returns:
            list: List of tuples (prediction, confidence) for each text
        """
        results = []

        text_tensor = []
        attention_mask = []
        with torch.no_grad():
            if self.model.get_name() == "bert":
                # ---- BERT batch prediction ----
                device = self.device

                for text in texts:
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        padding="max_length",
                        max_length=128,
                        return_tensors="pt",
                    )
                    text_tensor.append(encoding["input_ids"])
                    attention_mask.append(encoding["attention_mask"])
                text_tensor = torch.cat(text_tensor, dim=0).to(device)
                attention_mask = torch.cat(attention_mask, dim=0).to(device)
                outputs = self.model(text_tensor, attention_mask)
                confs, preds = torch.max(outputs, dim=1)

                results.extend(list(zip(preds.cpu().tolist(), confs.cpu().tolist())))

            else:
                # ---- Word2Vec + classifier batch prediction ----
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
                confs, preds = torch.max(outputs, dim=1)
                results = list(zip(preds.cpu().tolist(), confs.cpu().tolist()))
        return results
