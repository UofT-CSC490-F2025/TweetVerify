import torch
import numpy as np
from gensim.models import Word2Vec
import sys

class Predictor:
    def __init__(
        self,
        model: torch.nn.Module,
        device,
        word2vec_model_path: str = "src/w2vmodel.model"
    ):
        self.model = model
        self.device = device
        self.model_w2v = Word2Vec.load(word2vec_model_path)
        self.model.eval()  # Set model to evaluation mode
    def predict(self, text, tokenizer):
        """
        Predict whether the text is human-written or AI-generated

        Args:
            text (str): Input text to classify

        Returns:
            tuple: (prediction, confidence)
                - prediction: 0 for AI-generated, 1 for human-written
                - confidence: confidence score (0-1)
        """
        if self.model.get_name() == 'bert':
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            text_tensor = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            with torch.no_grad():
                outputs = self.model(text_tensor, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1).item()
                confidence = torch.max(probabilities).item()
            return prediction, confidence
        else:
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
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1).item()
                confidence = torch.max(probabilities).item()
            return prediction, confidence

    def predict_batch(self, texts, tokenizer):
        """
        Predict multiple texts at once

        Args:
            texts (list): List of input texts to classify
            tokenizer: Tokenizer for BERT models

        Returns:
            list: List of tuples (prediction, confidence) for each text
        """
        results = []
        for text in texts:
            pred, conf = self.predict(text, tokenizer)
            results.append((pred, conf))
        return results
