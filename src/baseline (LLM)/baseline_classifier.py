"""
Baseline classifier using logistic regression on embeddings.
"""
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Using Word2Vec fallback.")

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False


class BaselineClassifier:
    """
    Baseline classifier using logistic regression on text embeddings.
    Uses SentenceTransformer by default, falls back to Word2Vec if needed.
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 use_word2vec: bool = False,
                 w2v_model_path: str = "src/data_tokenize/results/w2vmodel.model"):
        """
        Initialize baseline classifier.
        
        Args:
            embedding_model: SentenceTransformer model name
            use_word2vec: Whether to use Word2Vec instead of SentenceTransformer
            w2v_model_path: Path to Word2Vec model (if using Word2Vec)
        """
        self.use_word2vec = use_word2vec
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        
        if use_word2vec:
            if not GENSIM_AVAILABLE:
                raise ImportError("gensim required for Word2Vec embeddings")
            if not os.path.exists(w2v_model_path):
                raise FileNotFoundError(f"Word2Vec model not found: {w2v_model_path}")
            self.w2v_model = Word2Vec.load(w2v_model_path)
            self.embedding_model = None
            print(f"[baseline] Using Word2Vec model from {w2v_model_path}")
        else:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")
            self.embedding_model = SentenceTransformer(embedding_model)
            self.w2v_model = None
            print(f"[baseline] Using SentenceTransformer: {embedding_model}")
    
    def _get_embedding_w2v(self, text: str) -> np.ndarray:
        """Get embedding using Word2Vec (average of word vectors)."""
        import nltk
        from nltk.tokenize import word_tokenize
        
        try:
            tokens = word_tokenize(text.lower())
        except LookupError:
            nltk.download('punkt', quiet=True)
            tokens = word_tokenize(text.lower())
        
        # Get word vectors
        vectors = []
        for token in tokens:
            if token in self.w2v_model.wv:
                vectors.append(self.w2v_model.wv[token])
        
        if len(vectors) == 0:
            # Return zero vector if no words found
            return np.zeros(self.w2v_model.wv.vector_size)
        
        return np.mean(vectors, axis=0)
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings
        
        Returns:
            Array of embeddings (n_samples, embedding_dim)
        """
        if self.use_word2vec:
            embeddings = np.array([self._get_embedding_w2v(str(text)) for text in texts])
        else:
            embeddings = self.embedding_model.encode(
                [str(text) for text in texts],
                show_progress_bar=True,
                convert_to_numpy=True
            )
        
        return embeddings
    
    def fit(self, train_texts: List[str], train_labels: List[int]):
        """
        Train the classifier.
        
        Args:
            train_texts: List of training texts
            train_labels: List of training labels (0=Human, 1=AI)
        """
        print("[baseline] Computing embeddings for training data...")
        train_embeddings = self._get_embeddings(train_texts)
        
        print("[baseline] Scaling embeddings...")
        train_embeddings_scaled = self.scaler.fit_transform(train_embeddings)
        
        print("[baseline] Training logistic regression...")
        self.classifier.fit(train_embeddings_scaled, train_labels)
        print("[baseline] Training complete!")
    
    def predict(self, texts: List[str]) -> List[int]:
        """
        Predict labels for texts.
        
        Args:
            texts: List of texts to classify
        
        Returns:
            List of predictions (0=Human, 1=AI)
        """
        print("[baseline] Computing embeddings for prediction...")
        embeddings = self._get_embeddings(texts)
        
        print("[baseline] Scaling embeddings...")
        embeddings_scaled = self.scaler.transform(embeddings)
        
        print("[baseline] Making predictions...")
        predictions = self.classifier.predict(embeddings_scaled)
        
        return predictions.tolist()
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            texts: List of texts to classify
        
        Returns:
            Array of probabilities (n_samples, n_classes)
        """
        embeddings = self._get_embeddings(texts)
        embeddings_scaled = self.scaler.transform(embeddings)
        return self.classifier.predict_proba(embeddings_scaled)
    
    def predict_df(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """
        Predict on a dataframe.
        
        Args:
            df: Dataframe with text column
            text_col: Name of text column
        
        Returns:
            Dataframe with added 'prediction' column
        """
        texts = df[text_col].tolist()
        predictions = self.predict(texts)
        
        result_df = df.copy()
        result_df['prediction'] = predictions
        return result_df


