"""
LLM-based classifier using Hugging Face Transformers (local, no API key required).
"""
import os
from typing import List, Optional
import pandas as pd

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class LLMClassifierHF:
    """
    Classifier using Hugging Face models (runs locally, no API key needed).
    """
    
    def __init__(self, 
                 model_name: str = "distilgpt2"):
        """
        Initialize LLM classifier with Hugging Face model.
        
        Args:
            model_name: Hugging Face model name or path
                       Options: "distilgpt2", "gpt2", "microsoft/DialoGPT-small"
                       Default: "distilgpt2" (smaller, faster)
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "transformers library not available. Install with: pip install transformers torch"
            )
        
        self.model_name = model_name
        
        print(f"[LLM-HF] Loading model: {model_name} (this may take a minute for first run)...")
        print(f"[LLM-HF] Note: This runs locally without API key!")
        
        try:
            # Use text-generation pipeline - simpler and works well
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1,
                max_length=100,
                do_sample=False,
                pad_token_id=None
            )
            print(f"[LLM-HF] Model loaded successfully")
        except Exception as e:
            print(f"[LLM-HF] Pipeline failed: {e}")
            print(f"[LLM-HF] Falling back to heuristic-based classification")
            self.pipeline = None
        
        # Classification prompt template
        self.prompt_template = """Is this tweet AI-generated or human-written? Answer: "{text}" ->"""
    
    def _classify_single(self, text: str) -> str:
        """
        Classify a single text using the model.
        
        Args:
            text: Text to classify
        
        Returns:
            "AI" or "Human"
        """
        # If pipeline not available, use heuristics
        if not self.pipeline:
            return self._heuristic_classify(text)
        
        try:
            # Truncate text to reasonable length
            text_truncated = text[:150]
            prompt = self.prompt_template.format(text=text_truncated)
            
            # Generate continuation
            result = self.pipeline(
                prompt,
                max_new_tokens=10,
                return_full_text=False,
                num_return_sequences=1,
                pad_token_id=self.pipeline.tokenizer.eos_token_id
            )[0]['generated_text'].strip().lower()
            
            # Parse result - look for keywords
            if any(word in result for word in ["ai", "artificial", "generated", "machine"]):
                return "AI"
            elif any(word in result for word in ["human", "person", "written", "real"]):
                return "Human"
            else:
                # Fallback to heuristics
                return self._heuristic_classify(text)
                
        except Exception as e:
            # Fallback to heuristics on error
            return self._heuristic_classify(text)
    
    def _heuristic_classify(self, text: str) -> str:
        """
        Fallback heuristic classification based on text features.
        
        Args:
            text: Text to classify
        
        Returns:
            "AI" or "Human"
        """
        text_lower = text.lower()
        
        # Heuristics for AI-generated text
        ai_indicators = [
            len(text) > 200,  # Long texts
            text.count('#') > 5,  # Many hashtags
            text.count('http') > 1,  # Multiple URLs
            not any(c in text_lower for c in ['!', '?', '...']),  # No emotional punctuation
        ]
        
        # Heuristics for human-written text
        human_indicators = [
            'lol' in text_lower or 'omg' in text_lower,  # Casual language
            text.count('!') > 2,  # Emotional punctuation
            len(text) < 100,  # Short casual texts
            '@' in text and len(text.split()) < 20,  # Mentions with short text
        ]
        
        ai_score = sum(ai_indicators)
        human_score = sum(human_indicators)
        
        return "AI" if ai_score > human_score else "Human"
    
    def predict(self, texts: List[str], show_progress: bool = True) -> List[int]:
        """
        Classify a list of texts.
        
        Args:
            texts: List of texts to classify
            show_progress: Whether to show progress
        
        Returns:
            List of predictions (0 for Human, 1 for AI)
        """
        predictions = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{total} ({100 * (i + 1) / total:.1f}%)")
            
            result = self._classify_single(str(text))
            predictions.append(1 if result == "AI" else 0)
        
        return predictions
    
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

