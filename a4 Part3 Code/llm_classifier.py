"""
LLM-based classifier using OpenAI API with prompting.
"""
import os
import time
import openai
from typing import List, Optional
import pandas as pd


class LLMClassifier:
    """
    Classifier using LLM with prompting for AI-generated vs Human text detection.
    """
    
    def __init__(self, 
                 model: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 temperature: float = 0.0,
                 max_retries: int = 3,
                 delay: float = 0.1):
        """
        Initialize LLM classifier.
        
        Args:
            model: OpenAI model to use (e.g., 'gpt-3.5-turbo', 'gpt-4')
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            temperature: Sampling temperature (0.0 for deterministic)
            max_retries: Maximum retries for API calls
            delay: Delay between API calls (seconds)
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.delay = delay
        
        # Set up API key
        api_key_to_use = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key_to_use:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY env var or pass api_key.")
        
        self.client = openai.OpenAI(api_key=api_key_to_use)
        
        # System prompt
        self.system_prompt = """You are an expert at detecting AI-generated text. 
Your task is to classify tweets as either AI-generated or human-written.

Consider these indicators of AI-generated text:
- Overly formal or polished language
- Lack of personal voice or authenticity
- Repetitive patterns or structures
- Unusual word choices or phrasing
- Too perfect grammar with no casual errors
- Generic or formulaic content

Consider these indicators of human-written text:
- Casual language and slang
- Personal voice and authenticity
- Natural errors or typos
- Emotional expressions
- Unique phrasing
- Personal opinions or experiences

Respond with ONLY a single word: "AI" or "Human".
"""
    
    def _classify_single(self, text: str) -> str:
        """
        Classify a single text using LLM.
        
        Args:
            text: Text to classify
        
        Returns:
            "AI" or "Human"
        """
        user_prompt = f"Classify this tweet as AI-generated or Human-written:\n\n{text}"
        
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.delay)  # Rate limiting
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=10
                )
                
                result = response.choices[0].message.content.strip().upper()
                
                # Parse response
                if "AI" in result or "1" in result:
                    return "AI"
                elif "HUMAN" in result or "0" in result:
                    return "Human"
                else:
                    # Default fallback
                    print(f"Warning: Unexpected response '{result}', defaulting to 'Human'")
                    return "Human"
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"API error (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed after {self.max_retries} attempts: {e}")
                    return "Human"  # Default fallback
    
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

