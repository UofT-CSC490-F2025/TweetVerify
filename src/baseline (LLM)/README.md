# Baseline LLM Classifier

This module implements a baseline classifier for detecting AI-generated vs human-written tweets, using two approaches:

1. **LLM Classifier**: Uses OpenAI's GPT models with prompting for classification
2. **Baseline Classifier**: Uses logistic regression on text embeddings (SentenceTransformer or Word2Vec)

## Requirements

Install the required dependencies:

```bash
pip install openai pandas scikit-learn sentence-transformers matplotlib seaborn gensim nltk
```

For Word2Vec baseline, ensure you have trained the Word2Vec model first (run `src/data_tokenize/text_tokenizer.py`).

## Setup

### API Key (Optional - Only for OpenAI)

**Option 1: Use Hugging Face models (FREE, no API key needed!)**
```bash
# Just use the --use-hf flag - no setup needed!
python3 main.py --use-hf --skip-baseline
```

**Option 2: Use OpenAI (requires API key)**
Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or pass it via command line argument `--api-key`.

## Usage

### Basic Usage

Run both classifiers on the full dataset:

```bash
cd "src/baseline (LLM)"
python main.py
```

### Options

```bash
python main.py --help
```

Key options:
- `--max-samples N`: Limit to N samples per class (for faster testing)
- `--llm-model MODEL`: Choose OpenAI model (default: gpt-3.5-turbo)
- `--skip-llm`: Skip LLM classifier (only run baseline)
- `--skip-baseline`: Skip baseline classifier (only run LLM)
- `--use-word2vec`: Use Word2Vec instead of SentenceTransformer for baseline

### Examples

```bash
# Quick test with limited samples
python main.py --max-samples 100

# Use Hugging Face model (FREE, no API key!)
python3 main.py --use-hf --llm-model distilgpt2 --skip-baseline

# Use GPT-4 instead of GPT-3.5 (requires API key)
python3 main.py --llm-model gpt-4

# Only run baseline classifier
python3 main.py --skip-llm

# Use Word2Vec for baseline embeddings
python main.py --use-word2vec --skip-llm
```

## Output

Results are saved in `src/baseline (LLM)/results/`:

- `llm_<model>_test_results.csv`: LLM predictions on test set
- `baseline_test_results.csv`: Baseline predictions on test set
- `llm_<model>_errors.csv`: Error analysis for LLM
- `baseline_errors.csv`: Error analysis for baseline
- `model_comparison.csv`: Side-by-side comparison of metrics

## Metrics

**Primary Metric**: F1-Score (harmonic mean of precision and recall)

The evaluation includes comprehensive metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for AI class (true positives / (true positives + false positives))
- **Recall**: Recall for AI class (true positives / (true positives + false negatives))
- **F1-Score**: F1 score for AI class (2 * precision * recall / (precision + recall))
- **Confusion Matrix**: Detailed breakdown of predictions
- **Per-Class Metrics**: Precision and recall for both Human and AI classes

All metrics are computed on the test set, with validation set used for model selection and hyperparameter tuning.

## Error Analysis

The evaluator provides both qualitative and quantitative error analysis:

### Qualitative Analysis
- Examples of false positives (Human text classified as AI)
- Examples of false negatives (AI text classified as Human)

### Quantitative Analysis
- Average text length for error cases
- URL, hashtag, and mention frequency in error cases
- Error type distribution

## Limitations

### LLM Classifier
1. **Cost**: OpenAI API calls can be expensive for large datasets
2. **Latency**: Sequential API calls are slow (rate limiting)
3. **API Dependencies**: Requires internet connection and API key
4. **Prompt Sensitivity**: Performance depends heavily on prompt design
5. **Non-deterministic**: May produce different results on same input (even with temperature=0)

### Baseline Classifier
1. **Embedding Quality**: Limited by the quality of pre-trained embeddings
2. **Linear Model**: Logistic regression can only learn linear decision boundaries
3. **Feature Engineering**: No explicit feature engineering beyond embeddings
4. **Word2Vec Limitations**: Word2Vec averages may lose sentence-level semantics

### General Limitations
1. **Dataset Bias**: Performance depends on training data quality and distribution
2. **Class Imbalance**: May need balancing if classes are imbalanced
3. **Domain Specificity**: May not generalize to other text types beyond tweets
4. **Evaluation Metrics**: Binary metrics may not capture all nuances

## Next Steps

### Short-term Improvements
1. **Better Prompting**: Experiment with few-shot examples, chain-of-thought, or structured prompts
2. **Ensemble Methods**: Combine LLM and baseline predictions
3. **Feature Engineering**: Add explicit features (length, punctuation, vocabulary diversity)
4. **Hyperparameter Tuning**: Optimize regularization, learning rate, etc.
5. **Error Analysis**: Deep dive into specific error patterns

### Medium-term Improvements
1. **Fine-tuning**: Fine-tune a smaller LLM on the task
2. **Advanced Baselines**: Try SVM, Random Forest, or neural networks on embeddings
3. **Cross-validation**: Implement k-fold cross-validation for robust evaluation
4. **Confidence Scores**: Add confidence/uncertainty estimates
5. **Active Learning**: Select samples for annotation based on uncertainty

### Long-term Improvements
1. **Custom Model**: Train a dedicated model for this task
2. **Multi-modal Features**: Incorporate metadata, timing, user features
3. **Interpretability**: Add SHAP values or attention visualization
4. **Robustness Testing**: Test on adversarial examples and out-of-domain data
5. **Production Pipeline**: Optimize for inference speed and cost

## Architecture

```
baseline (LLM)/
├── data_loader.py          # Data loading and splitting
├── llm_classifier.py       # LLM-based classifier
├── baseline_classifier.py  # Logistic regression baseline
├── evaluator.py            # Metrics and error analysis
├── main.py                 # Main script
└── README.md              # This file
```

## Common Mistakes Analysis

The error analysis identifies several common failure patterns:

1. **Formal vs Casual**: LLM may classify overly formal human text as AI
2. **Short Texts**: Both models struggle with very short texts (< 50 chars)
3. **URLs and Hashtags**: High frequency of social media features in errors
4. **Context Missing**: Models may miss context clues in tweets
5. **Paraphrasing**: AI-generated paraphrases of human text are particularly challenging

See the error analysis CSVs for detailed examples.

