# Baseline Classifier Results Summary

## Experiment Configuration
- **Dataset**: 3,000 samples per class (6,000 total)
- **Train/Val/Test Split**: 70/15/15 (4,199/900/901 samples)
- **Model**: Logistic Regression on SentenceTransformer embeddings (all-MiniLM-L6-v2)
- **Date**: November 5, 2024

## Test Set Results

### Overall Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.5361** (53.61%) |
| **Precision** | **0.5336** |
| **Recall** | **0.5644** |
| **F1-Score** | **0.5486** |

### Confusion Matrix

```
                Predicted
              Human    AI
  Actual Human   229   222
          AI     196   254
```

- **True Positives (AI correctly identified)**: 254
- **True Negatives (Human correctly identified)**: 229
- **False Positives (Human misclassified as AI)**: 222
- **False Negatives (AI misclassified as Human)**: 196

### Per-Class Metrics

**Human Class:**
- Precision: 0.5388 (53.88%)
- Recall: 0.5078 (50.78%)
- F1-Score: 0.5230

**AI Class:**
- Precision: 0.5336 (53.36%)
- Recall: 0.5644 (56.44%)
- F1-Score: 0.5486

## Validation Set Results (for comparison)

| Metric | Value |
|--------|-------|
| Accuracy | 0.5733 (57.33%) |
| Precision | 0.5668 |
| Recall | 0.6222 |
| F1-Score | 0.5932 |

*Note: Validation set shows slightly better performance than test set, indicating some variance or potential overfitting.*

## Error Analysis

### Error Distribution
- **False Positives (Human→AI)**: 222 errors (24.64% of test set)
- **False Negatives (AI→Human)**: 196 errors (21.75% of test set)
- **Total Errors**: 418 out of 901 samples (46.39%)

### Quantitative Error Patterns

**False Positives (Human text classified as AI):**
- Average text length: 207.6 characters
- URLs per text: 1.20
- Hashtags per text: 5.70
- Mentions per text: 0.10

**False Negatives (AI text classified as Human):**
- Average text length: 184.6 characters
- URLs per text: 0.80
- Hashtags per text: 4.10
- Mentions per text: 0.30

### Qualitative Error Patterns

**Common False Positive Patterns (Human→AI):**
1. Formal political language with hashtags and URLs
2. News article titles or summaries
3. Structured information (e.g., "According to @CAWP_RU data...")
4. High hashtag density with political topics

**Common False Negative Patterns (AI→Human):**
1. Casual language that mimics human writing style
2. News-style content that appears natural
3. Content with emotional language
4. Tweets with minimal hashtags or structured formatting

## Key Insights

1. **Balanced Performance**: The model shows relatively balanced performance between classes (precision and recall are similar for both classes).

2. **Feature Importance**: 
   - Hashtag density doesn't clearly distinguish AI vs Human (AI has more hashtags in false negatives)
   - URL presence is similar in both error types
   - Text length is similar (193 vs 185 chars)

3. **Challenging Cases**:
   - Formal political language is often misclassified as AI
   - AI-generated content that mimics casual human style is hard to detect
   - News-style content is ambiguous regardless of source

4. **Model Limitations**:
   - Linear model (logistic regression) may not capture complex patterns
   - Embeddings may not preserve subtle stylistic differences
   - Limited training data (279 samples) may affect generalization

## Recommendations for Improvement

1. **Feature Engineering**: Add explicit features like:
   - Vocabulary diversity metrics
   - Punctuation patterns
   - Sentence structure complexity
   - Emoji usage

2. **Model Improvements**:
   - Try non-linear models (SVM with RBF kernel, Random Forest)
   - Use larger embedding models
   - Fine-tune embeddings on the task

3. **Data Improvements**:
   - Increase training data size
   - Balance classes if needed
   - Add more diverse examples

4. **Evaluation**:
   - Run on larger test set
   - Perform cross-validation
   - Test on out-of-domain data

## Files Generated

- `baseline_test_results.csv`: Full test set predictions
- `baseline_errors_false_positives.csv`: Human texts misclassified as AI
- `baseline_errors_false_negatives.csv`: AI texts misclassified as Human

