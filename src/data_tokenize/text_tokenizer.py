import os
import pandas as pd
import nltk
from nltk.tokenize.casual import TweetTokenizer
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec
import ssl

# Fix SSL certificate issue for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# ---------- Config ----------
HUMAN_PATH = "src/data_tokenize/datasets/high_quality_human.csv"
AI_PATH    = "src/data_tokenize/datasets/ai_generated.csv"

OUT_DIR = "src/data_tokenize/results"
HUMAN_TOKEN_PATH          = os.path.join(OUT_DIR, "human_token.csv")
AI_TOKEN_PATH             = os.path.join(OUT_DIR, "ai_token.csv")
HUMAN_TOKEN_INDEX_PATH    = os.path.join(OUT_DIR, "human_token_index.csv")
AI_TOKEN_INDEX_PATH       = os.path.join(OUT_DIR, "ai_token_index.csv")
W2V_MODEL_PATH            = os.path.join(OUT_DIR, "w2vmodel.model")

# Word2Vec params
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_SG = 1             # 1 = skip-gram, 0 = CBOW
W2V_EPOCHS = 10
W2V_NEGATIVE = 10
W2V_WORKERS = max(1, (os.cpu_count() or 1))
W2V_SEED = 34

# ---------- Helpers ----------
def _ensure_stopwords():
    try:
        _ = stopwords.words('english')
    except LookupError:
        nltk.download('stopwords', quiet=True)

def _load_and_validate():
    """
    Loads both CSVs and validates minimal schema & labels.
    Expects columns: ['text', 'label'].
    AI = 1, Human = 0 (as specified by the user).
    """
    if not os.path.exists(HUMAN_PATH):
        raise FileNotFoundError(f"Missing file: {HUMAN_PATH}")
    if not os.path.exists(AI_PATH):
        raise FileNotFoundError(f"Missing file: {AI_PATH}")

    human_df = pd.read_csv(HUMAN_PATH)
    ai_df = pd.read_csv(AI_PATH)

    # Basic schema checks
    for name, df in [("human", human_df), ("ai", ai_df)]:
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{name} dataset must have columns ['text', 'label'].")
        # Drop NAs and coerce label to int
        df.dropna(subset=["text"], inplace=True)
        df["label"] = df["label"].astype(int)

    # Validate labels per your new convention
    if not set(human_df["label"].unique()).issubset({0}):
        raise ValueError("Human dataset must have label = 0.")
    if not set(ai_df["label"].unique()).issubset({1}):
        raise ValueError("AI dataset must have label = 1.")

    return human_df[["text", "label"]], ai_df[["text", "label"]]

def process_data():
    """
    Loads, validates, and returns the human and AI dataframes.
    (No sampling by default; you can add sampling if desired.)
    """
    human_df, ai_df = _load_and_validate()
    print(f"[process_data] human rows: {len(human_df)}, ai rows: {len(ai_df)}")
    return human_df, ai_df

def baseline_tokenize(human_df: pd.DataFrame, ai_df: pd.DataFrame):
    """
    Tokenizes text, removes English stopwords, saves tokenized CSVs (with & without index),
    and returns combined list of token lists for Word2Vec (shuffled).
    """
    _ensure_stopwords()
    tokenizer = TweetTokenizer()
    stop_words = set(stopwords.words('english'))

    def _tokenize_series(text_series: pd.Series):
        out = []
        for text in text_series:
            tokens = tokenizer.tokenize(str(text))
            filtered = [w for w in tokens if w.lower() not in stop_words]
            out.append(filtered)
        return out

    print("[tokenize] tokenizing AI...")
    ai_tokens = _tokenize_series(ai_df["text"])
    print("[tokenize] tokenizing Human...")
    human_tokens = _tokenize_series(human_df["text"])

    # Build full tokenized dataset with labels
    df_all = pd.concat([
        pd.DataFrame({"text": human_tokens, "label": 0}),
        pd.DataFrame({"text": ai_tokens, "label": 1})
    ], ignore_index=True)

    # Shuffle before saving
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split back for separate saves (if you want to preserve human/AI csvs)
    df_human = df_all[df_all["label"] == 0]
    df_ai = df_all[df_all["label"] == 1]

    # Save tokenized CSVs
    df_human.to_csv(HUMAN_TOKEN_PATH, index=False)
    df_ai.to_csv(AI_TOKEN_PATH, index=False)
    df_human.to_csv(HUMAN_TOKEN_INDEX_PATH, index=True)
    df_ai.to_csv(AI_TOKEN_INDEX_PATH, index=True)

    print(f"[tokenize] saved shuffled tokenized CSVs:\n - {HUMAN_TOKEN_PATH}\n - {AI_TOKEN_PATH}")

    total_sentences = df_all["text"].tolist()
    print(f"[tokenize] total shuffled sentences for Word2Vec: {len(total_sentences)}")
    return total_sentences


def train_w2v(total_sentences):
    """
    Trains and saves a Word2Vec model on tokenized sentences.
    """
    if not total_sentences:
        raise ValueError("No sentences provided to train Word2Vec.")

    print(f"[w2v] training Word2Vec on {len(total_sentences)} sentences...")
    model = Word2Vec(
        sentences=total_sentences,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        sg=W2V_SG,
        hs=0,
        epochs=W2V_EPOCHS,
        negative=W2V_NEGATIVE,
        workers=W2V_WORKERS,
        seed=W2V_SEED
    )
    # Explicit train call is optional when `sentences` given, but harmless:
    model.train(total_sentences, total_examples=len(total_sentences), epochs=model.epochs)
    model.save(W2V_MODEL_PATH)
    print(f"[w2v] model saved to {W2V_MODEL_PATH}")

def main():
    human_df, ai_df = process_data()
    total_sentences = baseline_tokenize(human_df, ai_df)
    train_w2v(total_sentences)

if __name__ == "__main__":
    main()