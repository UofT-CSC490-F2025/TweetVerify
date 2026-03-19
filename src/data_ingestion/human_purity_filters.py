# ==========================================================
# human_purity_filters.py
# Human Purity Filtering Pipeline
# Includes:
# 1. Perplexity Filtering (GPT-2)
# 2. Cross-model Verification (BERT + RoBERTa + DeBERTa)
# 3. Embedding Clustering (Sentence-BERT + Isolation Forest)
# ==========================================================

import pandas as pd
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.ensemble import IsolationForest
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Config
# ------------------------------
data_file = "path/to/raw_political_tweets.csv"
text_column = "text"

# ------------------------------
# Load Data
# ------------------------------
df = pd.read_csv(data_file)
texts = df[text_column].astype(str).tolist()

# ==========================================================
# 1. Perplexity Filtering
# ==========================================================
print("Loading GPT-2 for perplexity...")
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")

def calc_ppl(sentence):
    enc = gpt2_tok(sentence, return_tensors="pt")
    with torch.no_grad():
        out = gpt2(**enc, labels=enc["input_ids"])
    return torch.exp(out.loss).item()

print("Computing perplexity...")
ppl_values = [calc_ppl(t) for t in texts]
df["perplexity"] = ppl_values

# threshold typically: AI < 15 or < 20
ppl_threshold = df["perplexity"].quantile(0.10)
df["ppl_flag"] = df["perplexity"] < ppl_threshold

print(f"PPL threshold = {ppl_threshold:.2f}")

# ==========================================================
# 2. Cross-model Verification
# ==========================================================
models = {
    "bert": "roberta-base-openai-detector",
    "roberta": "roberta-base",
    "deberta": "microsoft/deberta-base"
}

def load_clf(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tok, mod

detectors = {name: load_clf(m) for name, m in models.items()}

def model_predict(tokenizer, model, text):
    inp = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        out = model(**inp).logits
    prob = out.softmax(dim=-1)[0,1].item()  # probability of "AI"
    return prob

print("Running cross-model predictions...")
ai_scores = []

for t in texts:
    scores = []
    for tok, mod in detectors.values():
        scores.append(model_predict(tok, mod, t))
    ai_scores.append(np.mean(scores))

df["ai_prob"] = ai_scores
df["cross_flag"] = df["ai_prob"] > 0.7    # threshold can tune

# ==========================================================
# 3. Embedding Clustering (Sentence-BERT + Isolation Forest)
# ==========================================================
print("Loading Sentence-BERT for embeddings...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding embeddings...")
embeddings = embedder.encode(texts, show_progress_bar=True)
df["embedding"] = embeddings.tolist()

print("Running PCA + IsolationForest...")
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

iso = IsolationForest(contamination=0.10)
cluster_flag = iso.fit_predict(embeddings)
df["cluster_flag"] = (cluster_flag == -1)

# visualization
plt.figure(figsize=(7,6))
sns.scatterplot(x=reduced[:,0], y=reduced[:,1], hue=df["cluster_flag"], palette=["blue","red"])
plt.title("Embedding Clustering (Anomaly Detection)")
plt.tight_layout()
plt.savefig("embedding_cluster.png", dpi=300)
print("Saved embedding_cluster.png")

# ==========================================================
# Combine all filters
# ==========================================================
df["suspect"] = df[["ppl_flag","cross_flag","cluster_flag"]].sum(axis=1) >= 2

print(df[["text","perplexity","ai_prob","ppl_flag","cross_flag","cluster_flag","suspect"]].head())

df.to_csv("filtered_tweets.csv", index=False)
print("Saved filtered_tweets.csv")
