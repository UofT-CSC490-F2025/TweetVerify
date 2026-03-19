# ==========================================================
# token_distribution.py
# Human vs AI Token Distribution Visualization
# Includes:
# 1. Token Frequency Rank Plot (Zipf)
# 2. KL Divergence Heatmap
# 3. Function Word Distribution vs Content Word Distribution
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import entropy
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

import re
from pathlib import Path
from datetime import datetime
# ------------------------------
# Config
# ------------------------------

PROJECT_DIR = Path(__file__).resolve()
for _ in range(5):
    if (PROJECT_DIR / "datalake").exists():
        break
    PROJECT_DIR = PROJECT_DIR.parent
DATA_DIR = PROJECT_DIR / "datalake" / "curated"
text_column = "text"
human_file = DATA_DIR / "twitter" / "high_quality_human.csv"
ai_file = DATA_DIR / "llm" / "ai_generated.csv"

# ------------------------------
# Load Data
# ------------------------------
print("Loading datasets...")
human_df = pd.read_csv(human_file)
ai_df = pd.read_csv(ai_file)

tokenizer = TweetTokenizer()
stop_words = set(stopwords.words("english"))

# ------------------------------
# Tokenization Helpers
# ------------------------------
def tokenize(texts):
    tokens = []
    for t in texts:
        tokens.extend(tokenizer.tokenize(str(t).lower()))
    return tokens

human_tokens = tokenize(human_df[text_column])
ai_tokens = tokenize(ai_df[text_column])

print(f"Human token count: {len(human_tokens)}")
print(f"AI token count: {len(ai_tokens)}")

# ------------------------------
# Frequency Counters
# ------------------------------
human_freq = Counter(human_tokens)
ai_freq = Counter(ai_tokens)

# restrict to shared vocabulary
vocab = list((set(human_freq) | set(ai_freq)))
top_k = 500    # use top 500 for KL/heatmap

human_vec = np.array([human_freq[w] for w in vocab])
ai_vec = np.array([ai_freq[w] for w in vocab])

# normalize
human_prob = human_vec / human_vec.sum()
ai_prob = ai_vec / ai_vec.sum()

# ------------------------------
# 1. Token Frequency Rank Plot (Zipf)
# ------------------------------
def plot_zipf(freq_counter, label, color):
    sorted_freq = np.array(sorted(freq_counter.values(), reverse=True))
    ranks = np.arange(1, len(sorted_freq) + 1)
    plt.loglog(ranks, sorted_freq, label=label, color=color)

plt.figure(figsize=(8,6))
plot_zipf(human_freq, "Human", "blue")
plot_zipf(ai_freq, "AI", "red")
plt.title("Token Frequency Rank Plot (Zipf)")
plt.xlabel("Rank (log)")
plt.ylabel("Frequency (log)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("zipf_plot.png", dpi=300)
print("Saved zipf_plot.png")

# ------------------------------
# 2. KL Divergence Heatmap
# ------------------------------

def is_bad_token(tok):
    if tok.startswith("#"): return True
    if tok.startswith("@"): return True
    if tok.startswith("http"): return True
    if tok.startswith("www"): return True
    if tok.strip() == "": return True
    if re.fullmatch(r"[\d\W]+", tok): return True  # numbers or punctuation
    return False

kl_h2a = entropy(human_prob + 1e-12, ai_prob + 1e-12)
kl_a2h = entropy(ai_prob + 1e-12, human_prob + 1e-12)

print(f"KL(Human || AI): {kl_h2a:.4f}")
print(f"KL(AI || Human): {kl_a2h:.4f}")

# compute per-token KL contributions on top-K tokens
# ------------------------------
# 2. KL Divergence Analysis (Improved)
# ------------------------------

# compute per-token KL contributions for combined top-k vocabulary
human_top = dict(human_freq.most_common(top_k))
ai_top = dict(ai_freq.most_common(top_k))

tok_list = list(set(human_top.keys()) | set(ai_top.keys()))
tok_list = [t for t in tok_list if not is_bad_token(t)]

kl_values = []
for tok in tok_list:
    p = human_top.get(tok, 1e-9) / sum(human_top.values())
    q = ai_top.get(tok, 1e-9) / sum(ai_top.values())
    kl_values.append(p * np.log((p + 1e-12) / (q + 1e-12)))

# Create KL dataframe
kl_df = pd.DataFrame({
    "token": tok_list,
    "kl_value": kl_values,
    "abs_kl": np.abs(kl_values)
})

# Sort by absolute KL divergence
kl_df = kl_df.sort_values("abs_kl", ascending=False)

# Export full KL list (optional)
kl_df.to_csv("token_kl_ranking.csv", index=False)
print("Saved token_kl_ranking.csv (sorted by KL difference)")

# Select top-N tokens for visualization
TOP_N = 40
kl_top = kl_df.head()

plt.figure(figsize=(14, 3))
sns.heatmap(
    kl_top["kl_value"].values.reshape(1, -1),
    cmap="coolwarm",
    xticklabels=kl_top["token"],
    yticklabels=["KL Contribution"]
)
plt.xticks(rotation=90)
plt.title(f"Top {TOP_N} Tokens by KL Divergence (AI vs Human)")
plt.tight_layout()
plt.savefig("kl_heatmap_top_tokens.png", dpi=300)
print("Saved kl_heatmap_top_tokens.png")

print("\n=== Top KL Difference Tokens ===")
print(kl_top[["token", "kl_value"]])


# ------------------------------
# 3. Function vs Content Word Distribution (Relative Frequency)
# ------------------------------
def count_function_content(tokens):
    func = sum(1 for t in tokens if t in stop_words)
    content = len(tokens) - func
    return func, content, len(tokens)

h_func, h_cont, h_total = count_function_content(human_tokens)
a_func, a_cont, a_total = count_function_content(ai_tokens)

# Compute proportions
h_func_prop = h_func / h_total
h_cont_prop = h_cont / h_total
a_func_prop = a_func / a_total
a_cont_prop = a_cont / a_total

df_prop = pd.DataFrame({
    "Type": ["Function Words", "Content Words"],
    "Human (%)": [h_func_prop * 100, h_cont_prop * 100],
    "AI (%)": [a_func_prop * 100, a_cont_prop * 100]
})

df_plot = df_prop.melt(id_vars="Type", var_name="Source", value_name="Percentage")

plt.figure(figsize=(7,5))
sns.barplot(data=df_plot, x="Type", y="Percentage", hue="Source")
plt.title("Function vs Content Word Distribution (Relative Frequency)")
plt.ylabel("Percentage (%)")
plt.tight_layout()
plt.savefig("function_content_distribution_relative.png", dpi=300)

print("Saved function_content_distribution_relative.png")
print(df_prop)



# ------------------------------
# A. Fine-grained Function Word Category Analysis
# ------------------------------

# Linguistic categories
PRONOUNS = set(["i","you","he","she","they","we","me","him","her","them","us","my","your","his","their","our"])
ARTICLES = set(["a","an","the"])
PREPOSITIONS = set(["in","on","at","to","for","with","from","by","about","over","under","between"])
CONJUNCTIONS = set(["and","but","or","nor","so","yet"])
AUXILIARIES = set(["am","is","are","was","were","be","been","being","have","has","had","do","does","did","would","could","should","may","might","must"])
DISCOURSE_MARKERS = set(["however","moreover","additionally","furthermore","overall","clearly","importantly","significantly"])

CATEGORIES = {
    "Pronouns": PRONOUNS,
    "Articles": ARTICLES,
    "Prepositions": PREPOSITIONS,
    "Conjunctions": CONJUNCTIONS,
    "Aux Verbs": AUXILIARIES,
    "Discourse Markers": DISCOURSE_MARKERS
}

def categorize(tokens):
    total = len(tokens)
    counts = {}
    for name, wordset in CATEGORIES.items():
        counts[name] = sum(1 for t in tokens if t in wordset) / total * 100
    return counts

h_cat = categorize(human_tokens)
a_cat = categorize(ai_tokens)

df_cat = pd.DataFrame({
    "Category": list(CATEGORIES.keys()),
    "Human (%)": [h_cat[k] for k in CATEGORIES.keys()],
    "AI (%)": [a_cat[k] for k in CATEGORIES.keys()]
})

df_melt = df_cat.melt(id_vars="Category", var_name="Source", value_name="Percentage")

plt.figure(figsize=(10,6))
sns.barplot(data=df_melt, x="Category", y="Percentage", hue="Source")
plt.title("Fine-grained Function Word Category Comparison (Relative Frequency)")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("finegrained_function_categories.png", dpi=300)

print("Saved finegrained_function_categories.png")
print(df_cat)



def is_topic_token(t):
    if t.startswith("#"): return True
    if t.startswith("@"): return True
    if t.startswith("http"): return True
    if t.startswith("www"): return True
    if re.fullmatch(r"\d+", t): return True
    if re.fullmatch(r"[^\w]+", t): return True
    if t.strip() == "": return True
    return False

# Filter topic tokens out of top-k sets
filtered_tokens = [t for t in tok_list if not is_topic_token(t)]

filtered_human_top = {t: human_top[t] for t in filtered_tokens if t in human_top}
filtered_ai_top = {t: ai_top[t] for t in filtered_tokens if t in ai_top}

# Recompute KL on filtered vocabulary
kl_vals = []
tok_filtered = list(filtered_human_top.keys() | filtered_ai_top.keys()) if isinstance(filtered_human_top.keys(), set) else list(set(filtered_human_top.keys()) | set(filtered_ai_top.keys()))

for t in tok_filtered:
    p = filtered_human_top.get(t, 1e-9) / sum(filtered_human_top.values())
    q = filtered_ai_top.get(t, 1e-9) / sum(filtered_ai_top.values())
    kl_vals.append(p * np.log((p + 1e-12) / (q + 1e-12)))

df_kl = pd.DataFrame({
    "token": tok_filtered,
    "kl_value": kl_vals,
    "abs_kl": np.abs(kl_vals)
}).sort_values("abs_kl", ascending=False)

TOP_N = 40
df_kl_top = df_kl.head(TOP_N)

plt.figure(figsize=(14,3))
sns.heatmap(
    df_kl_top["kl_value"].values.reshape(1,-1),
    cmap="coolwarm",
    xticklabels=df_kl_top["token"],
    yticklabels=["KL"]
)
plt.xticks(rotation=90)
plt.title(f"Topic-filtered KL Divergence Top {TOP_N} Tokens")
plt.tight_layout()
plt.savefig("kl_style_filtered.png", dpi=300)

print("Saved kl_style_filtered.png")
print(df_kl_top)





# ------------------------------
# C. TF-IDF Style Distance Comparison (Fixed)
# ------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA

human_texts = human_df[text_column].astype(str).tolist()
ai_texts = ai_df[text_column].astype(str).tolist()

tfidf = TfidfVectorizer(
    ngram_range=(1,2),
    min_df=5,
    max_df=0.8,
    stop_words="english"
)

X = tfidf.fit_transform(human_texts + ai_texts)

H = X[:len(human_texts)]
A = X[len(human_texts):]

# FIX: reshape centroids to 2D arrays
h_centroid = H.mean(axis=0).A.reshape(1, -1)
a_centroid = A.mean(axis=0).A.reshape(1, -1)

dist = cosine_distances(h_centroid, a_centroid)[0,0]
print(f"Cosine distance between Human and AI TF-IDF centroids: {dist:.4f}")

# PCA for visualization
pca = PCA(n_components=2)
X2d = pca.fit_transform(X.toarray())

labels = np.array(["Human"]*len(human_texts) + ["AI"]*len(ai_texts))

plt.figure(figsize=(7,6))
sns.scatterplot(x=X2d[:,0], y=X2d[:,1], hue=labels, alpha=0.4)
plt.title("TF-IDF PCA Visualization (Style Clustering)")
plt.tight_layout()
plt.savefig("tfidf_pca_style.png", dpi=300)

print("Saved tfidf_pca_style.png")
