import pandas as pd
from gensim.models import Word2Vec

df1 = pd.read_csv("datasets/ai_token.csv")
df2 = pd.read_csv("datasets/human_token.csv")
df = pd.concat([df1, df2], ignore_index=True)
texts = df["text"].dropna().astype(str).tolist()
sentences = [t.lower().split() for t in texts]
model_w2v = Word2Vec(
    sentences=sentences,  # 输入句子列表
    vector_size=100,      # 向量维度
    window=5,             # 上下文窗口大小
    min_count=2,          # 忽略出现少于2次的词
    workers=4,            # 并行线程数
    sg=1,                 # 1=skip-gram, 0=CBOW
    seed=42               # 保证结果可复现
)
model_w2v.save("datasets/w2vmodel_new.model")
print("✅ 模型已保存到 datasets/w2vmodel_new.model")