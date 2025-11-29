import torch
import pandas as pd
import numpy as np
import nltk
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.tokenize import sent_tokenize

nltk.data.path.append("/home/richard8/scratch/aaas")
nltk.download('punkt', download_dir="/home/richard8/scratch/aaas")
nltk.download('punkt_tab', download_dir="/home/richard8/scratch/aaas")

MODEL_ID = "/model-weights/Qwen3-14B" 

CSV_INPUT = "/home/richard8/projects/aip-agoldenb/richard8/TweetVerify/datasets/ai_token.csv"               
CSV_OUTPUT = "/home/richard8/projects/aip-agoldenb/richard8/TweetVerify/datasets/ai_token_with_features.csv" 

print(f"Loading {MODEL_ID} in FP16...")

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print(f"Using dtype: {dtype}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,      
    device_map="auto",      
    trust_remote_code=True
)
model.eval()

def get_llama_features(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0 
    length = len(text) + 1 
    caps_ratio = sum(1 for c in text if c.isupper()) / length
    punc_count = sum(1 for c in text if c in "!?.") / length
    digit_ratio = sum(1 for c in text if c.isdigit()) / length
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    
    if len(cleaned_text) == 0:
        return 0.0, 0.0, caps_ratio, punc_count, digit_ratio
    try:
        sentences = sent_tokenize(cleaned_text)
    except:
        sentences = [cleaned_text]
    
    if len(sentences) < 1:
        return 0.0, 0.0, caps_ratio, punc_count, digit_ratio
        
    ppls = []
    with torch.no_grad():
        for sentence in sentences:
            encodings = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=2048)
            input_ids = encodings.input_ids.to(model.device)
            if input_ids.shape[1] < 2: continue

            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
            if not torch.isnan(loss):
                ppl = torch.exp(loss).item()
                if ppl < 100000: ppls.append(ppl)
    if not ppls:
        mean_ppl = 0.0
        max_ppl = 0.0
    else:
        ppls_np = np.array(ppls)
        mean_ppl = np.mean(ppls_np)
        max_ppl = np.max(ppls_np)

    log_mean_ppl = np.log1p(mean_ppl)
    log_max_ppl = np.log1p(max_ppl)
    return log_mean_ppl, log_max_ppl, caps_ratio, punc_count, digit_ratio


df = pd.read_csv(CSV_INPUT)

log_mean_ppl_list = []
log_max_ppl_list = []
caps_ratio_list = []
punc_count_list = []
digit_ratio_list = []

print("Extracting features...")
for text in tqdm(df['text']):
    log_mean_ppl, log_max_ppl, caps_ratio, punc_count, digit_ratio = get_llama_features(text)
    log_mean_ppl_list.append(log_mean_ppl)
    log_max_ppl_list.append(log_max_ppl)
    caps_ratio_list.append(caps_ratio)
    punc_count_list.append(punc_count)
    digit_ratio_list.append(digit_ratio)

df['log_mean_ppl'] = log_mean_ppl_list
df['log_max_ppl'] = log_max_ppl_list
df['caps_ratio'] = caps_ratio_list
df['punc_count'] = punc_count_list
df['digit_ratio'] = digit_ratio_list


df.to_csv(CSV_OUTPUT, index=False)
print(f"Done! Saved to {CSV_OUTPUT}")