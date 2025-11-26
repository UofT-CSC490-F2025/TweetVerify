import torch
import pandas as pd
import numpy as np
import nltk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.tokenize import sent_tokenize

nltk.data.path.append("/home/richard8/scratch/aaas")
nltk.download('punkt', download_dir="/home/richard8/scratch/aaas")
nltk.download('punkt_tab', download_dir="/home/richard8/scratch/aaas")

MODEL_ID = "/model-weights/Qwen3-14B" 

CSV_INPUT = "/home/richard8/projects/aip-agoldenb/richard8/TweetVerify/datasets/human_token.csv"               
CSV_OUTPUT = "/home/richard8/projects/aip-agoldenb/richard8/TweetVerify/datasets/human_token_with_features.csv" 

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
        return 0.0, 0.0
    
    sentences = sent_tokenize(text)
    if len(sentences) < 1:
        return 0.0, 0.0
        
    ppls = []
    with torch.no_grad():
        for sentence in sentences:
            encodings = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=2048)
            input_ids = encodings.input_ids.to(model.device)
            
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
            if not torch.isnan(loss):
                ppl = torch.exp(loss).item()
            
                if ppl < 50000: 
                    ppls.append(ppl)
    
    if not ppls:
        return 0.0, 0.0
        
    mean_ppl = np.mean(ppls)
    burstiness = np.std(ppls)
    return mean_ppl, burstiness


df = pd.read_csv(CSV_INPUT)

ppl_list = []
burst_list = []

print("Extracting features...")
for text in tqdm(df['text']):
    p, b = get_llama_features(text)
    ppl_list.append(p)
    burst_list.append(b)

df['ppl'] = ppl_list
df['burstiness'] = burst_list


df.to_csv(CSV_OUTPUT, index=False)
print(f"Done! Saved to {CSV_OUTPUT}")