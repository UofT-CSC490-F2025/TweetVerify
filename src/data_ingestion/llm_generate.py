import os
from openai import OpenAI
from llm_db import LLMDB  # reuse existing processor

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_posts(prompts, model="gpt-4o-mini", temperature=0.7):
    llm_records = []
    for i, prompt in enumerate(prompts):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=120
        )
        text = resp.choices[0].message.content
        llm_records.append({
            "text_id": f"gen_{i}",
            "text": text,
            "label": 1,  # ai
            "prompt_name": f"prompt_{i}",
            "model": model,
            "temperature": temperature,
        })
    return llm_records

if __name__ == "__main__":
    prompts = [
        "Write a short tweet about AI and creativity.",
        "Compose a sarcastic tweet about exams.",
        "Generate a tweet mixing emojis and hashtags about sports."
    ]
    records = generate_posts(prompts)
    db = LLMDB(records)
    path = db.process_llm_dataset()
    print("LLM data saved at:", path)
