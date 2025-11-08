import os
from transformers import AutoTokenizer, AutoModel


if __name__ == "__main__":

    model = AutoModel.from_pretrained(
        "Qwen/Qwen3-7B-Instruct",
        token=os.environ["HF_TOKEN"],
        trust_remote_code=True
    )
    print("Loaded:", model.config.hidden_size)
