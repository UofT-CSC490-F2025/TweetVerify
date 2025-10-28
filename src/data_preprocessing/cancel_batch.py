import os

from openai import OpenAI

API_KEY = os.getenv("OPEN_AI_API_KEY")

client = OpenAI(api_key=API_KEY)

batches = client.batches.list(limit=100)
for b in batches.data:
    if b.status in ("validating", "running"):
        print(f"🛑 Cancelling batch {b.id} ({b.status})...")
        client.batches.cancel(b.id)
