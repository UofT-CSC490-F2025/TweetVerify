import os

from openai import OpenAI

API_KEY = os.getenv("OPEN_AI_API_KEY")

client = OpenAI(api_key=API_KEY)
print(client.models.list())
