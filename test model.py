import os
import requests
from flask.cli import load_dotenv

load_dotenv("HF_TOKEN.env")

HF_TOKEN = os.getenv("HF_TOKEN")
headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

res = requests.post(
    "https://router.huggingface.co/hf-inference/models/Helsinki-NLP/opus-mt-en-ar",
    headers=headers,
    json={"inputs": "hello how are you"}
)

print("Status:", res.status_code)
print("Response:", res.text)