# app/download_model.py

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "SamLowe/roberta-base-go_emotions"
local_dir = os.path.join(os.path.dirname(__file__), "..", "models", "goemotions")
local_dir = os.path.abspath(local_dir)

if os.path.exists(local_dir) and os.path.isdir(local_dir):
    print(f"Model already exists at {local_dir}. Skipping download.")
else:
    print(f"Downloading GoEmotions model to {local_dir}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)
    print("Download complete.")
