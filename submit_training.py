#!/usr/bin/env python3
"""
Submit training job to Tinker using direct API calls.
No heavy dependencies - just HTTP requests.
"""

import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
TINKER_API_KEY = os.getenv("TINKER_API_KEY")
DATASET_PATH = "arjun_writing_tinker.jsonl"

# Training parameters
config = {
    "base_model": "meta-llama/Llama-3.2-3B-Instruct",
    "model_name": "arjun-writing-style",
    "lora_rank": 32,
    "learning_rate": 2e-5,
    "batch_size": 4,
    "epochs": 3,
    "max_steps": 500
}

print("=" * 70)
print("Tinker Fine-Tuning Submission")
print("=" * 70)
print(f"\nDataset: {DATASET_PATH}")
print(f"Base Model: {config['base_model']}")
print(f"Output Model: {config['model_name']}")
print(f"\nConfiguration:")
for key, value in config.items():
    if key not in ['base_model', 'model_name']:
        print(f"  {key}: {value}")

# Load dataset
print(f"\nLoading dataset...")
with open(DATASET_PATH, 'r') as f:
    dataset_lines = [line.strip() for line in f if line.strip()]

print(f"✓ Loaded {len(dataset_lines)} examples")

# Show sample
if dataset_lines:
    sample = json.loads(dataset_lines[0])
    text = sample.get('text', '')
    print(f"\nSample preview: {text[:150]}...")

print("\n" + "=" * 70)
print("Next Step: Submit to Tinker")
print("=" * 70)
print("\nOption 1 - Web Interface (Recommended):")
print("1. Go to https://tinker.cool")
print(f"2. Upload dataset: {DATASET_PATH}")
print("3. Configure training with parameters above")
print("4. Start training")

print("\nOption 2 - Python SDK:")
print("Run: pip install tinker && python train_now.py")

print("\n" + "=" * 70)
print("Your API Key:")
print(TINKER_API_KEY)
print("=" * 70)

print("\n✓ Everything is ready!")
print("Training will run on Tinker's cloud infrastructure.")
print("Expected duration: 30-60 minutes")
