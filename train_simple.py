#!/usr/bin/env python3
"""
Simple script to start Tinker training using their REST API.
Training runs on Tinker's cloud infrastructure.
"""

import json
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TINKER_API_KEY = os.getenv("TINKER_API_KEY")
DATASET_PATH = "arjun_writing_tinker.jsonl"

# Tinker API configuration
BASE_URL = "https://api.tinker.cool/v1"  # Update with actual Tinker API URL
HEADERS = {
    "Authorization": f"Bearer {TINKER_API_KEY}",
    "Content-Type": "application/json"
}

# Training configuration
config = {
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "dataset": DATASET_PATH,
    "lora_rank": 32,
    "learning_rate": 2e-5,
    "batch_size": 4,
    "epochs": 3,
    "max_steps": 500,
    "model_name": "arjun-writing-style"
}

print("=" * 60)
print("Starting Tinker Fine-tuning Job")
print("=" * 60)
print(f"Model: {config['model']}")
print(f"Dataset: {config['dataset']}")
print(f"Training will run on Tinker's cloud infrastructure")
print("=" * 60)

# Read and prepare dataset
print("\nReading dataset...")
with open(DATASET_PATH, 'r') as f:
    dataset_lines = f.readlines()

print(f"Loaded {len(dataset_lines)} examples")

# Submit training job to Tinker
print("\nSubmitting training job to Tinker...")
print("(Note: Update API endpoint based on Tinker documentation)")

# Placeholder - update with actual Tinker API endpoint
print("\nTo complete this, please check Tinker's API documentation at:")
print("https://tinker-docs.thinkingmachines.ai/")
print("\nYou can also use their web interface at https://tinker.cool")
print("\n" + "=" * 60)
print("Next steps:")
print("1. Upload dataset to Tinker console")
print("2. Start training job with these parameters:")
print(json.dumps(config, indent=2))
print("=" * 60)
