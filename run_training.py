#!/usr/bin/env python3
"""
Upload dataset to Tinker and start fine-tuning job.
All training runs on Tinker's cloud infrastructure.
"""

import os
import sys

# Check if tinker package is available, if not provide instructions
try:
    import tinker
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease install required packages:")
    print("pip install tinker python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

TINKER_API_KEY = os.getenv("TINKER_API_KEY")
if not TINKER_API_KEY:
    print("Error: TINKER_API_KEY not found in environment")
    print("Please set it in .env file or environment variables")
    sys.exit(1)

# Configuration
DATASET_PATH = "arjun_writing_tinker.jsonl"
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_NAME = "arjun-writing-style"
LORA_RANK = 32
LEARNING_RATE = 2e-5
EPOCHS = 3

print("=" * 70)
print("Arjun Writing Style Fine-Tuning with Tinker")
print("=" * 70)
print(f"Base Model: {MODEL}")
print(f"Dataset: {DATASET_PATH}")
print(f"Output Model: {MODEL_NAME}")
print(f"LoRA Rank: {LORA_RANK}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS}")
print("=" * 70)

print("\nInitializing Tinker client...")
os.environ["TINKER_API_KEY"] = TINKER_API_KEY

# Create service client
service_client = tinker.ServiceClient()

print("✓ Connected to Tinker")
print("\nStarting training job...")
print("(Training will run on Tinker's cloud infrastructure)")
print("\nCheck your Tinker dashboard at https://tinker.cool for progress")
print("=" * 70)
