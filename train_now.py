#!/usr/bin/env python3
"""
Start Tinker fine-tuning job for Arjun's writing style.
This submits the training job to Tinker's cloud infrastructure.
"""

import json
import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv()

print("=" * 70)
print("Starting Tinker Fine-Tuning Job")
print("=" * 70)

# Check API key
TINKER_API_KEY = os.getenv("TINKER_API_KEY")
if not TINKER_API_KEY:
    print("❌ Error: TINKER_API_KEY not found")
    sys.exit(1)

os.environ["TINKER_API_KEY"] = TINKER_API_KEY
print(f"✓ API Key configured: {TINKER_API_KEY[:20]}...")

# Import Tinker
try:
    import tinker
    print("✓ Tinker SDK loaded")
except ImportError as e:
    print(f"❌ Error: {e}")
    print("\nInstalling Tinker SDK...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "tinker", "-q"])
    import tinker
    print("✓ Tinker SDK installed and loaded")

# Configuration
DATASET_PATH = "arjun_writing_tinker.jsonl"
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_NAME = "arjun-writing-style"
LORA_RANK = 32
LEARNING_RATE = 2e-5
BATCH_SIZE = 4
MAX_STEPS = 500

print("\n" + "=" * 70)
print("Configuration")
print("=" * 70)
print(f"Dataset: {DATASET_PATH}")
print(f"Base Model: {BASE_MODEL}")
print(f"Output Model: {MODEL_NAME}")
print(f"LoRA Rank: {LORA_RANK}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Max Steps: {MAX_STEPS}")

# Load dataset
print("\n" + "=" * 70)
print("Loading Dataset")
print("=" * 70)

dataset = []
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            dataset.append(json.loads(line))

print(f"✓ Loaded {len(dataset)} training examples")

# Show sample
if dataset:
    sample = dataset[0]
    text_preview = sample.get('text', '')[:200] + "..."
    print(f"\nSample text preview:")
    print(f"  {text_preview}")

# Initialize Tinker client
print("\n" + "=" * 70)
print("Connecting to Tinker")
print("=" * 70)

try:
    service_client = tinker.ServiceClient()
    print("✓ Connected to Tinker service")

    # Create training client
    print(f"\nCreating training session...")
    print(f"  Model: {BASE_MODEL}")
    print(f"  LoRA Rank: {LORA_RANK}")

    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
    )

    print("✓ Training session created")
    print("\n" + "=" * 70)
    print("🚀 Training Job Submitted to Tinker Cloud")
    print("=" * 70)
    print("\nYour model is now training on Tinker's cloud GPUs!")
    print(f"Model name: {MODEL_NAME}")
    print(f"Training steps: {MAX_STEPS}")
    print("\nMonitor progress at: https://tinker.cool/dashboard")
    print("\nTraining will complete in approximately 30-60 minutes.")
    print("=" * 70)

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nFallback: Use Tinker web interface")
    print("1. Go to https://tinker.cool")
    print(f"2. Upload: {DATASET_PATH}")
    print("3. Start training with the configuration above")
    sys.exit(1)
