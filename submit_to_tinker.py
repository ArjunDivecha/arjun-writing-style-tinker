#!/usr/bin/env python3
"""
Actually submit training to Tinker using their Python SDK.
Based on real Tinker API patterns.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

TINKER_API_KEY = os.getenv("TINKER_API_KEY")
if not TINKER_API_KEY:
    print("❌ Error: TINKER_API_KEY not found in .env")
    exit(1)

os.environ["TINKER_API_KEY"] = TINKER_API_KEY

print("=" * 70)
print("Submitting Training Job to Tinker")
print("=" * 70)

# Configuration
DATASET_PATH = "arjun_writing_tinker.jsonl"
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LORA_RANK = 32
LEARNING_RATE = 2e-5
MAX_STEPS = 500

print(f"\nConfiguration:")
print(f"  Dataset: {DATASET_PATH}")
print(f"  Model: {BASE_MODEL}")
print(f"  LoRA Rank: {LORA_RANK}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Max Steps: {MAX_STEPS}")

# Load dataset
print(f"\nLoading dataset...")
data = []
with open(DATASET_PATH, 'r') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

print(f"✓ Loaded {len(data)} examples")

# Show sample
if data:
    sample_text = data[0].get('text', '')[:150]
    print(f"\nSample: {sample_text}...")

print("\n" + "=" * 70)
print("Initializing Tinker Client")
print("=" * 70)

try:
    import tinker

    # Create service client
    print("\nConnecting to Tinker service...")
    service_client = tinker.ServiceClient()
    print("✓ Connected to Tinker")

    # Create LoRA training client
    print(f"\nCreating training client for {BASE_MODEL}...")
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
    )
    print("✓ Training client created")

    print("\n" + "=" * 70)
    print("🚀 Training Session Initialized!")
    print("=" * 70)
    print("\nYour training job is ready to start on Tinker's cloud.")
    print(f"Model will be trained for {MAX_STEPS} steps.")
    print("\nNext: Run the actual training loop to submit batches.")
    print("See train_full.py for the complete training implementation.")

except ImportError as e:
    print(f"\n❌ Error importing Tinker: {e}")
    print("\nTinker SDK not fully installed.")
    print("This requires PyTorch and other dependencies.")
    print("\nAlternative: Use Tinker CLI or API directly")
    exit(1)

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check API key is valid")
    print("2. Verify network connection")
    print("3. Check Tinker service status")
    exit(1)
