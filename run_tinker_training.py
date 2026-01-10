#!/usr/bin/env python3
"""
Actually run training on Tinker using the proper SDK workflow.
This submits batches to Tinker's distributed training API.
"""

import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load API key
load_dotenv()
TINKER_API_KEY = os.getenv("TINKER_API_KEY")
if not TINKER_API_KEY:
    print("❌ Error: TINKER_API_KEY not found in .env file")
    sys.exit(1)

os.environ["TINKER_API_KEY"] = TINKER_API_KEY

print("=" * 70)
print("Arjun Writing Style - Tinker Training")
print("=" * 70)

# Import Tinker
try:
    import tinker
    from transformers import AutoTokenizer
    import torch
    print("✓ Tinker SDK loaded")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Configuration
DATASET_PATH = "arjun_writing_tinker.jsonl"
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LORA_RANK = 32
LEARNING_RATE = 2e-5
BATCH_SIZE = 4
MAX_STEPS = 500
MODEL_NAME = "arjun-writing-style"

print(f"\nConfiguration:")
print(f"  Base Model: {BASE_MODEL}")
print(f"  Dataset: {DATASET_PATH}")
print(f"  LoRA Rank: {LORA_RANK}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Max Steps: {MAX_STEPS}")
print(f"  Output Model: {MODEL_NAME}")

# Load dataset
print(f"\n" + "=" * 70)
print("Loading Dataset")
print("=" * 70)

dataset = []
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            # Convert to chat format
            dataset.append({
                "messages": [
                    {"role": "user", "content": "Write investment research analysis in Arjun Divecha's style with quantitative rigor."},
                    {"role": "assistant", "content": entry["text"]}
                ]
            })

print(f"✓ Loaded {len(dataset)} examples")
if dataset:
    preview = dataset[0]["messages"][1]["content"][:120]
    print(f"\nSample: {preview}...")

# Initialize Tinker
print(f"\n" + "=" * 70)
print("Connecting to Tinker")
print("=" * 70)

try:
    service_client = tinker.ServiceClient()
    print("✓ Connected to Tinker service")

    # Load tokenizer
    print(f"\nLoading tokenizer for {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")

    # Create training client
    print(f"\nCreating LoRA training client...")
    print(f"  Rank: {LORA_RANK}")
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
    )
    print("✓ Training client created")

    # Training loop
    print(f"\n" + "=" * 70)
    print("🚀 Starting Training")
    print("=" * 70)
    print(f"Training for {MAX_STEPS} steps...")
    print("This will run on Tinker's distributed GPUs")

    step = 0
    total_loss = 0

    while step < MAX_STEPS:
        # Sample random batch
        batch_idx = torch.randint(0, len(dataset), (BATCH_SIZE,))
        batch = [dataset[i] for i in batch_idx]

        # Format messages
        texts = []
        for example in batch:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)

        # Tokenize
        tokens = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )

        # Forward & backward pass on Tinker's GPUs
        try:
            result = training_client.forward_backward(
                input_ids=tokens["input_ids"].tolist(),
                attention_mask=tokens["attention_mask"].tolist(),
            )

            # Optimizer step
            training_client.optim_step(learning_rate=LEARNING_RATE)

            step += 1

            # Log progress
            if step % 10 == 0:
                print(f"  Step {step}/{MAX_STEPS}")

            # Save checkpoint
            if step % 100 == 0:
                print(f"\n  [Step {step}] Saving checkpoint...")
                training_client.save_state(f"checkpoint_step_{step}")

        except Exception as e:
            print(f"\n❌ Error at step {step}: {e}")
            break

    # Save final model
    print(f"\n" + "=" * 70)
    print("Saving Final Model")
    print("=" * 70)

    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=MODEL_NAME
    )

    print(f"\n✅ Training Complete!")
    print(f"✓ Model saved as: {MODEL_NAME}")
    print(f"✓ Model path: {sampling_client.model_path}")

    # Test the model
    print(f"\n" + "=" * 70)
    print("Testing Model")
    print("=" * 70)

    test_prompt = "Analyze emerging market valuations and country selection strategies."
    print(f"\nPrompt: {test_prompt}")

    response = sampling_client.sample(
        prompt=test_prompt,
        max_tokens=300,
        temperature=0.7,
    )

    print(f"\nGenerated Response:")
    print(response)

    print(f"\n" + "=" * 70)
    print("✅ Success! Model is ready to use")
    print("=" * 70)

except Exception as e:
    print(f"\n❌ Training failed: {e}")
    print("\nDebugging info:")
    print(f"  API Key set: {bool(TINKER_API_KEY)}")
    print(f"  Dataset loaded: {len(dataset)} examples")
    sys.exit(1)
