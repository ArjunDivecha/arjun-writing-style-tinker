#!/usr/bin/env python3
"""
Fine-tune a language model using Tinker API to replicate Arjun Divecha's writing style.
"""

import json
import os
from typing import List, Dict
import tinker
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

# Configuration
TINKER_API_KEY = os.getenv("TINKER_API_KEY")
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DATASET_PATH = "arjun_writing_tinker.jsonl"
LORA_RANK = 32
LEARNING_RATE = 2e-5
BATCH_SIZE = 4
MAX_STEPS = 500
SAVE_FREQ = 100
MODEL_NAME = "arjun-writing-style"

def load_and_prepare_dataset(dataset_path: str) -> List[Dict]:
    """Load JSONL dataset and convert to instruction format."""
    print(f"Loading dataset from {dataset_path}...")

    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            # Convert raw text to instruction-response format
            # We'll create a prompt asking for investment analysis in Arjun's style
            text = entry.get('text', '')

            # Create instruction-following format
            messages = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Write an investment research analysis in the style of Arjun Divecha."
                    },
                    {
                        "role": "assistant",
                        "content": text
                    }
                ]
            }
            data.append(messages)

    print(f"Loaded {len(data)} examples")
    return data

def format_messages(messages: List[Dict], tokenizer) -> str:
    """Format messages using the chat template."""
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return formatted

def main():
    print("=" * 60)
    print("Arjun Writing Style Fine-tuning with Tinker")
    print("=" * 60)

    # Validate API key
    if not TINKER_API_KEY:
        raise ValueError("TINKER_API_KEY environment variable not set")

    # Set API key
    os.environ["TINKER_API_KEY"] = TINKER_API_KEY

    # Load dataset
    dataset = load_and_prepare_dataset(DATASET_PATH)
    print(f"\nDataset size: {len(dataset)} examples\n")

    # Initialize tokenizer
    print(f"Loading tokenizer for {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize Tinker client
    print(f"\nInitializing Tinker service client...")
    service_client = tinker.ServiceClient()

    # Create LoRA training client
    print(f"Creating LoRA training client...")
    print(f"  Base Model: {BASE_MODEL}")
    print(f"  LoRA Rank: {LORA_RANK}")
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
    )

    # Training loop
    print(f"\nStarting training...")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Max Steps: {MAX_STEPS}")
    print(f"  Save Frequency: Every {SAVE_FREQ} steps")
    print("=" * 60)

    step = 0
    total_loss = 0

    # Create progress bar
    pbar = tqdm(total=MAX_STEPS, desc="Training")

    while step < MAX_STEPS:
        # Sample a batch
        batch_indices = torch.randint(0, len(dataset), (BATCH_SIZE,))
        batch_data = [dataset[i] for i in batch_indices]

        # Format batch
        formatted_texts = []
        for example in batch_data:
            formatted_text = format_messages(
                example["messages"],
                tokenizer
            )
            formatted_texts.append(formatted_text)

        # Tokenize
        tokens = tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )

        # Forward and backward pass
        try:
            result = training_client.forward_backward(
                input_ids=tokens["input_ids"].tolist(),
                attention_mask=tokens["attention_mask"].tolist(),
            )

            # Optimizer step
            training_client.optim_step(learning_rate=LEARNING_RATE)

            # Track loss
            if hasattr(result, 'loss'):
                total_loss += result.loss

            step += 1
            pbar.update(1)

            # Log progress
            if step % 10 == 0:
                avg_loss = total_loss / step if step > 0 else 0
                pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})

            # Save checkpoint
            if step % SAVE_FREQ == 0:
                print(f"\n[Step {step}] Saving checkpoint...")
                training_client.save_state(f"checkpoint_step_{step}")

        except Exception as e:
            print(f"\nError at step {step}: {e}")
            print("Continuing training...")
            continue

    pbar.close()

    # Save final model
    print("\n" + "=" * 60)
    print("Training complete! Saving final model...")
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=MODEL_NAME
    )

    print(f"✓ Model saved as: {MODEL_NAME}")
    print(f"✓ Model path: {sampling_client.model_path}")
    print("=" * 60)

    # Test the model with a sample prompt
    print("\nTesting model with sample prompt...")
    test_prompt = "Analyze emerging market valuations and country selection strategies."

    response = sampling_client.sample(
        prompt=test_prompt,
        max_tokens=500,
        temperature=0.7,
    )

    print(f"\nPrompt: {test_prompt}")
    print(f"\nResponse:\n{response}")
    print("\n" + "=" * 60)
    print("Fine-tuning complete!")
    print("=" * 60)

    return sampling_client

if __name__ == "__main__":
    main()
