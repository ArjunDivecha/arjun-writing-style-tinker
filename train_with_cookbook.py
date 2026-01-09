#!/usr/bin/env python3
"""
Train Arjun's writing style using Tinker Cookbook.
"""

import asyncio
import json
from pathlib import Path
from typing import Tuple, Iterator
import chz
from dotenv import load_dotenv

# Load environment
load_dotenv()

@chz.chz
class Config:
    """Training configuration for Arjun writing style model."""

    # Model settings
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    rank: int = 32

    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 4
    max_length: int = 2048
    max_steps: int = 500

    # Evaluation and checkpointing
    eval_every: int = 50
    save_every: int = 100

    # Paths
    dataset_path: str = "arjun_writing_tinker.jsonl"
    log_path: str = "arjun-writing-style"


class ArjunWritingDataset:
    """Dataset builder for Arjun's writing samples."""

    def __init__(self, config: Config):
        self.config = config
        self.dataset_path = Path(config.dataset_path)

    def load_data(self) -> list:
        """Load JSONL dataset."""
        data = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    # Convert to chat format
                    chat_entry = {
                        "messages": [
                            {
                                "role": "user",
                                "content": "Write an investment research analysis in the style of Arjun Divecha, focusing on emerging markets with quantitative rigor."
                            },
                            {
                                "role": "assistant",
                                "content": entry.get("text", "")
                            }
                        ]
                    }
                    data.append(chat_entry)
        return data

    def build(self) -> Tuple[list, list]:
        """Build train and test datasets."""
        data = self.load_data()

        # Split 90/10 train/test
        split_idx = int(len(data) * 0.9)
        train_data = data[:split_idx]
        test_data = data[split_idx:]

        return train_data, test_data


async def main():
    """Main training function."""

    print("=" * 70)
    print("Arjun Writing Style - Tinker Training")
    print("=" * 70)

    # Create config
    config = Config()

    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA Rank: {config.rank}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Max Steps: {config.max_steps}")
    print(f"  Dataset: {config.dataset_path}")

    # Load and prepare dataset
    print(f"\nLoading dataset...")
    dataset_builder = ArjunWritingDataset(config)
    train_data, test_data = dataset_builder.build()

    print(f"✓ Train examples: {len(train_data)}")
    print(f"✓ Test examples: {len(test_data)}")

    # Show sample
    if train_data:
        sample = train_data[0]["messages"][1]["content"]
        print(f"\nSample text: {sample[:150]}...")

    print("\n" + "=" * 70)
    print("Starting Training on Tinker Cloud")
    print("=" * 70)

    try:
        # Import tinker cookbook
        from tinker_cookbook.supervised import train as tinker_train

        # Submit training job
        print("\n🚀 Submitting training job to Tinker...")
        await tinker_train.main(config)

        print("\n✓ Training job submitted successfully!")
        print(f"✓ Model will be saved as: {config.log_path}")
        print("\nMonitor progress at: https://tinker.cool/dashboard")

    except ImportError:
        print("\n⚠️  Tinker cookbook not installed")
        print("\nInstall with: pip install tinker tinker-cookbook")
        print("\nOr use Tinker web interface:")
        print("1. Go to https://tinker.cool")
        print(f"2. Upload: {config.dataset_path}")
        print("3. Configure with parameters above")
        print("4. Start training")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nFallback: Use Tinker web interface at https://tinker.cool")


if __name__ == "__main__":
    asyncio.run(main())
