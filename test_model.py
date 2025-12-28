#!/usr/bin/env python3
"""
Test the fine-tuned Arjun writing style model.
"""

import os
import tinker

# Configuration
TINKER_API_KEY = os.getenv("TINKER_API_KEY")
MODEL_NAME = "arjun-writing-style"  # Update with your model name/path

# Test prompts that match Arjun's style themes
TEST_PROMPTS = [
    "Analyze emerging market valuations and country selection strategies.",
    "Discuss the relationship between GDP growth and market returns in emerging markets.",
    "Explain the effectiveness of value investing in country selection.",
    "What are the key factors for emerging market country picking?",
    "Analyze the impact of currency movements on emerging market returns.",
]

def test_model(model_path: str = None):
    """Test the fine-tuned model with various prompts."""
    print("=" * 80)
    print("Testing Arjun Writing Style Model")
    print("=" * 80)

    # Validate API key
    if not TINKER_API_KEY:
        raise ValueError("TINKER_API_KEY environment variable not set")

    os.environ["TINKER_API_KEY"] = TINKER_API_KEY

    # Initialize Tinker client
    print("\nInitializing Tinker client...")
    service_client = tinker.ServiceClient()

    # Get sampling client
    if model_path:
        print(f"Loading model from: {model_path}")
        # You would load the specific model path here
        # This depends on how Tinker exposes saved models
        # For now, we'll assume the model name is available
    else:
        print(f"Using model: {MODEL_NAME}")

    print("\n" + "=" * 80)
    print("Running test prompts...")
    print("=" * 80)

    # Note: You'll need to get the sampling_client from your trained model
    # This is typically returned from the training script
    # For demonstration, this shows the structure

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[Test {i}/{len(TEST_PROMPTS)}]")
        print(f"Prompt: {prompt}")
        print("-" * 80)

        try:
            # You'll need to replace this with actual sampling
            # sampling_client.sample() based on your saved model
            print("Response:")
            print("[Note: Connect to your saved model to get actual responses]")
            print("-" * 80)

        except Exception as e:
            print(f"Error generating response: {e}")
            continue

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)

def compare_with_original(model_response: str, original_style_sample: str):
    """
    Compare model output with original Arjun writing samples.

    Check for:
    - Quantitative rigor (mentions of statistics, regressions, T-stats)
    - Clear structure ("What We Did", "Conclusions")
    - Data-driven language
    - Academic yet accessible tone
    """
    style_markers = {
        "quantitative": ["T-stat", "regression", "quintile", "correlation", "R-squared", "significant"],
        "structure": ["What We Did:", "Conclusions:", "Figure", "Table"],
        "data_driven": ["data shows", "results show", "we found", "analysis", "study"],
        "academic": ["however", "therefore", "furthermore", "in conclusion", "interestingly"],
    }

    print("\nStyle Analysis:")
    print("-" * 80)

    for category, markers in style_markers.items():
        found = [m for m in markers if m.lower() in model_response.lower()]
        print(f"{category.capitalize()}: {len(found)} markers found")
        if found:
            print(f"  Found: {', '.join(found)}")

    print("-" * 80)

if __name__ == "__main__":
    # For now, this is a template
    # You'll need to update it with the actual model path after training
    print("\nThis script provides a testing framework for your fine-tuned model.")
    print("After training completes, you can use this to validate the model outputs.")
    print("\nTo use:")
    print("1. Update MODEL_NAME or model_path with your trained model identifier")
    print("2. Run: python test_model.py")
