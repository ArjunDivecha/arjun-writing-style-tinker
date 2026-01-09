# Arjun Writing Style Fine-Tuning

Quick start guide to train a model that replicates Arjun Divecha's investment writing style.

## Dataset Overview

- **Examples**: 35 investment research essays
- **Size**: ~270 KB
- **Content**: Emerging markets analysis with quantitative rigor
- **Format**: JSONL (JSON Lines)

The dataset captures Arjun's distinctive style:
- Statistical analysis (T-stats, regressions)
- Structured presentation
- Data-driven insights
- Professional yet accessible tone

## Training Status

✅ Dataset prepared and validated (31 train / 4 test split)
✅ Configuration optimized for 3B parameter model
✅ Environment configured with API key

## Quick Start

See **SIMPLE_START.md** for step-by-step instructions.

**TL;DR**: Upload `arjun_writing_tinker.jsonl` to https://tinker.cool and start training with the recommended configuration.

## Files

- `arjun_writing_tinker.jsonl` - Training dataset
- `SIMPLE_START.md` - Step-by-step training guide
- `TRAINING_INSTRUCTIONS.md` - Detailed documentation
- `.env` - API key configuration (not committed)
- `train_with_cookbook.py` - Python training script (alternative)

## Training Parameters

Optimized for Llama-3.2-3B-Instruct:

```
Model: meta-llama/Llama-3.2-3B-Instruct
LoRA Rank: 32
Learning Rate: 2e-5
Batch Size: 4
Epochs: 3-5
Max Steps: 500
```

## Output

After training completes (~30-60 minutes), you'll have:
- Fine-tuned model: `arjun-writing-style`
- Inference API endpoint
- Downloadable model weights

Ready to generate investment research in Arjun's distinctive analytical style!
