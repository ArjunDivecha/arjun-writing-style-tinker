# Arjun Writing Style - Tinker Training

Train an AI model to replicate Arjun Divecha's professional investment writing style using [Tinker](https://tinker.cool).

## 🚀 Quick Start (2 Minutes)

**→ See [SIMPLE_START.md](SIMPLE_START.md) for step-by-step instructions**

**TL;DR:**
1. Go to https://tinker.cool
2. Upload `arjun_writing_tinker.jsonl`
3. Start training with recommended config
4. Wait 30-60 minutes
5. Done!

## 📊 Dataset

- **File**: `arjun_writing_tinker.jsonl`
- **Examples**: 35 investment research essays (~270 KB)
- **Content**: Emerging markets analysis with quantitative rigor
- **Format**: JSONL (JSON Lines)

## ✅ Training Status

**Ready to train!** Everything is configured:
- ✅ Dataset validated (31 train / 4 test)
- ✅ Optimal hyperparameters set
- ✅ API key configured
- ✅ Instructions prepared

## 📝 What You'll Get

A model that writes like Arjun:
- Quantitative rigor (T-stats, regressions, statistical analysis)
- Clear structure ("What We Did", "Conclusions")
- Data-driven investment insights
- Academic yet accessible tone

## 📖 Documentation

- **[SIMPLE_START.md](SIMPLE_START.md)** - Quick training guide (start here!)
- **[README_TRAINING.md](README_TRAINING.md)** - Detailed overview
- **[TRAINING_INSTRUCTIONS.md](TRAINING_INSTRUCTIONS.md)** - Complete documentation

## 🔧 Training Configuration

```
Model: meta-llama/Llama-3.2-3B-Instruct
LoRA Rank: 32
Learning Rate: 2e-5
Batch Size: 4
Epochs: 3
Output: arjun-writing-style
```

## 💡 Example Prompts

After training, try:
- "Analyze emerging market valuations and country selection strategies"
- "Discuss the relationship between GDP growth and market returns"
- "Evaluate value investing effectiveness in emerging markets"

## 📖 Resources

- [Tinker Platform](https://tinker.cool)
- [Tinker Documentation](https://tinker-docs.thinkingmachines.ai)
