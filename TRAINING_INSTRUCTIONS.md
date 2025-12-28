# Tinker Fine-Tuning Instructions

## Quick Start (Web Interface - Easiest)

1. **Go to https://tinker.cool and log in**

2. **Upload Dataset**
   - Click "Datasets" → "Upload"
   - Upload `arjun_writing_tinker.jsonl`

3. **Start Training**
   - Click "Train New Model"
   - Select your uploaded dataset
   - Configure:
     - Model: `meta-llama/Llama-3.2-3B-Instruct`
     - LoRA Rank: 32
     - Learning Rate: 2e-5
     - Epochs: 3-5
     - Batch Size: 4-8
   - Click "Start Training"

4. **Monitor Progress**
   - Training runs on Tinker's cloud GPUs
   - Check dashboard for progress
   - Model will be saved when complete

## Using Python SDK (Alternative)

If you want to use the Python SDK:

```bash
# Install dependencies
pip install tinker

# Set API key
export TINKER_API_KEY="your-api-key"

# Run training script
python run_training.py
```

## API Key

Your API key is stored in `.env`:
```
TINKER_API_KEY=tml-wCPrwV6ZBfzF6N51Ors440nYZexhTahd3XiVRAC7zLRhWObrVDNzrtzeTKxxWiudAAAAA
```

## Dataset Format

The dataset (`arjun_writing_tinker.jsonl`) contains 35 examples of Arjun's investment writing style.

Each line is a JSON object with investment analysis text demonstrating:
- Quantitative rigor (T-stats, regressions)
- Clear structure ("What We Did", "Conclusions")
- Data-driven analysis
- Academic yet accessible tone

## After Training

Once training completes, you can:
1. Test the model via Tinker's inference API
2. Download the model weights
3. Deploy for production use

Training typically takes 30-60 minutes depending on configuration.
