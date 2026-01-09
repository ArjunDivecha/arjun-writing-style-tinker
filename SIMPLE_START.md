# Start Training - Simple Guide

## ✅ What's Ready

Your dataset is prepared and validated:
- **File**: `arjun_writing_tinker.jsonl`
- **Examples**: 35 investment research samples
- **Split**: 31 training, 4 validation
- **Format**: Instruction-following chat format

## 🚀 Start Training (2 Minutes)

### Step 1: Go to Tinker
Visit **https://tinker.cool** and log in with your account

### Step 2: Upload Dataset
1. Click **"Datasets"** → **"Upload"**
2. Select: `arjun_writing_tinker.jsonl` from this repository
3. Wait for upload to complete (should be quick, ~270KB)

### Step 3: Configure Training
Click **"Train New Model"** and use these exact settings:

```
Base Model: meta-llama/Llama-3.2-3B-Instruct
Model Name: arjun-writing-style
LoRA Rank: 32
Learning Rate: 0.00002 (or 2e-5)
Batch Size: 4
Epochs: 3
Max Steps: 500
```

### Step 4: Start
Click **"Start Training"**

## ⏱️ Timeline

- **Training Time**: 30-60 minutes
- **Location**: Runs on Tinker's cloud GPUs
- **Cost**: Check your Tinker plan
- **Monitoring**: View progress in Tinker dashboard

## 🔑 Your API Key

```
tml-wCPrwV6ZBfzF6N51Ors440nYZexhTahd3XiVRAC7zLRhWObrVDNzrtzeTKxxWiudAAAAA
```

## 📊 What the Model Will Learn

The model will learn Arjun Divecha's writing style:
- Quantitative rigor (T-stats, regressions, statistical analysis)
- Clear structure ("What We Did", "Conclusions", "Figure X shows...")
- Data-driven investment analysis
- Academic yet accessible tone
- Emerging markets focus

## ✨ After Training

Once complete, you can:
1. Test the model via Tinker's inference API
2. Download model weights
3. Use it to generate investment research in Arjun's style

## 💡 Testing Prompts

Try these prompts after training:
- "Analyze emerging market valuations and country selection strategies."
- "Discuss the relationship between GDP growth and market returns."
- "Evaluate the effectiveness of value investing in country selection."
