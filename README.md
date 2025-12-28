# Arjun Writing Style - Tinker Training Dataset

Training dataset for replicating Arjun Divecha's professional investment writing style using [Tinker](https://tinker.cool).

## 📊 Dataset

- **File**: `arjun_writing_tinker.jsonl`
- **Format**: JSONL (JSON Lines)
- **Examples**: 35 investment research essays
- **Size**: ~270 KB
- **Content**: Emerging markets analysis, quantitative research, country selection strategies

## 🚀 Quick Start

1. Go to [tinker.cool](https://tinker.cool) and create account
2. Download `arjun_writing_tinker.jsonl` from this repo
3. Upload to Tinker → **Datasets** → **Upload**
4. Configure training:
   - Model: `Llama-3.2-3B-Instruct`
   - Epochs: 3-5
   - Learning Rate: 2e-5
   - Batch Size: 4-8
5. Train and test with prompts like: "Analyze emerging market valuations"

## 📝 Style Characteristics

✅ Quantitative rigor (T-stats, regressions)  
✅ Clear structure ("What We Did", "Conclusions")  
✅ Data-driven analysis  
✅ Academic yet accessible tone

## 💡 Use with Claude Code

Tell Claude Code on the web to run this:

\`\`\`bash
claude code "Train my writing style model using arjun_writing_tinker.jsonl from the repo"
\`\`\`

## 📖 Resources

- [Tinker Docs](https://docs.tinker.cool)
- [Example: Gertrude Stein Style](https://muratcankoylan.com/projects/gertrude-stein-style-training/)
