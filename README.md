# 🌍 Transformer-based English to Spanish Machine Translation

A complete machine translation pipeline using the Transformer architecture from the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762), built from scratch in PyTorch.

This project supports:
- 🧠 End-to-end training on a open source English to Spanish translations dataset
- 📊 Evaluation with BLEU and ROUGE using Hugging Face's `evaluate`
- 💬 Streamlit demo for real-time translation
- 🔍 Attention visualization (optional)
- 🔁 Beam search decoding
- ☁️ Push to Hugging Face Hub for open source sharing (pending)

---

## 🚀 Project Structure

```bash
.
├── app.py                    # Streamlit web UI
├── train.py                  # Training loop with evaluation and checkpointing
├── translation_model.py      # Core Transformer architecture & utilities
├── decode_utils.py           # decoding functions BEAM and Greedy
├── english_spanish.csv       # Dataset
├── requirements.txt          # Requirements file 
└── README.md

---

## 🧠 Model Architecture
This is a classic encoder-decoder Transformer, built using PyTorch:

- Multi-Head Self-Attention
- Positional Encoding
- Layer Normalization
- Masked Decoding
- Beam Search & Greedy Decoding

```bash

The model was trained on ~300,000 English-Spanish pairs.

## Results (trained on smaller dataset):
BLEU Scores: ~36%
ROUGE Scores: ~46%

## Next Steps:
✅ Pretrained multilingual models like MBart50 for faster convergence and real-world performance
✅ Replacing the custom tokenizer with AutoTokenizer from Hugging Face for subword and BPE support
✅ Scaling up with larger datasets (e.g., Europarl, Tatoeba), longer training, and better hardware utilization
✅ Cloud deployment using AWS or GCP to host the model and Streamlit app as a translation service
