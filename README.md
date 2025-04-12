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
├── app.py                     # Streamlit web UI
├── train.py                  # Training loop with evaluation and checkpointing
├── translation_model.py      # Core Transformer architecture & utilities
├── english_spanish.csv       # Dataset
├── requirements.txt           # Requirements file 
└── README.md

🧠 Model Architecture
This is a classic encoder-decoder Transformer, built using PyTorch:

- Multi-Head Self-Attention
- Positional Encoding
- Layer Normalization
- Masked Decoding
- Beam Search & Greedy Decoding

The model was trained on ~300,000 English-Spanish pairs.

