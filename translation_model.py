import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import evaluate
import nltk
nltk.download('punkt')

# === Config ===
MODEL_NAME = "Helsinki-NLP/opus-mt-en-es"
MAX_LENGTH = 64
BATCH_SIZE = 16


# === Load Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
vocab_size = tokenizer.vocab_size


# === Dataset and Dataloader ===
class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer):
        self.src = src_texts
        self.tgt = tgt_texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        # return torch.tensor(self.src_vocab.encode(self.src[idx])), torch.tensor(self.tgt_vocab.encode(self.tgt[idx]))
        src_enc = tokenizer(self.src[idx], return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)
        tgt_enc = tokenizer(self.tgt[idx], return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)
        return src_enc["input_ids"].squeeze(0), tgt_enc["input_ids"].squeeze(0)



def collate_fn(batch):
    src, tgt = zip(*batch)
    src = torch.stack(src)
    tgt = torch.stack(tgt)
    return src, tgt


# === Positional Encoding ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
        self.register_buffer('pe_buffer', self.pe)

    def forward(self, x):
        return x + self.pe_buffer[:, :x.size(1)]


# === Attention and Transformer ===
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        # self.qkv = nn.Linear(d_model * 3, d_model * 3)
        self.q_linear = nn.Linear(d_model, d_model) # Create separate linear layers for q, k, and v
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None, kv=None):
        if kv is None:
            kv = x
        batch_size, seq_len, d_model = x.size()

        # qkv = self.qkv(torch.cat([x, kv, kv], dim=-1))
        # q, k, v = qkv.chunk(3, dim=-1)

        # q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(kv).view(batch_size, kv.size(1), self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(kv).view(batch_size, kv.size(1), self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.d_k ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.norm1(x + self.attn(x, mask))
        return self.norm2(x + self.ff(x))



class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, tgt_mask, src_mask):
        x = self.norm1(x + self.self_attn(x, tgt_mask))
        x = self.norm2(x + self.cross_attn(x, src_mask, enc_out))
        return self.norm3(x + self.ff(x))


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=4):
        super().__init__()
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.out = nn.Linear(d_model, vocab_size)

    def encode(self, src, src_mask):
        x = self.pos_enc(self.src_embed(src))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, tgt_mask, src_mask):
        x = self.pos_enc(self.tgt_embed(tgt))
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.out(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(tgt, memory, tgt_mask, src_mask)


# === Masks ===
def create_mask(seq, pad_idx=0):
    return (seq != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)

def create_subsequent_mask(size):
    return torch.tril(torch.ones((size, size), dtype=torch.bool))



# === Training ===
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        src_mask = create_mask(src).to(device)
        tgt_mask = create_mask(tgt_input).to(device) & create_subsequent_mask(tgt_input.size(1)).to(device)

        logits = model(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# === Evaluation ===
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    preds = []
    refs = []

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
            memory = model.encode(src, src_mask)

            ys = torch.ones(src.size(0), 1).fill_(1).long().to(device)  # <sos>

            for _ in range(30):  # max length
                tgt_mask = create_subsequent_mask(ys.size(1)).to(device)
                out = model.decode(ys, memory, tgt_mask, src_mask)
                next_word = out[:, -1].argmax(-1).unsqueeze(1)
                ys = torch.cat([ys, next_word], dim=1)
                if torch.all(next_word == 2):  # all <eos>
                    break

            for pred_seq, true_seq in zip(ys, tgt):
                pred_text = tokenizer.decode(pred_seq.tolist(), skip_special_tokens=True)
                true_text = tokenizer.decode(true_seq.tolist(), skip_special_tokens=True)

                preds.append(pred_text)
                refs.append(true_text)

    # Evaluate with Hugging Face
    bleu = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])['bleu']
    rouge = rouge_metric.compute(predictions=preds, references=refs)['rougeL']

    return bleu, rouge