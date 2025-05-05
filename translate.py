import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import evaluate
import pandas as pd
from tqdm import tqdm
import torch.optim as optim

import nltk
nltk.download('punkt')


# === Tokenizer and Vocab ===
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


class Vocab:
    def __init__(self, texts, min_freq=1):
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))

        self.itos = ['<pad>', '<sos>', '<eos>', '<unk>'] + [word for word, freq in counter.items() if freq >= min_freq]
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def encode(self, text):
        return [1] + [self.stoi.get(tok, self.stoi['<unk>']) for tok in tokenize(text)] + [2]

    def decode(self, indices):
        return ' '.join([self.itos[i] for i in indices if i not in [0, 1, 2]])

    def __len__(self):
        return len(self.itos)


# === Dataset and Dataloader ===
class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab):
        self.src = src_texts
        self.tgt = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return torch.tensor(self.src_vocab.encode(self.src[idx])), torch.tensor(self.tgt_vocab.encode(self.tgt[idx]))


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, tgt_padded

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
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_heads=8, num_layers=4):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.out = nn.Linear(d_model, tgt_vocab_size)

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
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_subsequent_mask(size):
    return torch.tril(torch.ones((size, size), dtype=torch.bool))


# === Training ===
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for src, tgt in loop:
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
        loop.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

def evaluate(model, dataloader, src_vocab, tgt_vocab, device):
    model.eval()
    preds = []
    refs = []

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
            memory = model.encode(src, src_mask)

            ys = torch.ones(src.size(0), 1).fill_(1).long().to(device)  # <sos>

            for _ in range(64):  # max length
                tgt_mask = create_subsequent_mask(ys.size(1)).to(device)
                out = model.decode(ys, memory, tgt_mask, src_mask)
                next_word = out[:, -1].argmax(-1).unsqueeze(1)
                ys = torch.cat([ys, next_word], dim=1)
                if torch.all(next_word == 2):  # all <eos>
                    break

            for pred_seq, true_seq in zip(ys, tgt):
                pred_text = tgt_vocab.decode(pred_seq.cpu().tolist())
                true_text = tgt_vocab.decode(true_seq.cpu().tolist())

                preds.append(pred_text)
                refs.append(true_text)

    # Evaluate with Hugging Face
    bleu = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])['bleu']
    rouge = rouge_metric.compute(predictions=preds, references=refs)['rougeL']

    return bleu, rouge


# === Load your dataset ===
df = pd.read_csv("/home/ssehg1@cfreg.local/Bilingual-English-Spanish-Translation/data/df_merged.csv")

# === Split and tokenize ===
train_src, val_src, train_tgt, val_tgt = train_test_split(df["english"], df["spanish"], test_size=0.2, random_state=42)

eng_vocab = Vocab(train_src.tolist())
spa_vocab = Vocab(train_tgt.tolist())

train_dataset = TranslationDataset(train_src.tolist(), train_tgt.tolist(), eng_vocab, spa_vocab)
val_dataset = TranslationDataset(val_src.tolist(), val_tgt.tolist(), eng_vocab, spa_vocab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# === Set up model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(len(eng_vocab), len(spa_vocab), d_model=256)
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])  # Use 4 GPUs
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# === Training loop ===
best_bleu = 0
patience_counter = 0
early_stop_patience = 5
epochs = 30
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    bleu, rouge = evaluate(model.module, val_loader, eng_vocab, spa_vocab, device)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, BLEU = {bleu:.4f}, ROUGE-L = {rouge:.4f}")
    scheduler.step(train_loss)
    # Checkpointing
    if bleu > best_bleu:
        best_bleu = bleu
        torch.save(model.module.state_dict(), "transformer_model.pth")
        torch.save(eng_vocab, "eng_vocab.pth")
        torch.save(spa_vocab, "spa_vocab.pth")
        print("Saved best model.")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"Patience: {patience_counter}/{early_stop_patience}")

    if patience_counter >= early_stop_patience:
        print("Early stopping triggered!")
        break
# === Save model and vocabs ===
# torch.save(model.module.state_dict(), "transformer_model_1.pth")
# torch.save(eng_vocab, "eng_vocab_1.pth")
# torch.save(spa_vocab, "spa_vocab_1.pth")