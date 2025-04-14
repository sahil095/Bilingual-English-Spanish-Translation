import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from translation_model import (
    TranslationDataset, collate_fn,
    Transformer, create_mask, create_subsequent_mask,
    train_epoch, evaluate
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


# === Load your dataset ===
df = pd.read_csv(r"D:\\UNH Materials\\Projects\\Bilingual Translation\\data\\data.csv")

# === Split and tokenize ===
train_src, val_src, train_tgt, val_tgt = train_test_split(df["english"], df["spanish"], test_size=0.1, random_state=42)

eng_vocab = Vocab(train_src.tolist())
spa_vocab = Vocab(train_tgt.tolist())

train_dataset = TranslationDataset(train_src.tolist(), train_tgt.tolist(), eng_vocab, spa_vocab)
val_dataset = TranslationDataset(val_src.tolist(), val_tgt.tolist(), eng_vocab, spa_vocab)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# === Set up model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(len(eng_vocab), len(spa_vocab), d_model=256).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)


# === LR scheduler & TensorBoard ===
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)


# === Training loop with Early Stopping & Checkpoints ===
best_bleu = 0
patience_counter = 0
early_stop_patience = 5
epochs = 30

# === Training loop ===
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    bleu, rouge = evaluate(model, val_loader, eng_vocab, spa_vocab, device)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, BLEU = {bleu:.4f}, ROUGE-L = {rouge:.4f}")


    scheduler.step(train_loss)

    # # Checkpointing
    # if bleu > best_bleu:
    #     best_bleu = bleu
    #     torch.save(model.state_dict(), "transformer_model.pth")
    #     print("✅ Saved best model.")
    #     patience_counter = 0
    # else:
    #     patience_counter += 1
    #     print(f"⏳ Patience: {patience_counter}/{early_stop_patience}")

    # if patience_counter >= early_stop_patience:
    #     print("🛑 Early stopping triggered!")
    #     break

# # === Save model and vocabs ===
torch.save(model.state_dict(), "transformer_model.pth")
torch.save(eng_vocab, "eng_vocab.pth")
torch.save(spa_vocab, "spa_vocab.pth")