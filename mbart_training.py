from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import get_scheduler
from tqdm import tqdm
import evaluate
from sklearn.model_selection import train_test_split

class TranslationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src = self.df.iloc[idx]["english"]
        tgt = self.df.iloc[idx]["spanish"]

        batch = self.tokenizer(
            src,
            text_target=tgt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {k: v.squeeze(0) for k, v in batch.items()}

def collate_fn(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    loop = tqdm(dataloader, desc="Training", leave=True)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss = loss.mean()
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    loop.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    bleu = evaluate.load("bleu")

    preds = []
    refs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            generated_ids = model.module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                forced_bos_token_id=tokenizer.lang_code_to_id[TGT_LANG],
                max_length=128,
                num_beams=5,
                early_stopping=True
            )

            pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

            preds.extend(pred_texts)
            refs.extend([[r] for r in ref_texts])  # BLEU expects list of refs

    score = bleu.compute(predictions=preds, references=refs)
    return score["bleu"]


MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
SRC_LANG = "en_XX"
TGT_LANG = "es_XX"

tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.src_lang = SRC_LANG
tokenizer.tgt_lang = TGT_LANG

model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
model = nn.DataParallel(model)  # for multi-GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Load and split dataset
df = pd.read_csv("data/data.csv")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Create datasets and loaders
train_dataset = TranslationDataset(train_df, tokenizer)
val_dataset = TranslationDataset(val_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=5e-5)
num_training_steps = len(train_loader) * 5
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)


# Training loop
for epoch in range(5):
    print(f"\nEpoch {epoch+1}")
    train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, device)
    bleu_score = evaluate(model, val_loader, tokenizer, device)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation BLEU Score: {bleu_score:.4f}")

# Save the model
model.module.save_pretrained("finetuned_mbart")
tokenizer.save_pretrained("finetuned_mbart")