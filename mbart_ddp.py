import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, get_scheduler
import pandas as pd
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import evaluate
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.set_float32_matmul_precision("high")

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

def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()



def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    rank = dist.get_rank()
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1} | GPU {rank}", position=rank, leave=True, dynamic_ncols=True, disable=False)
    
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
        scheduler.step()

    loop.set_postfix(loss=loss.item())
    # print(f"Rank {dist.get_rank()} | Step running on GPU: {device} | Loss: {loss.item()}")
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate_epoch(model, dataloader, tokenizer, device, max_samples=None, use_beam=True):
    model.eval()
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    preds, refs = [], []

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if max_samples and idx * dataloader.batch_size >= max_samples:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            generated_ids = model.module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"],
                max_length=128,
                num_beams=5 if use_beam else 1,
                early_stopping=True
            )

            pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

            preds.extend(pred_texts)
            refs.extend([[r] for r in ref_texts])  # BLEU expects list of list of references

    if dist.get_rank() == 0:
        bleu_score = bleu.compute(predictions=preds, references=refs)["bleu"]
        rouge_score = rouge.compute(predictions=preds, references=refs)["rougeL"]
        return {
            "bleu": bleu_score,
            "rougeL": rouge_score
        }
    else:
        return None


MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
SRC_LANG = "en_XX"
TGT_LANG = "es_XX"

local_rank = setup_ddp()
device = torch.device(f"cuda:{local_rank}")

tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.src_lang = SRC_LANG
tokenizer.tgt_lang = TGT_LANG

model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
model.gradient_checkpointing_enable()
model = model.to(device)
model = DDP(model, device_ids=[local_rank])

# Load and split dataset
df = pd.read_csv("data/data.csv")
# df = df.iloc[:1000, :]
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = TranslationDataset(train_df, tokenizer)
val_dataset = TranslationDataset(val_df, tokenizer)

train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, sampler=val_sampler, collate_fn=collate_fn)


best_bleu = 0.0
epochs = 15
# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=5e-5)
num_training_steps = len(train_loader) * 5
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

for epoch in range(epochs):
    train_sampler.set_epoch(epoch)

    print(f"\n[GPU {local_rank}] Epoch {epoch+1}")
    train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, device)

    if dist.get_rank() == 0:
        metrics = evaluate_epoch(model, val_loader, tokenizer, device)
        print(f"BLEU: {metrics['bleu']:.4f} | ROUGE-L: {metrics['rougeL']:.4f}")

        if metrics["bleu"] > best_bleu:
            best_bleu = metrics["bleu"]
            model.module.save_pretrained("finetuned_mbart_ddp")
            tokenizer.save_pretrained("finetuned_mbart_ddp")


cleanup_ddp()