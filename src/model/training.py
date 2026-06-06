import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from transformers import AutoTokenizer
import torch
import torch.nn as nn
from loadmodel import CustomLoraDistilBertQA

from config_model import Config
from src.data_loader import build_qa_datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm



#Lựa chọn dữ liệu train
config = Config.from_yaml(profile="train_en")
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

datasets = build_qa_datasets(tokenizer, config)
#Gọi data train
train_data = datasets["train"]

print(train_data)
print(train_data[0].keys())
print("input_ids shape:", train_data[0]["input_ids"].shape)
print("attention_mask shape:", train_data[0]["attention_mask"].shape)
print("start_positions:", train_data[0]["start_positions"])
print("end_positions:", train_data[0]["end_positions"])


train_loader = DataLoader(
    train_data,
    batch_size=config.batch_size,
    shuffle=True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CustomLoraDistilBertQA().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
loss_fn = nn.CrossEntropyLoss()

model.train()

for epoch in range(config.epochs):
    total_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        optimizer.zero_grad()

        start_logits, end_logits = model(input_ids, attention_mask)

        start_loss = loss_fn(start_logits, start_positions)
        end_loss = loss_fn(end_logits, end_positions)
        loss = (start_loss + end_loss) / 2

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    print(f"Epoch {epoch + 1} average loss: {total_loss / len(train_loader):.4f}")
