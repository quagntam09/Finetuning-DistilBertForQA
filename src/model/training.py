import sys
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from config_model import Config
from src.data_loader import build_qa_datasets

config = Config.from_yaml()
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

datasets = build_qa_datasets(tokenizer, config)
train_data = datasets["train"]

print(train_data)
print(train_data[0].keys())
print("input_ids shape:", train_data[0]["input_ids"].shape)
print("attention_mask shape:", train_data[0]["attention_mask"].shape)
print("start_positions:", train_data[0]["start_positions"])
print("end_positions:", train_data[0]["end_positions"])
