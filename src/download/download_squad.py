"""
Download rajpurkar/squad from Hugging Face and save as JSONL to data/data_train/
"""

from pathlib import Path
import json

from datasets import load_dataset

OUT_DIR = Path("data/data_train")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_jsonl(split_name: str, hf_split):
    path = OUT_DIR / f"{split_name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in hf_split:
            record = {
                "id": row["id"],
                "question": row["question"],
                "context": row["context"],
                "answers": {
                    "text": row["answers"]["text"],
                    "answer_start": row["answers"]["answer_start"],
                },
                "is_impossible": False,
                "language": "en",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(hf_split):,} samples → {path}")


print("Loading rajpurkar/squad ...")
ds = load_dataset("rajpurkar/squad")

save_jsonl("train", ds["train"])
save_jsonl("validation", ds["validation"])

print("Done.")
