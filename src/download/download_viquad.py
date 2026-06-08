"""
Download taidng/UIT-ViQuAD2.0 from Hugging Face and save as JSONL to data/data_vi/
"""

from pathlib import Path
import json

from datasets import load_dataset

OUT_DIR = Path("data/data_vi")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_jsonl(split_name: str, hf_split):
    path = OUT_DIR / f"{split_name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in hf_split:
            ans = row["answers"]
            record = {
                "id": row["id"],
                "question": row["question"],
                "context": row["context"],
                "answers": {
                    "text": ans["text"] if ans else [],
                    "answer_start": ans["answer_start"] if ans else [],
                },
                "is_impossible": row["is_impossible"],
                "language": "vi",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(hf_split):,} samples → {path}")


print("Loading taidng/UIT-ViQuAD2.0 ...")
ds = load_dataset("taidng/UIT-ViQuAD2.0")

save_jsonl("train", ds["train"])
save_jsonl("validation", ds["validation"])
save_jsonl("test", ds["test"])

print("Done.")
