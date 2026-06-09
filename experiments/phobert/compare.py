from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src" / "model"))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config_model import Config
from evalmodel import evaluate_qa_metrics, plot_compare_checkpoints, qa_eval_collate
from experiments.phobert.modeling import PhoBertLoraQA
from loadmodel import CustomLoraDistilBertQA
from src.data_loader import build_qa_datasets, prepare_metric_raw_examples


def load_checkpoint_config(checkpoint_dir):
    config_path = Path(checkpoint_dir) / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    return Config.from_yaml(path=config_path)


def load_raw_examples(config):
    val_file = config.validation_file or config.test_file
    if not val_file:
        raise ValueError("Checkpoint config must define validation_file or test_file.")

    path = Path(val_file)
    if not path.is_absolute():
        path = ROOT / path

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_model(config):
    if getattr(config, "model_family", None) == "phobert" or str(config.model_name).startswith("vinai/phobert"):
        return PhoBertLoraQA(config)
    return CustomLoraDistilBertQA(config)


def evaluate_checkpoint(checkpoint_dir, device):
    checkpoint_dir = Path(checkpoint_dir)
    config = load_checkpoint_config(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir if checkpoint_dir.exists() else config.model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError(
            f"{config.model_name} tokenizer does not provide fast offsets. "
            "QA metric evaluation needs return_offsets_mapping."
        )
    tokenized = build_qa_datasets(tokenizer, config, is_training=False)
    eval_split = tokenized.get("validation", tokenized.get("test"))
    if eval_split is None:
        raise RuntimeError(f"No validation/test split for {checkpoint_dir}")

    eval_loader = DataLoader(
        eval_split,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=qa_eval_collate,
    )
    raw_examples = prepare_metric_raw_examples(load_raw_examples(config), config)

    model = build_model(config).to(device)
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing checkpoint state: {state_path}")
    state = torch.load(state_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    metrics = evaluate_qa_metrics(
        model,
        eval_loader,
        tokenizer,
        raw_examples,
        device,
        no_answer_threshold=float(getattr(config, "no_answer_threshold", 0.0)),
        tune_no_answer_threshold=bool(getattr(config, "tune_no_answer_threshold", False)),
    )
    return config, metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Danh sach thu muc best_model can so sanh.",
    )
    parser.add_argument("--save-dir", default="outputs/compare_vi_models")
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = []
    for checkpoint in args.checkpoints:
        config, metrics = evaluate_checkpoint(checkpoint, device)
        label = Path(checkpoint).parent.name
        print(f"[{label}] EM={metrics['exact_match']:.2f}% F1={metrics['f1']:.2f}%")

        record = {
            "label": label,
            "checkpoint": str(checkpoint),
            "model_name": config.model_name,
            "em": metrics["exact_match"],
            "f1": metrics["f1"],
        }
        records.append(record)

        with open(save_dir / f"eval_{label}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    with open(save_dir / "compare_results.json", "w", encoding="utf-8") as f:
        json.dump({"records": records}, f, ensure_ascii=False, indent=2)

    plot_compare_checkpoints(records, save_dir)
    print(f"Saved comparison: {save_dir}")


if __name__ == "__main__":
    main()
