from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src" / "model"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

from config_model import Config
from experiments.phobert.modeling import PhoBertLoraQA
from evalmodel import (
    evaluate_loss,
    evaluate_qa_metrics,
    plot_accuracy_f1_recall_per_epoch,
    plot_loss_curves,
    qa_eval_collate,
)
from src.data_loader import build_qa_datasets, load_raw_datasets, prepare_metric_raw_examples


CONFIG_PATH = ROOT / "experiments" / "phobert" / "config.yaml"


def save_history(history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def save_checkpoint(checkpoint_dir, model, optimizer, tokenizer, config, epoch, train_loss, val_loss, best_val_loss, history):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "history": history,
            "config": config.__dict__,
        },
        checkpoint_dir / "training_state.pt",
    )
    tokenizer.save_pretrained(checkpoint_dir)
    config.to_yaml(checkpoint_dir / "config.yaml")
    save_history(history, checkpoint_dir)


def _config_with_threshold(config, metrics):
    if not metrics or "no_answer_threshold" not in metrics:
        return config
    checkpoint_config = copy.copy(config)
    checkpoint_config.no_answer_threshold = float(metrics["no_answer_threshold"])
    checkpoint_config.tune_no_answer_threshold = False
    return checkpoint_config


class PhoBertTrainer:
    def __init__(self, profile_name):
        self.profile_name = profile_name
        self.config = Config.from_yaml(path=CONFIG_PATH, profile=profile_name)
        self.device = self._resolve_device()
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.metric_eval_loader = None
        self.metric_raw_examples = []
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.best_val_loss = float("inf")
        self.best_metric_score = None
        self.history = {"train_loss": [], "val_loss": []}

    @property
    def output_dir(self):
        return Path(self.config.output_dir)

    def _resolve_device(self):
        if getattr(self.config, "force_cpu", False):
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def setup_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
        model_config = AutoConfig.from_pretrained(self.config.model_name)
        max_supported_length = int(getattr(model_config, "max_position_embeddings", 0) or 0) - 2
        if max_supported_length > 0 and int(self.config.max_length) > max_supported_length:
            print(
                f"Reducing max_length from {self.config.max_length} to "
                f"{max_supported_length} for {self.config.model_name} position embeddings."
            )
            self.config.max_length = max_supported_length
        if not self.tokenizer.is_fast:
            print(
                f"{self.config.model_name} uses a slow tokenizer; "
                "using manual QA span offsets."
            )
        return self.tokenizer

    def setup_dataloaders(self):
        if self.tokenizer is None:
            self.setup_tokenizer()

        datasets = build_qa_datasets(self.tokenizer, self.config)
        train_data = datasets["train"]
        train_sampler = self._build_train_sampler(train_data)
        self.train_loader = DataLoader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
        )
        self.val_loader = DataLoader(datasets["validation"], batch_size=self.config.batch_size, shuffle=False)
        self.setup_metric_eval_loader()
        return self.train_loader, self.val_loader

    def setup_metric_eval_loader(self):
        if not getattr(self.config, "track_eval_metrics", True):
            return None
        if not self.config.validation_file:
            return None

        eval_config = copy.copy(self.config)
        eval_config.train_file = None
        eval_config.test_file = None

        raw_datasets = load_raw_datasets(eval_config)
        if "validation" not in raw_datasets:
            return None

        self.metric_raw_examples = prepare_metric_raw_examples(
            [dict(row) for row in raw_datasets["validation"]],
            eval_config,
        )
        eval_datasets = build_qa_datasets(self.tokenizer, eval_config, is_training=False)
        self.metric_eval_loader = DataLoader(
            eval_datasets["validation"],
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=qa_eval_collate,
        )
        print(f"Metric eval enabled: {len(self.metric_raw_examples):,} validation samples")
        return self.metric_eval_loader

    def _build_train_sampler(self, train_data):
        if not getattr(self.config, "use_question_group_sampler", False):
            return None
        if "question_group" not in train_data.column_names:
            print("Question-group sampler disabled: missing question_group column.")
            return None

        groups = list(train_data.with_format(None)["question_group"])
        counts = collections.Counter(groups)
        power = float(getattr(self.config, "question_group_sampling_power", 0.5))
        weights = torch.tensor(
            [1.0 / (counts[group] ** power) for group in groups],
            dtype=torch.double,
        )
        print(
            "Question-group sampler enabled: "
            f"{len(counts)} groups, power={power:.2f}, "
            f"min_count={min(counts.values()):,}, max_count={max(counts.values()):,}"
        )
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    def setup_model(self):
        self.model = PhoBertLoraQA(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=float(getattr(self.config, "weight_decay", 0.0)),
        )
        total_steps = max(len(self.train_loader) * int(self.config.epochs), 1)
        warmup_steps = int(total_steps * float(getattr(self.config, "warmup_ratio", 0.0)))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        self.model.train()
        return self.model

    def setup(self):
        self.setup_tokenizer()
        self.setup_dataloaders()
        self.setup_model()

    def train_one_epoch(self, epoch_index):
        total_loss = 0.0
        progress = tqdm(self.train_loader, desc=f"{self.profile_name} epoch {epoch_index + 1}/{self.config.epochs}")

        for batch in progress:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            start_positions = batch["start_positions"].to(self.device)
            end_positions = batch["end_positions"].to(self.device)

            self.optimizer.zero_grad()
            start_logits, end_logits = self.model(input_ids, attention_mask)
            loss = (
                self.loss_fn(start_logits, start_positions)
                + self.loss_fn(end_logits, end_positions)
            ) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                float(getattr(self.config, "max_grad_norm", 1.0)),
            )
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    def evaluate(self):
        return evaluate_loss(self.model, self.val_loader, self.loss_fn, self.device)

    def evaluate_metrics(self):
        if self.metric_eval_loader is None or not self.metric_raw_examples:
            return {}
        return evaluate_qa_metrics(
            self.model,
            self.metric_eval_loader,
            self.tokenizer,
            self.metric_raw_examples,
            self.device,
            no_answer_threshold=float(getattr(self.config, "no_answer_threshold", 0.0)),
            tune_no_answer_threshold=bool(getattr(self.config, "tune_no_answer_threshold", False)),
        )

    def _best_metric_value(self, val_loss, metrics):
        metric_name = getattr(self.config, "best_metric", "val_loss")
        if metric_name in {"loss", "val_loss"}:
            return "val_loss", val_loss, False
        if metrics and metric_name in metrics:
            return metric_name, float(metrics[metric_name]), True
        return "val_loss", val_loss, False

    def save_best_if_needed(self, epoch_number, train_loss, val_loss, metrics=None):
        min_delta = float(getattr(self.config, "early_stopping_min_delta", 0.0))
        metric_name, metric_value, higher_is_better = self._best_metric_value(val_loss, metrics)
        if self.best_metric_score is None:
            is_best = True
        elif higher_is_better:
            is_best = metric_value > self.best_metric_score + min_delta
        else:
            is_best = metric_value < self.best_metric_score - min_delta

        if not is_best:
            self.best_val_loss = min(self.best_val_loss, val_loss)
            return False

        self.best_metric_score = metric_value
        self.best_val_loss = min(self.best_val_loss, val_loss)
        if getattr(self.config, "save_best_model", True):
            best_dir = self.output_dir / "best_model"
            checkpoint_config = _config_with_threshold(self.config, metrics)
            save_checkpoint(
                best_dir,
                self.model,
                self.optimizer,
                self.tokenizer,
                checkpoint_config,
                epoch_number,
                train_loss,
                val_loss,
                self.best_val_loss,
                self.history,
            )
            print(f"Saved best model: {best_dir} ({metric_name}={metric_value:.4f})")
        return True

    def train(self):
        if self.model is None:
            self.setup()

        print(f"Training PhoBERT profile: {self.profile_name}")
        print(f"Model: {self.config.model_name}")
        print(f"Train file: {self.config.train_file}")
        print(f"Validation file: {self.config.validation_file}")
        print(f"Output dir: {self.output_dir}")

        patience = getattr(self.config, "early_stopping_patience", None)
        patience = None if patience is None else int(patience)
        no_improve_epochs = 0

        for epoch in range(self.config.epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.evaluate()
            metrics = self.evaluate_metrics()
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            if metrics:
                self.history.setdefault("em", []).append(metrics["exact_match"])
                self.history.setdefault("accuracy", []).append(metrics["accuracy"])
                self.history.setdefault("precision", []).append(metrics["precision"])
                self.history.setdefault("recall", []).append(metrics["recall"])
                self.history.setdefault("f1", []).append(metrics["f1"])
                self.history.setdefault("has_answer_f1", []).append(metrics["has_answer_f1"])
                self.history.setdefault("no_answer_exact", []).append(metrics["no_answer_exact"])
                self.history.setdefault("no_answer_threshold", []).append(metrics["no_answer_threshold"])
            save_history(self.history, self.output_dir)
            metric_text = ""
            if metrics:
                metric_text = (
                    f", accuracy={metrics['accuracy']:.2f}%, "
                    f"precision={metrics['precision']:.2f}%, "
                    f"recall={metrics['recall']:.2f}%, "
                    f"f1={metrics['f1']:.2f}%, "
                    f"has_answer_f1={metrics['has_answer_f1']:.2f}%, "
                    f"no_answer_exact={metrics['no_answer_exact']:.2f}%, "
                    f"no_answer_threshold={metrics['no_answer_threshold']:.4f}"
                )
            print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}{metric_text}")
            improved = self.save_best_if_needed(epoch + 1, train_loss, val_loss, metrics)
            if improved:
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if patience is not None and no_improve_epochs >= patience:
                print(
                    "Early stopping: "
                    f"val_loss did not improve by at least "
                    f"{float(getattr(self.config, 'early_stopping_min_delta', 0.0)):.6f} "
                    f"for {patience} epoch(s)."
                )
                break

        plot_loss_curves(self.history["train_loss"], self.history["val_loss"], self.output_dir)
        plot_loss_curves(self.history["train_loss"], self.history["val_loss"], self.output_dir / "best_model")
        if self.history.get("f1") and self.history.get("recall"):
            plot_accuracy_f1_recall_per_epoch(
                self.history.get("accuracy", self.history.get("em", [])),
                self.history["f1"],
                self.history["recall"],
                self.output_dir,
                precision_list=self.history.get("precision", []),
            )
            plot_accuracy_f1_recall_per_epoch(
                self.history.get("accuracy", self.history.get("em", [])),
                self.history["f1"],
                self.history["recall"],
                self.output_dir / "best_model",
                precision_list=self.history.get("precision", []),
            )
        return {
            "profile": self.profile_name,
            "model_name": self.config.model_name,
            "output_dir": str(self.output_dir),
            "best_model_dir": str(self.output_dir / "best_model"),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    profiles = Config.available_profiles(path=CONFIG_PATH)
    parser.add_argument("--profile", default=None, choices=profiles)
    parser.add_argument("--profiles", nargs="+", default=None, choices=profiles)
    return parser.parse_args()


def resolve_profiles(args):
    if args.profile and args.profiles:
        raise ValueError("Chi dung mot trong hai tuy chon: --profile hoac --profiles.")
    if args.profile:
        return [args.profile]
    if args.profiles:
        return args.profiles
    return Config.default_pipeline_profiles(path=CONFIG_PATH)


def main():
    args = parse_args()
    records = []
    for profile in resolve_profiles(args):
        trainer = PhoBertTrainer(profile)
        records.append(trainer.train())

    if records:
        out_dir = Path(records[-1]["output_dir"])
        with open(out_dir / "phobert_training_runs.json", "w", encoding="utf-8") as f:
            json.dump({"runs": records}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
