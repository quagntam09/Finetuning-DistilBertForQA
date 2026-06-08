import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config_model import Config
from evalmodel import evaluate_loss, plot_loss_curves
from loadmodel import CustomLoraDistilBertQA
from src.data_loader import build_qa_datasets

def save_checkpoint(
    checkpoint_dir,
    model,
    optimizer,
    tokenizer,
    config,
    epoch,
    train_loss,
    val_loss,
    best_val_loss,
    history=None,
):
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


def save_history(history, output_dir):
    if history is None:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def update_checkpoint_history(checkpoint_dir, history):
    checkpoint_dir = Path(checkpoint_dir)
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.exists():
        return

    state = torch.load(state_path, map_location="cpu")
    state["history"] = history
    torch.save(state, state_path)
    save_history(history, checkpoint_dir)


def save_loss_plot(history, output_dir):
    if not history:
        return

    train_losses = history.get("train_loss", [])
    val_losses = history.get("val_loss", [])
    if train_losses and val_losses:
        plot_loss_curves(train_losses, val_losses, Path(output_dir))


def save_pipeline_history(step_records, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "pipeline_history.json", "w", encoding="utf-8") as f:
        json.dump({"steps": step_records}, f, ensure_ascii=False, indent=2)


def load_checkpoint(
    checkpoint_dir,
    model,
    optimizer=None,
    map_location="cpu",
    load_optimizer_state=False,
):
    checkpoint_dir = Path(checkpoint_dir)
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {state_path}")

    state = torch.load(state_path, map_location=map_location)
    model.load_state_dict(state["model_state_dict"])

    if load_optimizer_state and optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    return state


class QATrainer:
    def __init__(
        self,
        profile_name=None,
    ):
        self.profile_name = profile_name
        self.config = Config.from_yaml(profile=profile_name)

        self.device = self._resolve_device()
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": []}

    def _resolve_device(self):
        if getattr(self.config, "force_cpu", False):
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def init_checkpoint_dir(self):
        return getattr(self.config, "init_checkpoint_dir", None)

    @property
    def output_dir(self):
        return Path(self.config.output_dir)

    def setup_tokenizer(self):
        tokenizer_source = self.init_checkpoint_dir or self.config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        return self.tokenizer

    def setup_dataloaders(self):
        if self.tokenizer is None:
            self.setup_tokenizer()

        datasets = build_qa_datasets(self.tokenizer, self.config)
        train_data = datasets["train"]
        val_data = datasets["validation"]

        self.train_loader = DataLoader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            val_data,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        self._print_dataset_info(train_data)
        return self.train_loader, self.val_loader

    def _print_dataset_info(self, train_data):
        print(train_data)
        print(train_data[0].keys())
        print("input_ids shape:", train_data[0]["input_ids"].shape)
        print("attention_mask shape:", train_data[0]["attention_mask"].shape)
        print("start_positions:", train_data[0]["start_positions"])
        print("end_positions:", train_data[0]["end_positions"])

    def setup_model(self):
        self.model = CustomLoraDistilBertQA(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        self.load_initial_checkpoint()
        self.model.train()
        return self.model

    def load_initial_checkpoint(self):
        if not self.init_checkpoint_dir:
            return None

        load_optimizer_state = getattr(self.config, "load_optimizer_state", False)
        state = load_checkpoint(
            self.init_checkpoint_dir,
            self.model,
            optimizer=self.optimizer,
            map_location=self.device,
            load_optimizer_state=load_optimizer_state,
        )
        print(
            f"Loaded model weights from {self.init_checkpoint_dir} "
            f"(checkpoint epoch={state.get('epoch')})"
        )
        # Always restore history when present, independent of optimizer state loading
        if state.get("history"):
            self.history = state["history"]
        return state

    def setup(self):
        self.setup_tokenizer()
        self.setup_dataloaders()
        self.setup_model()

    def train_one_epoch(self, epoch_index):
        total_loss = 0.0
        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch_index + 1}/{self.config.epochs}",
        )

        for batch in progress:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            start_positions = batch["start_positions"].to(self.device)
            end_positions = batch["end_positions"].to(self.device)

            self.optimizer.zero_grad()

            start_logits, end_logits = self.model(input_ids, attention_mask)

            start_loss = self.loss_fn(start_logits, start_positions)
            end_loss = self.loss_fn(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    def evaluate(self):
        return evaluate_loss(self.model, self.val_loader, self.loss_fn, self.device)

    def save_best_if_needed(self, epoch_number, train_loss, val_loss):
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss

        if getattr(self.config, "save_best_model", True) and is_best:
            best_dir = self.output_dir / "best_model"
            save_checkpoint(
                best_dir,
                self.model,
                self.optimizer,
                self.tokenizer,
                self.config,
                epoch_number,
                train_loss,
                val_loss,
                self.best_val_loss,
                self.history,
            )
            print(f"Saved best model: {best_dir}")

    def train(self):
        if self.train_loader is None or self.model is None:
            self.setup()

        print(f"Training profile: {self.profile_name}")
        print(f"Train file: {self.config.train_file}")
        print(f"Validation file: {self.config.validation_file}")
        print(f"Output dir: {self.output_dir}")
        if self.init_checkpoint_dir:
            print(f"Initial model dir: {self.init_checkpoint_dir}")

        for epoch in range(self.config.epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.evaluate()
            self.history.setdefault("train_loss", []).append(train_loss)
            self.history.setdefault("val_loss", []).append(val_loss)
            save_history(self.history, self.output_dir)
            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}"
            )
            self.save_best_if_needed(epoch + 1, train_loss, val_loss)
            update_checkpoint_history(self.output_dir / "best_model", self.history)

        save_loss_plot(self.history, self.output_dir)
        save_loss_plot(self.history, self.output_dir / "best_model")
        return {
            "profile": self.profile_name,
            "output_dir": str(self.output_dir),
            "best_model_dir": str(self.output_dir / "best_model"),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default=None,
        choices=Config.available_profiles(),
        help=(
            "Train mot profile duy nhat trong config/model.yaml. "
            "Neu bo trong se chay pipeline mac dinh."
        ),
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        choices=Config.available_profiles(),
        help=(
            "Danh sach profile train lien tiep. "
            "Neu bo trong se lay default_pipeline_profiles tu config/model.yaml."
        ),
    )
    return parser.parse_args()


def resolve_profile_sequence(args):
    if args.profile and args.profiles:
        raise ValueError("Chi dung mot trong hai tuy chon: --profile hoac --profiles.")

    if args.profiles:
        return args.profiles

    if args.profile:
        return [args.profile]

    default_profiles = Config.default_pipeline_profiles()
    if not default_profiles:
        raise ValueError("Chua cau hinh default_pipeline_profiles trong config/model.yaml.")

    return default_profiles


def run_single_profile(profile_name):
    trainer = QATrainer(profile_name=profile_name)
    return [trainer.train()]


def run_profile_pipeline(profile_names):
    step_records = []

    for step_index, profile_name in enumerate(profile_names, start=1):
        print(f"\n{'=' * 60}")
        print(f"Pipeline step {step_index}/{len(profile_names)}: {profile_name}")
        print(f"{'=' * 60}")

        trainer = QATrainer(profile_name=profile_name)
        record = trainer.train()
        step_records.append(record)

    if step_records:
        save_pipeline_history(step_records, step_records[-1]["output_dir"])

    return step_records


def main():
    args = parse_args()
    profile_names = resolve_profile_sequence(args)

    is_single_profile = len(profile_names) == 1 and args.profile is not None
    if is_single_profile:
        run_single_profile(profile_names[0])
    else:
        run_profile_pipeline(profile_names)


if __name__ == "__main__":
    main()
