import argparse
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
from evalmodel import evaluate_loss
from loadmodel import CustomLoraDistilBertQA
from src.data_loader import build_qa_datasets


DEFAULT_PROFILE_NAME = "train_en"


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
            "config": config.__dict__,
        },
        checkpoint_dir / "training_state.pt",
    )
    tokenizer.save_pretrained(checkpoint_dir)
    config.to_yaml(checkpoint_dir / "config.yaml")


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
        profile_name=DEFAULT_PROFILE_NAME,
        init_checkpoint_dir=None,
        output_dir=None,
        train_file=None,
        validation_file=None,
        load_optimizer_state=None,
        use_profile_checkpoint=True,
    ):
        self.profile_name = profile_name
        self.config = Config.from_yaml(profile=profile_name)

        if init_checkpoint_dir is not None:
            self.config.init_checkpoint_dir = str(init_checkpoint_dir)
        elif not use_profile_checkpoint:
            self.config.init_checkpoint_dir = None
        if output_dir is not None:
            self.config.output_dir = str(output_dir)
        if train_file is not None:
            self.config.train_file = str(train_file)
        if validation_file is not None:
            self.config.validation_file = str(validation_file)
        if load_optimizer_state is not None:
            self.config.load_optimizer_state = load_optimizer_state

        self.device = self._resolve_device()
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.best_val_loss = float("inf")

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
            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}"
            )
            self.save_best_if_needed(epoch + 1, train_loss, val_loss)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE_NAME,
        choices=Config.available_profiles(),
        help="Profile trong config/model.yaml dùng để train từ base model.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Thu muc luu checkpoint moi. Neu bo trong se lay tu profile.",
    )
    parser.add_argument(
        "--train-file",
        default=None,
        help="File train jsonl. Neu bo trong se lay train_file tu profile.",
    )
    parser.add_argument(
        "--validation-file",
        default=None,
        help="File validation jsonl. Neu bo trong se lay validation_file tu profile.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = QATrainer(
        profile_name=args.profile,
        output_dir=args.output_dir,
        train_file=args.train_file,
        validation_file=args.validation_file,
        use_profile_checkpoint=False,
    )
    trainer.train()


if __name__ == "__main__":
    main()
