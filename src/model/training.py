import argparse
import collections
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config_model import Config
from evalmodel import (
    evaluate_loss,
    evaluate_qa_metrics,
    plot_accuracy_f1_recall_per_epoch,
    plot_loss_curves,
    qa_eval_collate,
)
from loadmodel import CustomDistilBertQA
from src.data_loader import build_qa_datasets, load_raw_datasets, prepare_metric_raw_examples

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


def _config_with_threshold(config, metrics):
    if not metrics or "no_answer_threshold" not in metrics:
        return config
    checkpoint_config = copy.copy(config)
    checkpoint_config.no_answer_threshold = float(metrics["no_answer_threshold"])
    checkpoint_config.tune_no_answer_threshold = False
    return checkpoint_config


def save_loss_plot(history, output_dir):
    if not history:
        return

    train_losses = history.get("train_loss", [])
    val_losses = history.get("val_loss", [])
    if train_losses and val_losses:
        plot_loss_curves(train_losses, val_losses, Path(output_dir))

    f1_scores = history.get("f1", [])
    recall_scores = history.get("recall", [])
    if f1_scores and recall_scores:
        plot_accuracy_f1_recall_per_epoch(
            history.get("accuracy", history.get("em", [])),
            f1_scores,
            recall_scores,
            Path(output_dir),
            precision_list=history.get("precision", []),
        )


def save_pipeline_history(step_records, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "pipeline_history.json", "w", encoding="utf-8") as f:
        json.dump({"steps": step_records}, f, ensure_ascii=False, indent=2)


def save_pipeline_loss_plot(step_records, output_dir):
    if not step_records:
        return

    train_losses = []
    val_losses = []
    step_starts = []
    cursor = 1

    for record in step_records:
        history = record.get("history", {})
        step_train = history.get("train_loss", [])
        step_val = history.get("val_loss", [])
        if not step_train or not step_val:
            continue

        step_starts.append((cursor, record.get("profile", "step")))
        train_losses.extend(step_train)
        val_losses.extend(step_val)
        cursor += len(step_train)

    if not train_losses or not val_losses:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, marker="o", label="Train Loss", color="#2196F3")
    ax.plot(epochs, val_losses, marker="s", label="Val Loss", color="#F44336")

    for start_epoch, profile in step_starts[1:]:
        ax.axvline(start_epoch - 0.5, color="#555555", linestyle="--", linewidth=1)
        ax.text(
            start_epoch - 0.45,
            ax.get_ylim()[1],
            f"start {profile}",
            rotation=90,
            va="top",
            ha="left",
            fontsize=8,
            color="#333333",
        )

    ax.set_xlabel("Pipeline Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Pipeline Training & Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.savefig(output_dir / "pipeline_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'pipeline_loss_curves.png'}")


def save_pipeline_score_plot(step_records, output_dir):
    if not step_records:
        return

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    step_starts = []
    cursor = 1

    for record in step_records:
        history = record.get("history", {})
        step_f1 = history.get("f1", [])
        step_recall = history.get("recall", [])
        if not step_f1 or not step_recall:
            continue

        step_starts.append((cursor, record.get("profile", "step")))
        f1_scores.extend(step_f1)
        recall_scores.extend(step_recall)
        accuracy_scores.extend(history.get("accuracy", history.get("em", [])))
        precision_scores.extend(history.get("precision", []))
        cursor += len(step_f1)

    if not f1_scores or not recall_scores:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(f1_scores) + 1))
    fig, ax = plt.subplots(figsize=(9, 5))
    if len(accuracy_scores) == len(f1_scores):
        ax.plot(epochs, accuracy_scores, marker="o", label="Accuracy/EM", color="#4CAF50")
    if len(precision_scores) == len(f1_scores):
        ax.plot(epochs, precision_scores, marker="^", label="Precision", color="#2196F3")
    ax.plot(epochs, recall_scores, marker="d", label="Recall", color="#7E57C2")
    ax.plot(epochs, f1_scores, marker="s", label="F1 Score", color="#FF9800")

    for start_epoch, profile in step_starts[1:]:
        ax.axvline(start_epoch - 0.5, color="#555555", linestyle="--", linewidth=1)
        ax.text(
            start_epoch - 0.45,
            ax.get_ylim()[1],
            f"start {profile}",
            rotation=90,
            va="top",
            ha="left",
            fontsize=8,
            color="#333333",
        )

    ax.set_xlabel("Pipeline Epoch")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Pipeline Accuracy, Recall & F1")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.savefig(output_dir / "pipeline_accuracy_f1_recall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'pipeline_accuracy_f1_recall.png'}")


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

    state = torch.load(state_path, map_location=map_location, weights_only=False)
    try:
        model.load_state_dict(state["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture mismatch. If this checkpoint was trained "
            "with the old LoRA model, retrain the source checkpoint with the "
            "current full fine-tune model before using it for transfer."
        ) from exc

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
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = bool(
                getattr(self.config, "use_tf32", False)
            )
            torch.backends.cudnn.allow_tf32 = bool(
                getattr(self.config, "use_tf32", False)
            )
        self.use_amp = self.device == "cuda" and bool(getattr(self.config, "use_amp", False))
        self.amp_device_type = "cuda" if self.device == "cuda" else "cpu"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
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
        if "train" not in datasets or "validation" not in datasets:
            raise ValueError(
                "Training requires both train_file and validation_file. "
                f"Profile {self.profile_name!r} resolved splits: {list(datasets.keys())}."
            )
        train_data = datasets["train"]
        val_data = datasets["validation"]
        train_sampler = self._build_train_sampler(train_data)

        self.train_loader = DataLoader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            **self._dataloader_worker_kwargs(),
        )
        self.val_loader = DataLoader(
            val_data,
            batch_size=self.config.batch_size,
            shuffle=False,
            **self._dataloader_worker_kwargs(),
        )

        print(
            f"Tokenized features: train={len(train_data):,}, "
            f"validation={len(val_data):,}"
        )
        self.setup_metric_eval_loader()
        return self.train_loader, self.val_loader

    def _dataloader_worker_kwargs(self):
        num_workers = int(getattr(self.config, "num_workers", 0) or 0)
        kwargs = {
            "num_workers": num_workers,
            "pin_memory": self.device == "cuda"
            and bool(getattr(self.config, "pin_memory", False)),
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = bool(
                getattr(self.config, "persistent_workers", False)
            )
            prefetch_factor = getattr(self.config, "prefetch_factor", None)
            if prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(prefetch_factor)
        return kwargs

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
        eval_data = eval_datasets["validation"]
        self.metric_eval_loader = DataLoader(
            eval_data,
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
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )

    def setup_model(self):
        self.model = CustomDistilBertQA(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=float(getattr(self.config, "weight_decay", 0.0)),
        )
        accumulation_steps = max(
            1,
            int(getattr(self.config, "gradient_accumulation_steps", 1) or 1),
        )
        updates_per_epoch = (len(self.train_loader) + accumulation_steps - 1) // accumulation_steps
        total_steps = max(updates_per_epoch * int(self.config.epochs), 1)
        warmup_steps = int(total_steps * float(getattr(self.config, "warmup_ratio", 0.0)))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
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
        if load_optimizer_state and state.get("history"):
            self.history = state["history"]
        return state

    def setup(self):
        self.setup_tokenizer()
        self.setup_dataloaders()
        self.setup_model()

    def train_one_epoch(self, epoch_index):
        total_loss = 0.0
        accumulation_steps = max(
            1,
            int(getattr(self.config, "gradient_accumulation_steps", 1) or 1),
        )
        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch_index + 1}/{self.config.epochs}",
        )

        self.optimizer.zero_grad()
        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            start_positions = batch["start_positions"].to(self.device)
            end_positions = batch["end_positions"].to(self.device)

            with torch.amp.autocast(self.amp_device_type, enabled=self.use_amp):
                start_logits, end_logits = self.model(input_ids, attention_mask)
                start_loss = self.loss_fn(start_logits, start_positions)
                end_loss = self.loss_fn(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2
            raw_loss = loss.item()
            loss = loss / accumulation_steps

            self.scaler.scale(loss).backward()
            should_step = step % accumulation_steps == 0 or step == len(self.train_loader)
            if should_step:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(getattr(self.config, "max_grad_norm", 1.0)),
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += raw_loss
            progress.set_postfix(loss=f"{raw_loss:.4f}")

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

        if is_best:
            self.best_metric_score = metric_value
        self.best_val_loss = min(self.best_val_loss, val_loss)

        if getattr(self.config, "save_best_model", True) and is_best:
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
        return is_best

    def train(self):
        if self.train_loader is None or self.model is None:
            self.setup()

        print(f"Training profile: {self.profile_name}")
        print(f"Train file: {self.config.train_file}")
        print(f"Validation file: {self.config.validation_file}")
        print(f"Output dir: {self.output_dir}")
        if self.init_checkpoint_dir:
            print(f"Initial model dir: {self.init_checkpoint_dir}")

        patience = getattr(self.config, "early_stopping_patience", None)
        patience = None if patience is None else int(patience)
        no_improve_epochs = 0

        for epoch in range(self.config.epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.evaluate()
            metrics = self.evaluate_metrics()
            self.history.setdefault("train_loss", []).append(train_loss)
            self.history.setdefault("val_loss", []).append(val_loss)
            if metrics:
                self.history.setdefault("em", []).append(metrics["exact_match"])
                self.history.setdefault("accuracy", []).append(metrics["accuracy"])
                self.history.setdefault("precision", []).append(metrics["precision"])
                self.history.setdefault("recall", []).append(metrics["recall"])
                self.history.setdefault("f1", []).append(metrics["f1"])
                self.history.setdefault("has_answer_f1", []).append(metrics["has_answer_f1"])
                self.history.setdefault("no_answer_exact", []).append(metrics["no_answer_exact"])
                self.history.setdefault("no_answer_threshold", []).append(metrics["no_answer_threshold"])

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
            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}"
                f"{metric_text}"
            )

            # May save a best-model checkpoint, which can directly embed the current in-memory history
            improved = self.save_best_if_needed(epoch + 1, train_loss, val_loss, metrics)
            if improved:
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # Persist training_history.json once per epoch, after any best-checkpoint update
            save_history(self.history, self.output_dir)

            if patience is not None and no_improve_epochs >= patience:
                print(
                    "Early stopping: "
                    f"val_loss did not improve by at least "
                    f"{float(getattr(self.config, 'early_stopping_min_delta', 0.0)):.6f} "
                    f"for {patience} epoch(s)."
                )
                break

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
        save_pipeline_loss_plot(step_records, step_records[-1]["output_dir"])
        save_pipeline_score_plot(step_records, step_records[-1]["output_dir"])

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
