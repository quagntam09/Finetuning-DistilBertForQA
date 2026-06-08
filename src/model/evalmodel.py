from __future__ import annotations

import collections
import json
import re
import string
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ─── Metrics helpers ──────────────────────────────────────────────────────────

def _normalize_answer(s: str) -> str:
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())

def _get_tokens(s: str) -> list[str]:
    return _normalize_answer(s).split() if s else []

def compute_exact(prediction: str, ground_truth: str) -> int:
    return int(_normalize_answer(prediction) == _normalize_answer(ground_truth))

def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens  = _get_tokens(prediction)
    truth_tokens = _get_tokens(ground_truth)
    common   = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)

def _best_scores(prediction: str, gold_answers: list[str]) -> tuple[int, float]:
    if not gold_answers:
        return 0, 0.0
    return (
        max(compute_exact(prediction, a) for a in gold_answers),
        max(compute_f1(prediction, a)    for a in gold_answers),
    )


# ─── evaluate_loss (training.py gọi mỗi epoch) ───────────────────────────────

def evaluate_loss(model, val_loader, loss_fn, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids       = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions   = batch["end_positions"].to(device)
            start_logits, end_logits = model(input_ids, attention_mask)
            loss = (loss_fn(start_logits, start_positions) + loss_fn(end_logits, end_positions)) / 2
            total_loss += loss.item()
    model.train()
    return total_loss / max(len(val_loader), 1)


# ─── evaluate_qa_metrics (tính EM / F1) ──────────────────────────────────────

def evaluate_qa_metrics(
    model,
    eval_loader,
    tokenizer,
    raw_examples: list[dict],
    device: str,
    n_best: int = 20,
    max_answer_length: int = 30,
) -> dict:
    model.eval()
    all_start_logits, all_end_logits = [], []
    all_sample_ids, all_offset_maps  = [], []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Inference", leave=False):
            s_logits, e_logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            all_start_logits.extend(s_logits.cpu().numpy())
            all_end_logits.extend(e_logits.cpu().numpy())

            sids = batch.get("sample_id", [])
            offs = batch.get("offset_mapping", [])
            if hasattr(sids, "tolist"): sids = sids.tolist()
            if hasattr(offs, "tolist"): offs = offs.tolist()
            all_sample_ids.extend(sids)
            all_offset_maps.extend(offs)

    model.train()

    sample_features: dict[int, list[dict]] = collections.defaultdict(list)
    for sid, s_log, e_log, off in zip(all_sample_ids, all_start_logits, all_end_logits, all_offset_maps):
        sample_features[sid].append({"start_logits": s_log, "end_logits": e_log, "offset_mapping": off})

    predictions, references = [], []
    for sample_idx, example in enumerate(raw_examples):
        features     = sample_features.get(sample_idx, [])
        gold_answers = example.get("answers", {}).get("text", [])

        if not features:
            predictions.append("")
            references.append(gold_answers)
            continue

        context    = example.get("context", "")
        best_score = float("-inf")
        best_text  = ""

        for feat in features:
            s_log   = feat["start_logits"]
            e_log   = feat["end_logits"]
            offsets = feat["offset_mapping"]
            start_indexes = np.argsort(s_log)[-1:-n_best-1:-1].tolist()
            end_indexes   = np.argsort(e_log)[-1:-n_best-1:-1].tolist()

            for si in start_indexes:
                for ei in end_indexes:
                    if ei < si or ei - si + 1 > max_answer_length:
                        continue
                    if offsets[si] is None or offsets[ei] is None:
                        continue
                    score = s_log[si] + e_log[ei]
                    if score > best_score:
                        best_score = score
                        best_text  = context[offsets[si][0]:offsets[ei][1]].strip()

        predictions.append(best_text)
        references.append(gold_answers)

    per_em, per_f1 = [], []
    for pred, golds in zip(predictions, references):
        em, f1 = _best_scores(pred, golds)
        per_em.append(em)
        per_f1.append(f1)

    return {
        "exact_match":   100.0 * np.mean(per_em) if per_em else 0.0,
        "f1":            100.0 * np.mean(per_f1) if per_f1 else 0.0,
        "per_sample_em": per_em,
        "per_sample_f1": per_f1,
        "predictions":   predictions,
        "references":    references,
    }


# ─── evaluate_epoch (training.py gọi để track EM/F1 mỗi epoch) ───────────────

def evaluate_epoch(
    model,
    val_loader,
    loss_fn,
    tokenizer,
    raw_examples: list[dict],
    device: str,
    epoch: int,
    history: dict,
) -> dict:
    """
    Gọi sau mỗi epoch trong training.py để thu thập đầy đủ metrics.

    Args:
        history: dict tích luỹ qua các epoch, dạng:
                 {"train_loss": [], "val_loss": [], "em": [], "f1": []}
                 training.py tự append train_loss trước khi gọi hàm này,
                 hàm này sẽ append val_loss, em, f1.

    Returns:
        dict metrics của epoch hiện tại: val_loss, exact_match, f1
    """
    val_loss = evaluate_loss(model, val_loader, loss_fn, device)
    metrics  = evaluate_qa_metrics(model, val_loader, tokenizer, raw_examples, device)

    history.setdefault("val_loss", []).append(val_loss)
    history.setdefault("em",       []).append(metrics["exact_match"])
    history.setdefault("f1",       []).append(metrics["f1"])

    print(
        f"  Epoch {epoch} | val_loss={val_loss:.4f} "
        f"| EM={metrics['exact_match']:.2f}% | F1={metrics['f1']:.2f}%"
    )
    return {"val_loss": val_loss, **metrics}


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def _savefig(fig, path: Path, name: str) -> None:
    fpath = path / name
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


def plot_loss_curves(train_losses, val_losses, save_dir):
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", label="Train Loss", color="#2196F3")
    ax.plot(epochs, val_losses,   marker="s", label="Val Loss",   color="#F44336")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _savefig(fig, save_dir, "loss_curves.png")


def plot_em_f1_per_epoch(em_list, f1_list, save_dir):
    epochs = list(range(1, len(em_list) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, em_list, marker="o", label="Exact Match", color="#4CAF50")
    ax.plot(epochs, f1_list, marker="s", label="F1 Score",    color="#FF9800")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score (%)")
    ax.set_title("EM & F1 qua các Epoch")
    ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _savefig(fig, save_dir, "em_f1_per_epoch.png")


def plot_em_f1_bar(metrics, save_dir, title="Exact Match & F1 Score"):
    labels = ["Exact Match", "F1 Score"]
    values = [metrics.get("exact_match", 0), metrics.get("f1", 0)]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=["#4CAF50", "#FF9800"], width=0.4, edgecolor="black")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.2f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, 105); ax.set_ylabel("Score (%)"); ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, save_dir, "em_f1_bar.png")


def plot_f1_histogram(per_f1, save_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist([f * 100 for f in per_f1], bins=20, range=(0, 100),
            color="#9C27B0", edgecolor="white", alpha=0.85)
    ax.set_xlabel("F1 Score (%)"); ax.set_ylabel("Số lượng mẫu")
    ax.set_title("Phân phối F1 Score"); ax.grid(axis="y", alpha=0.3)
    _savefig(fig, save_dir, "f1_histogram.png")


def plot_em_pie(per_em, save_dir):
    correct = sum(per_em)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([correct, len(per_em) - correct],
           labels=["Exact Match", "Không khớp"], autopct="%1.1f%%",
           colors=["#4CAF50", "#F44336"], startangle=90,
           wedgeprops={"edgecolor": "white"})
    ax.set_title("Tỉ lệ Exact Match")
    _savefig(fig, save_dir, "em_pie.png")


def plot_f1_by_answer_length(per_f1, references, save_dir):
    def _bucket(n):
        if n <= 5:  return "0–5"
        if n <= 10: return "6–10"
        if n <= 20: return "11–20"
        if n <= 50: return "21–50"
        return "51+"

    buckets = {"0–5": [], "6–10": [], "11–20": [], "21–50": [], "51+": []}
    for f1, golds in zip(per_f1, references):
        if golds and golds[0]:
            buckets[_bucket(len(golds[0].split()))].append(f1 * 100)

    data_nonempty = [(k, v) for k, v in buckets.items() if v]
    if not data_nonempty:
        return
    labels_ne, data_ne = zip(*data_nonempty)
    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data_ne, patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#42A5F5", "#66BB6A", "#FFA726", "#AB47BC", "#EF5350"]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_xticklabels(labels_ne)
    ax.set_xlabel("Độ dài câu trả lời (words)"); ax.set_ylabel("F1 Score (%)")
    ax.set_title("F1 Score theo độ dài câu trả lời"); ax.grid(axis="y", alpha=0.3)
    _savefig(fig, save_dir, "f1_by_answer_length.png")


def plot_confidence_distribution(model, eval_loader, device, save_dir):
    model.eval()
    best_start, best_end = [], []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Conf dist", leave=False):
            s, e = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            best_start.extend(s.max(dim=-1).values.cpu().numpy())
            best_end.extend(e.max(dim=-1).values.cpu().numpy())
    model.train()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, vals, label, color in zip(axes,
            [best_start, best_end], ["Start logit", "End logit"], ["#2196F3", "#FF5722"]):
        ax.hist(vals, bins=40, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(vals), color="black", linestyle="--", label=f"Mean={np.mean(vals):.2f}")
        ax.set_xlabel(label); ax.set_ylabel("Count")
        ax.set_title(f"Phân phối {label}"); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Confidence Score Distribution", fontsize=14, fontweight="bold")
    _savefig(fig, save_dir, "confidence_distribution.png")


def plot_pred_length_vs_f1(per_f1, predictions, save_dir):
    lengths = [len(p.split()) for p in predictions]
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(lengths, [f * 100 for f in per_f1], alpha=0.35, s=15, c=per_f1, cmap="RdYlGn")
    plt.colorbar(sc, ax=ax, label="F1")
    ax.set_xlabel("Độ dài dự đoán (words)"); ax.set_ylabel("F1 Score (%)")
    ax.set_title("Độ dài dự đoán vs F1 Score"); ax.grid(alpha=0.3)
    _savefig(fig, save_dir, "pred_length_vs_f1.png")


def plot_loss_heatmap(train_losses, val_losses, save_dir):
    if len(train_losses) < 2:
        return
    data = np.array([train_losses, val_losses])
    fig, ax = plt.subplots(figsize=(max(8, len(train_losses)), 3))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Train", "Val"])
    ax.set_xticks(range(len(train_losses)))
    ax.set_xticklabels([f"E{i+1}" for i in range(len(train_losses))])
    ax.set_title("Loss Heatmap")
    plt.colorbar(im, ax=ax, label="Loss")
    for i in range(2):
        for j in range(len(train_losses)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=8)
    _savefig(fig, save_dir, "loss_heatmap.png")


def plot_compare_checkpoints(records: list[dict], save_dir: Path):
    """
    So sánh nhiều checkpoint trên cùng 1 biểu đồ.
    records: [{"label": "en", "em": 72.1, "f1": 81.3}, ...]
    """
    labels = [r["label"] for r in records]
    ems    = [r["em"]    for r in records]
    f1s    = [r["f1"]    for r in records]
    x      = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.5), 5))
    bars1 = ax.bar(x - 0.2, ems, width=0.35, label="Exact Match", color="#4CAF50", edgecolor="black")
    bars2 = ax.bar(x + 0.2, f1s, width=0.35, label="F1 Score",    color="#FF9800", edgecolor="black")

    for bar, val in list(zip(bars1, ems)) + list(zip(bars2, f1s)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 110); ax.set_ylabel("Score (%)")
    ax.set_title("So sánh các Checkpoint"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    _savefig(fig, save_dir, "compare_checkpoints.png")


def save_error_analysis(predictions, references, raw_examples, save_dir: Path, top_n: int = 50):
    errors = []
    for i, (pred, golds, example) in enumerate(zip(predictions, references, raw_examples)):
        _, f1 = _best_scores(pred, golds)
        if f1 < 1.0:
            errors.append({
                "index":      i,
                "question":   example.get("question", ""),
                "context":    example.get("context", "")[:200],
                "gold":       golds,
                "prediction": pred,
                "f1":         round(f1 * 100, 2),
            })

    errors.sort(key=lambda x: x["f1"])
    worst = errors[:top_n]

    out_path = save_dir / "error_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(worst, f, ensure_ascii=False, indent=2)
    print(f"  Saved error analysis ({len(worst)} mẫu): {out_path}")


# ─── evaluate_full (gọi sau khi train xong) ──────────────────────────────────

def evaluate_full(
    model,
    eval_loader,
    tokenizer,
    raw_examples: list[dict],
    device: str,
    save_dir: str | Path = "outputs/checkpoints_en/best_model",
    history: Optional[dict] = None,
    dataset_label: str = "Eval",
) -> dict:
    """
    Đánh giá toàn diện và vẽ tất cả biểu đồ.

    Args:
        history: dict tích luỹ từ training {"train_loss", "val_loss", "em", "f1"}
                 nếu có sẽ vẽ thêm loss curve, EM/F1 per epoch.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n  Đánh giá mô hình – {dataset_label}\n{'='*60}")

    metrics = evaluate_qa_metrics(model, eval_loader, tokenizer, raw_examples, device)
    print(f"  Exact Match : {metrics['exact_match']:.2f}%")
    print(f"  F1 Score    : {metrics['f1']:.2f}%")
    print(f"  Số mẫu      : {len(metrics['per_sample_f1'])}")

    with open(save_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset":     dataset_label,
            "num_samples": len(metrics["per_sample_f1"]),
            "exact_match": round(metrics["exact_match"], 4),
            "f1":          round(metrics["f1"], 4),
        }, f, ensure_ascii=False, indent=2)

    print("\n  Vẽ biểu đồ ...")

    if history:
        train_losses = history.get("train_loss", [])
        val_losses   = history.get("val_loss",   [])
        em_list      = history.get("em",         [])
        f1_list      = history.get("f1",         [])

        if train_losses and val_losses:
            plot_loss_curves(train_losses, val_losses, save_dir)
            plot_loss_heatmap(train_losses, val_losses, save_dir)
        if em_list and f1_list:
            plot_em_f1_per_epoch(em_list, f1_list, save_dir)

    plot_em_f1_bar(metrics, save_dir, title=f"EM & F1 – {dataset_label}")
    plot_f1_histogram(metrics["per_sample_f1"], save_dir)
    plot_em_pie(metrics["per_sample_em"], save_dir)
    plot_f1_by_answer_length(metrics["per_sample_f1"], metrics["references"], save_dir)
    plot_confidence_distribution(model, eval_loader, device, save_dir)
    plot_pred_length_vs_f1(metrics["per_sample_f1"], metrics["predictions"], save_dir)
    save_error_analysis(metrics["predictions"], metrics["references"], raw_examples, save_dir)

    print(f"\n  Biểu đồ đã lưu tại: {save_dir}\n{'='*60}\n")
    return metrics


# ─── compare_checkpoints (so sánh nhiều model đã train) ──────────────────────

def compare_checkpoints(
    checkpoint_dirs: list[str | Path],
    eval_loader,
    tokenizer,
    raw_examples: list[dict],
    device: str,
    save_dir: str | Path = "outputs/compare",
    config_yaml: str = "config/model.yaml",
) -> list[dict]:
    """
    Load từng checkpoint, chạy evaluate_qa_metrics, vẽ biểu đồ so sánh.

    Args:
        checkpoint_dirs: danh sách đường dẫn thư mục checkpoint
                         (mỗi thư mục có training_state.pt)

    Returns:
        list[dict] kết quả từng checkpoint
    """
    from loadmodel import CustomLoraDistilBertQA

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for ckpt_dir in checkpoint_dirs:
        ckpt_dir  = Path(ckpt_dir)
        ckpt_file = ckpt_dir / "training_state.pt"
        label     = ckpt_dir.name

        model = CustomLoraDistilBertQA().to(device)
        if ckpt_file.exists():
            ckpt = torch.load(ckpt_file, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded: {ckpt_file}")
        else:
            print(f"  Không tìm thấy {ckpt_file}, dùng trọng số mặc định.")

        metrics = evaluate_qa_metrics(model, eval_loader, tokenizer, raw_examples, device)
        print(f"  [{label}] EM={metrics['exact_match']:.2f}%  F1={metrics['f1']:.2f}%")

        records.append({
            "label": label,
            "em":    metrics["exact_match"],
            "f1":    metrics["f1"],
        })

        with open(save_dir / f"eval_{label}.json", "w", encoding="utf-8") as f:
            json.dump({
                "checkpoint":  str(ckpt_dir),
                "exact_match": round(metrics["exact_match"], 4),
                "f1":          round(metrics["f1"], 4),
            }, f, ensure_ascii=False, indent=2)

    plot_compare_checkpoints(records, save_dir)
    print(f"\n  So sánh checkpoint lưu tại: {save_dir}")
    return records


# ─── evaluate_on_viquad_test ─────────────────────────────────────────────────

def evaluate_on_viquad_test(
    model,
    tokenizer,
    config,
    device: str,
    save_dir: str | Path = "outputs/checkpoints_en/best_model",
) -> list[dict]:
    """
    Tải tập test UIT-ViQuAD2.0 từ HuggingFace, chạy inference, lưu dự đoán
    và vẽ biểu đồ phân tích (không tính EM/F1 vì test không có ground truth).

    Returns:
        list[dict] – predictions [{"id", "question", "context", "prediction"}]
    """
    from datasets import load_dataset as hf_load_dataset
    from torch.utils.data import DataLoader

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n  Đánh giá trên UIT-ViQuAD2.0 – split test\n{'='*60}")
    print("  Tải dataset từ HuggingFace ...")

    hf_ds       = hf_load_dataset("taidng/UIT-ViQuAD2.0", split="test")
    raw_examples = [hf_ds[i] for i in range(len(hf_ds))]
    print(f"  Số mẫu test: {len(raw_examples)}")

    # Kiểm tra ground truth có không
    has_answers = any(
        ex.get("answers", {}).get("text") for ex in raw_examples[:10]
    )

    # Tokenize
    from src.dataset import prepare_eval_features
    from src.vietnamese import normalize_text, has_vietnamese, segment_texts

    questions = [normalize_text(ex["question"]) for ex in raw_examples]
    contexts  = [normalize_text(ex["context"])  for ex in raw_examples]

    sample_dict = {
        config.question_column: questions,
        config.context_column:  contexts,
    }

    if config.use_vietnamese_segmentation and has_vietnamese(sample_dict):
        questions = segment_texts(questions)
        contexts  = segment_texts(contexts)
        sample_dict[config.question_column] = questions
        sample_dict[config.context_column]  = contexts

    from transformers import BatchEncoding
    tokenized = tokenizer(
        questions,
        contexts,
        max_length=config.max_length,
        stride=config.doc_stride,
        padding=config.padding,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    # Đánh dấu offset chỉ giữ context tokens
    clean_offsets = []
    sample_ids    = []
    for i in range(len(tokenized["input_ids"])):
        seq_ids = tokenized.sequence_ids(i)
        sample_ids.append(sample_mapping[i])
        clean_offsets.append([
            (o if seq_ids[k] == 1 else None)
            for k, o in enumerate(offset_mapping[i])
        ])

    import torch
    from torch.utils.data import TensorDataset

    input_ids_t      = torch.tensor(tokenized["input_ids"],      dtype=torch.long)
    attention_mask_t = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
    test_dataset     = TensorDataset(input_ids_t, attention_mask_t)
    test_loader      = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Inference
    model.eval()
    all_start_logits, all_end_logits = [], []
    with torch.no_grad():
        for input_ids_b, attention_mask_b in tqdm(test_loader, desc="ViQuAD Test Inference"):
            s, e = model(input_ids_b.to(device), attention_mask_b.to(device))
            all_start_logits.extend(s.cpu().numpy())
            all_end_logits.extend(e.cpu().numpy())
    model.train()

    # Group features by sample
    n_best = 20
    max_answer_length = 30
    feat_by_sample: dict[int, list] = collections.defaultdict(list)
    for sid, s_log, e_log, off in zip(sample_ids, all_start_logits, all_end_logits, clean_offsets):
        feat_by_sample[sid].append({"s": s_log, "e": e_log, "off": off})

    predictions = []
    for idx, example in enumerate(raw_examples):
        context    = example.get("context", "")
        best_score = float("-inf")
        best_text  = ""

        for feat in feat_by_sample.get(idx, []):
            s_log, e_log, offsets = feat["s"], feat["e"], feat["off"]
            for si in np.argsort(s_log)[-1:-n_best-1:-1].tolist():
                for ei in np.argsort(e_log)[-1:-n_best-1:-1].tolist():
                    if ei < si or ei - si + 1 > max_answer_length:
                        continue
                    if offsets[si] is None or offsets[ei] is None:
                        continue
                    score = s_log[si] + e_log[ei]
                    if score > best_score:
                        best_score = score
                        best_text  = context[offsets[si][0]:offsets[ei][1]].strip()

        predictions.append({
            "id":         example.get("id", idx),
            "question":   example.get("question", ""),
            "context":    context[:200],
            "prediction": best_text,
        })

    # Lưu predictions
    pred_path = save_dir / "viquad_test_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(predictions)} predictions: {pred_path}")

    # Tính EM/F1 nếu có ground truth
    if has_answers:
        per_em, per_f1 = [], []
        refs = []
        for pred, example in zip(predictions, raw_examples):
            golds = example.get("answers", {}).get("text", [])
            em, f1 = _best_scores(pred["prediction"], golds)
            per_em.append(em)
            per_f1.append(f1)
            refs.append(golds)

        em_score = 100.0 * np.mean(per_em)
        f1_score = 100.0 * np.mean(per_f1)
        print(f"  Exact Match : {em_score:.2f}%")
        print(f"  F1 Score    : {f1_score:.2f}%")

        with open(save_dir / "viquad_test_results.json", "w", encoding="utf-8") as f:
            json.dump({
                "dataset":     "UIT-ViQuAD2.0 test",
                "num_samples": len(per_f1),
                "exact_match": round(em_score, 4),
                "f1":          round(f1_score, 4),
            }, f, ensure_ascii=False, indent=2)

        plot_em_f1_bar(
            {"exact_match": em_score, "f1": f1_score},
            save_dir,
            title="EM & F1 – UIT-ViQuAD2.0 Test",
        )
        plot_f1_histogram(per_f1, save_dir)
        plot_em_pie(per_em, save_dir)
        plot_f1_by_answer_length(per_f1, refs, save_dir)
        save_error_analysis(
            [p["prediction"] for p in predictions], refs, raw_examples, save_dir
        )

    # Biểu đồ luôn vẽ dù có hay không có ground truth
    _plot_viquad_pred_length([p["prediction"] for p in predictions], save_dir)
    _plot_viquad_confidence(all_start_logits, all_end_logits, save_dir)

    print(f"\n  Kết quả lưu tại: {save_dir}\n{'='*60}\n")
    return predictions


def _plot_viquad_pred_length(predictions: list[str], save_dir: Path):
    lengths = [len(p.split()) for p in predictions]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=30, color="#00ACC1", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(lengths), color="black", linestyle="--",
               label=f"Mean={np.mean(lengths):.1f} words")
    ax.set_xlabel("Độ dài dự đoán (words)")
    ax.set_ylabel("Số lượng mẫu")
    ax.set_title("Phân phối độ dài câu trả lời dự đoán – ViQuAD Test")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _savefig(fig, save_dir, "viquad_pred_length.png")


def _plot_viquad_confidence(all_start_logits, all_end_logits, save_dir: Path):
    best_scores = [
        float(np.max(s) + np.max(e))
        for s, e in zip(all_start_logits, all_end_logits)
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(best_scores, bins=40, color="#7E57C2", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(best_scores), color="black", linestyle="--",
               label=f"Mean={np.mean(best_scores):.2f}")
    ax.set_xlabel("Best span score (start + end logit)")
    ax.set_ylabel("Số lượng mẫu")
    ax.set_title("Phân phối Confidence Score – ViQuAD Test")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _savefig(fig, save_dir, "viquad_confidence.png")


# ─── Standalone ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from datasets import load_dataset as hf_load_dataset
    from loadmodel import CustomLoraDistilBertQA
    from config_model import Config
    from src.data_loader import build_qa_datasets

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="outputs/checkpoints_en/best_model")
    parser.add_argument("--profile",     default="eval_en")
    parser.add_argument("--save_dir",    default="outputs/checkpoints_en/best_model")
    parser.add_argument("--config_yaml", default="config/model.yaml")
    parser.add_argument("--compare",     nargs="*", default=None,
                        help="Danh sách thư mục checkpoint để so sánh")
    parser.add_argument("--viquad_test", action="store_true",
                        help="Chạy inference trên tập test UIT-ViQuAD2.0")
    args = parser.parse_args()

    config    = Config.from_yaml(path=args.config_yaml, profile=args.profile)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint if Path(args.checkpoint).exists() else config.model_name
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model     = CustomLoraDistilBertQA().to(device)
    ckpt_file = Path(args.checkpoint) / "training_state.pt"
    history   = None
    if ckpt_file.exists():
        ckpt = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded: {ckpt_file}")
        history = ckpt.get("history", None)

    if args.viquad_test:
        evaluate_on_viquad_test(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            save_dir=args.save_dir,
        )

    elif args.compare:
        tokenized_datasets = build_qa_datasets(tokenizer, config, is_training=False)
        eval_split = tokenized_datasets.get("validation", tokenized_datasets.get("test"))
        eval_loader = DataLoader(eval_split, batch_size=config.batch_size, shuffle=False, num_workers=0)

        val_file = config.validation_file or config.test_file
        raw_examples = (
            [json.loads(l) for l in Path(val_file).read_text(encoding="utf-8").splitlines() if l.strip()]
            if val_file and Path(val_file).exists()
            else [hf_load_dataset("taidng/UIT-ViQuAD2.0", split="test")[i]
                  for i in range(len(hf_load_dataset("taidng/UIT-ViQuAD2.0", split="test")))]
        )
        compare_checkpoints(
            checkpoint_dirs=args.compare,
            eval_loader=eval_loader,
            tokenizer=tokenizer,
            raw_examples=raw_examples,
            device=device,
            save_dir=args.save_dir,
        )

    else:
        tokenized_datasets = build_qa_datasets(tokenizer, config, is_training=False)
        eval_split = tokenized_datasets.get("validation", tokenized_datasets.get("test"))
        if eval_split is None:
            raise RuntimeError("Không có split 'validation' hoặc 'test'.")
        eval_loader = DataLoader(eval_split, batch_size=config.batch_size, shuffle=False, num_workers=0)

        val_file = config.validation_file or config.test_file
        if val_file and Path(val_file).exists():
            raw_examples = [
                json.loads(line)
                for line in Path(val_file).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            print("Tải từ HuggingFace (taidng/UIT-ViQuAD2.0) ...")
            hf_ds = hf_load_dataset("taidng/UIT-ViQuAD2.0", split="test")
            raw_examples = [hf_ds[i] for i in range(len(hf_ds))]

        evaluate_full(
            model=model,
            eval_loader=eval_loader,
            tokenizer=tokenizer,
            raw_examples=raw_examples,
            device=device,
            save_dir=args.save_dir,
            history=history,
            dataset_label="UIT-ViQuAD2.0 Test" if "vi" in args.profile else "SQuAD Eval",
        )