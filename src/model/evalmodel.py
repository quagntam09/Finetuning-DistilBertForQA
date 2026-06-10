from __future__ import annotations

import collections
import re
import string
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import tqdm

try:
    import evaluate as hf_evaluate
except ImportError:  # pragma: no cover - surfaced when metrics are used
    hf_evaluate = None


# ─── Metrics helpers ──────────────────────────────────────────────────────────

_SQUAD_V2_METRIC = None


def _load_squad_v2_metric():
    global _SQUAD_V2_METRIC
    if _SQUAD_V2_METRIC is None:
        if hf_evaluate is None:
            raise RuntimeError(
                "Missing dependency: evaluate. Install it with `pip install evaluate`."
            )
        _SQUAD_V2_METRIC = hf_evaluate.load("squad_v2")
    return _SQUAD_V2_METRIC


def _squad_v2_references(references: list[list[str]]) -> list[dict]:
    return [
        {
            "id": str(idx),
            "answers": {
                "text": list(golds),
                "answer_start": [0] * len(golds),
            },
        }
        for idx, golds in enumerate(references)
    ]


def _squad_v2_predictions(predictions: list[str]) -> list[dict]:
    return [
        {
            "id": str(idx),
            "prediction_text": pred or "",
            "no_answer_probability": 1.0 if not (pred or "").strip() else 0.0,
        }
        for idx, pred in enumerate(predictions)
    ]


def _compute_squad_v2_metric(predictions: list[str], references: list[list[str]]) -> dict:
    if not predictions:
        return {
            "exact_match": 0.0,
            "f1": 0.0,
            "has_answer_exact": 0.0,
            "has_answer_f1": 0.0,
            "has_answer_total": 0,
            "no_answer_exact": 0.0,
            "no_answer_f1": 0.0,
            "no_answer_total": 0,
        }

    result = _load_squad_v2_metric().compute(
        predictions=_squad_v2_predictions(predictions),
        references=_squad_v2_references(references),
        no_answer_threshold=0.5,
    )
    return {
        "exact_match": float(result.get("exact", 0.0)),
        "f1": float(result.get("f1", 0.0)),
        "has_answer_exact": float(result.get("HasAns_exact", 0.0)),
        "has_answer_f1": float(result.get("HasAns_f1", 0.0)),
        "has_answer_total": int(result.get("HasAns_total", 0)),
        "no_answer_exact": float(result.get("NoAns_exact", 0.0)),
        "no_answer_f1": float(result.get("NoAns_f1", 0.0)),
        "no_answer_total": int(result.get("NoAns_total", 0)),
    }


def _normalize_answer(s: str) -> str:
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def _get_tokens(s: str) -> list[str]:
    return _normalize_answer(s).split() if s else []


def compute_exact(prediction: str, ground_truth: str) -> int:
    return int(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def compute_precision_recall_f1(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    pred_tokens  = _get_tokens(prediction)
    truth_tokens = _get_tokens(ground_truth)
    common   = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0 or not pred_tokens or not truth_tokens:
        return 0.0, 0.0, 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _best_metric_scores(prediction: str, gold_answers: list[str]) -> tuple[int, float, float, float]:
    if not gold_answers:
        is_empty = int(_normalize_answer(prediction) == "")
        score = float(is_empty)
        return is_empty, score, score, score

    candidates = []
    for answer in gold_answers:
        precision, recall, f1 = compute_precision_recall_f1(prediction, answer)
        candidates.append((compute_exact(prediction, answer), precision, recall, f1))
    return max(candidates, key=lambda item: (item[3], item[0], item[2], item[1]))


# ─── evaluate_loss (training.py gọi mỗi epoch) ───────────────────────────────


def qa_eval_collate(features: list[dict]) -> dict:
    """Collate QA eval batches while preserving offset_mapping None values."""

    def _normalize_offset(offset):
        if offset is None:
            return None
        if hasattr(offset, "tolist"):
            offset = offset.tolist()
        if isinstance(offset, (list, tuple)) and len(offset) == 2:
            return (int(offset[0]), int(offset[1]))
        return offset

    batch = {}
    tensor_keys = [
        "input_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
    ]
    for key in tensor_keys:
        if key in features[0]:
            batch[key] = default_collate([feature[key] for feature in features])

    if "sample_id" in features[0]:
        sample_ids = []
        for feature in features:
            value = feature["sample_id"]
            if hasattr(value, "item"):
                value = value.item()
            sample_ids.append(int(value))
        batch["sample_id"] = sample_ids

    if "offset_mapping" in features[0]:
        offsets = []
        for feature in features:
            value = feature["offset_mapping"]
            if hasattr(value, "tolist"):
                value = value.tolist()
            offsets.append([_normalize_offset(offset) for offset in value])
        batch["offset_mapping"] = offsets

    return batch


def evaluate_loss(model, val_loader, loss_fn, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation loss", leave=False):
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
    no_answer_threshold: float = 0.0,
    tune_no_answer_threshold: bool = False,
) -> dict:
    model.eval()
    all_start_logits, all_end_logits = [], []
    all_sample_ids, all_offset_maps  = [], []
    all_input_ids = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Inference", leave=False):
            input_ids = batch["input_ids"].to(device)
            s_logits, e_logits = model(
                input_ids,
                batch["attention_mask"].to(device),
            )
            all_start_logits.extend(s_logits.cpu().numpy())
            all_end_logits.extend(e_logits.cpu().numpy())
            all_input_ids.extend(input_ids.cpu().numpy())

            sids = batch.get("sample_id", [])
            offs = batch.get("offset_mapping", [])
            if hasattr(sids, "tolist"): sids = sids.tolist()
            if hasattr(offs, "tolist"): offs = offs.tolist()
            all_sample_ids.extend(sids)
            all_offset_maps.extend(offs)

    model.train()

    sample_features: dict[int, list[dict]] = collections.defaultdict(list)
    for sid, input_ids, s_log, e_log, off in zip(
        all_sample_ids,
        all_input_ids,
        all_start_logits,
        all_end_logits,
        all_offset_maps,
    ):
        sample_features[sid].append({
            "input_ids": input_ids,
            "start_logits": s_log,
            "end_logits": e_log,
            "offset_mapping": off,
        })

    span_predictions, references, no_answer_score_diffs = [], [], []
    for sample_idx, example in enumerate(
        tqdm(raw_examples, desc="Post-process spans", leave=False)
    ):
        features     = sample_features.get(sample_idx, [])
        gold_answers = example.get("answers", {}).get("text", [])

        if not features:
            span_predictions.append("")
            references.append(gold_answers)
            no_answer_score_diffs.append(float("-inf"))
            continue

        context    = example.get("context", "")
        best_span_score = float("-inf")
        best_span_text  = ""
        best_null_score = float("-inf")

        for feat in features:
            input_ids = feat["input_ids"]
            s_log   = feat["start_logits"]
            e_log   = feat["end_logits"]
            offsets = feat["offset_mapping"]

            cls_matches = np.where(input_ids == tokenizer.cls_token_id)[0]
            cls_index = int(cls_matches[0]) if len(cls_matches) else 0
            null_score = s_log[cls_index] + e_log[cls_index]
            if null_score > best_null_score:
                best_null_score = null_score

            start_indexes = np.argsort(s_log)[-1:-n_best-1:-1].tolist()
            end_indexes   = np.argsort(e_log)[-1:-n_best-1:-1].tolist()

            for si in start_indexes:
                for ei in end_indexes:
                    if ei < si or ei - si + 1 > max_answer_length:
                        continue
                    if offsets[si] is None or offsets[ei] is None:
                        continue
                    score = s_log[si] + e_log[ei]
                    if score > best_span_score:
                        best_span_score = score
                        best_span_text  = context[offsets[si][0]:offsets[ei][1]].strip()

        span_predictions.append(best_span_text)
        references.append(gold_answers)
        no_answer_score_diffs.append(float(best_span_score - best_null_score))

    def _apply_threshold(threshold: float) -> list[str]:
        return [
            "" if score_diff < threshold else pred
            for pred, score_diff in zip(span_predictions, no_answer_score_diffs)
        ]

    def _score_predictions(predictions: list[str], use_squad_backend: bool = True) -> dict:
        per_em, per_precision, per_recall, per_f1 = [], [], [], []
        for pred, golds in zip(predictions, references):
            em, precision, recall, f1 = _best_metric_scores(pred, golds)
            per_em.append(em)
            per_precision.append(precision)
            per_recall.append(recall)
            per_f1.append(f1)

        custom_exact_match = 100.0 * np.mean(per_em) if per_em else 0.0
        custom_f1 = 100.0 * np.mean(per_f1) if per_f1 else 0.0
        if use_squad_backend:
            squad_v2_scores = _compute_squad_v2_metric(predictions, references)
        else:
            squad_v2_scores = {
                "exact_match": custom_exact_match,
                "f1": custom_f1,
                "has_answer_exact": 0.0,
                "has_answer_f1": 0.0,
                "has_answer_total": 0,
                "no_answer_exact": 0.0,
                "no_answer_f1": 0.0,
                "no_answer_total": 0,
            }
        return {
            "exact_match":          squad_v2_scores["exact_match"],
            "accuracy":             squad_v2_scores["exact_match"],
            "precision":            100.0 * np.mean(per_precision) if per_precision else 0.0,
            "recall":               100.0 * np.mean(per_recall) if per_recall else 0.0,
            "f1":                   squad_v2_scores["f1"],
            "has_answer_exact":     squad_v2_scores["has_answer_exact"],
            "has_answer_f1":        squad_v2_scores["has_answer_f1"],
            "has_answer_total":     squad_v2_scores["has_answer_total"],
            "no_answer_exact":      squad_v2_scores["no_answer_exact"],
            "no_answer_f1":         squad_v2_scores["no_answer_f1"],
            "no_answer_total":      squad_v2_scores["no_answer_total"],
            "metric_backend":       "evaluate.squad_v2",
            "per_sample_em":        per_em,
            "per_sample_precision": per_precision,
            "per_sample_recall":    per_recall,
            "per_sample_f1":        per_f1,
            "predictions":          predictions,
        }

    selected_threshold = float(no_answer_threshold)
    if tune_no_answer_threshold and no_answer_score_diffs:
        finite_diffs = [d for d in no_answer_score_diffs if np.isfinite(d)]
        if finite_diffs:
            eps = 1e-6
            candidates = [min(finite_diffs) - eps, 0.0]
            candidates.extend(d + eps for d in sorted(set(finite_diffs)))
            best_threshold = selected_threshold
            best_scores = _score_predictions(
                _apply_threshold(best_threshold),
                use_squad_backend=False,
            )
            for threshold in tqdm(candidates, desc="Tune no-answer threshold", leave=False):
                scores = _score_predictions(
                    _apply_threshold(threshold),
                    use_squad_backend=False,
                )
                if (scores["f1"], scores["exact_match"]) > (best_scores["f1"], best_scores["exact_match"]):
                    best_threshold = float(threshold)
                    best_scores = scores
            selected_threshold = best_threshold

    predictions = _apply_threshold(selected_threshold)
    scores = _score_predictions(predictions)
    scores.update({
        "no_answer_threshold": selected_threshold,
        "tuned_no_answer_threshold": bool(tune_no_answer_threshold),
        "per_sample_no_answer_score_diff": no_answer_score_diffs,
        "span_predictions": span_predictions,
        "references": references,
    })
    return scores


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


def plot_accuracy_f1_recall_per_epoch(accuracy_list, f1_list, recall_list, save_dir, precision_list=None):
    n_epochs = min(len(f1_list), len(recall_list))
    if n_epochs == 0:
        return

    epochs = list(range(1, n_epochs + 1))
    f1_list = f1_list[:n_epochs]
    recall_list = recall_list[:n_epochs]
    fig, ax = plt.subplots(figsize=(8, 5))
    if accuracy_list and len(accuracy_list) >= n_epochs:
        ax.plot(epochs, accuracy_list[:n_epochs], marker="o", label="Accuracy/EM", color="#4CAF50")
    if precision_list and len(precision_list) >= n_epochs:
        ax.plot(epochs, precision_list[:n_epochs], marker="^", label="Precision", color="#2196F3")
    ax.plot(epochs, recall_list[:len(epochs)], marker="d", label="Recall", color="#7E57C2")
    ax.plot(epochs, f1_list, marker="s", label="F1 Score", color="#FF9800")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Accuracy, Recall & F1 qua các Epoch")
    ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _savefig(fig, save_dir, "accuracy_f1_recall_per_epoch.png")


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
