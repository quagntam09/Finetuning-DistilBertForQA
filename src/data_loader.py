"""
Data loading utilities for QA datasets.

Loads pre-filtered JSONL files from data/filtered_* (falls back to raw data).
"""

from __future__ import annotations

from pathlib import Path
import logging

from datasets import DatasetDict, load_dataset

from .dataset import prepare_train_features, prepare_eval_features
from .vietnamese import has_vietnamese, normalize_text, segment_texts, validate_answer_in_segmented


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

_FILTERED_DIRS: dict[str, Path] = {
    "data_en": DATA_DIR / "filtered_en",
    "data_vi": DATA_DIR / "filtered_vi",
}


def _resolve_filtered(raw_path: str) -> str | None:
    """Map raw path → filtered path using language-specific subfolder.

    data/data_en/train.jsonl  → data/filtered_en/data_en_train_filtered.jsonl
    data/data_vi/validation.jsonl → data/filtered_vi/data_vi_validation_filtered.jsonl
    """
    p = Path(raw_path)
    try:
        rel = p.relative_to(DATA_DIR)
    except ValueError:
        return None
    parts = rel.parts
    if not parts:
        return None
    base_dir = _FILTERED_DIRS.get(parts[0])
    if base_dir is None:
        return None
    name = rel.with_suffix("").as_posix().replace("/", "_") + "_filtered"
    candidate = base_dir / f"{name}.jsonl"
    if candidate.exists():
        return str(candidate)
    return None


def load_raw_datasets(config) -> DatasetDict:
    """
    Tải dataset QA từ local JSONL files.
    Ưu tiên bản filtered trong data/filtered_*/, fallback về raw data/.
    """

    def _resolve(path: str | None) -> str | None:
        if path is None:
            return None
        p = Path(path).expanduser()
        candidates = [p] if p.is_absolute() else [PROJECT_ROOT / p, DATA_DIR / p]

        # Prefer filtered version
        for candidate in candidates:
            filtered = _resolve_filtered(str(candidate))
            if filtered:
                return filtered

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        checked = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Dataset file not found: {path}. Checked: {checked}")

    data_files: dict[str, str] = {}

    train = _resolve(config.train_file)
    if train:
        data_files["train"] = train

    validation = _resolve(config.validation_file)
    if validation:
        data_files["validation"] = validation

    test = _resolve(config.test_file)
    if test:
        data_files["test"] = test

    if not data_files:
        raise ValueError(
            "Cần cung cấp ít nhất một trong: train_file, validation_file, test_file"
        )

    for split, fp in data_files.items():
        label = "(filtered)" if "filtered" in fp else "(raw)"
        print(f"  Load {split}: {fp} {label}")

    logger.info(f"Loading local dataset từ files: {list(data_files.keys())}")

    datasets = load_dataset(path="json", data_files=data_files, cache_dir=config.cache_dir)
    if getattr(config, "answer_only", False):
        answers_column = getattr(config, "answers_column", "answers")
        impossible_column = getattr(config, "impossible_column", "is_impossible")

        def _has_answer(row):
            answers = row.get(answers_column) or {}
            texts = answers.get("text") or []
            starts = answers.get("answer_start") or []
            return (
                not row.get(impossible_column, False)
                and any(str(text).strip() for text in texts)
                and bool(starts)
            )

        before_counts = {split: len(dataset) for split, dataset in datasets.items()}
        datasets = datasets.filter(_has_answer, desc="Filtering answer-only QA samples")
        after_counts = {split: len(dataset) for split, dataset in datasets.items()}
        for split in datasets:
            print(
                f"  Answer-only {split}: "
                f"{before_counts[split]:,} -> {after_counts[split]:,}"
            )

    logger.info(f"Loaded splits: {list(datasets.keys())}")
    return datasets


def build_qa_datasets(tokenizer, config, is_training: bool = True) -> DatasetDict:
    """
    Tải và tokenize QA datasets.

    Args:
        tokenizer: HuggingFace tokenizer (phải hỗ trợ offset_mapping)
        config: TrainingConfig object
        is_training: True nếu dùng train/eval features, False nếu chỉ cần tokens

    Returns:
        DatasetDict với các splits đã được tokenized:
        - input_ids: Tokenized sequence IDs
        - attention_mask: Attention mask
        - start_positions & end_positions (nếu is_training=True)
    """

    raw_datasets = load_raw_datasets(config=config)

    processed = DatasetDict()

    for split_name, dataset in raw_datasets.items():
        logger.info(f"Processing split '{split_name}' ({len(dataset)} samples)")

        has_answers = config.answers_column in dataset.column_names
        has_context_labels = (
            is_training
            and split_name in {"train", "validation"}
            and has_answers
        )

        if has_context_labels:
            prepare_fn = prepare_train_features
            prepare_kwargs = {
                "answers_column": config.answers_column,
                "impossible_column": config.impossible_column,
            }
        else:
            prepare_fn = prepare_eval_features
            prepare_kwargs = {}

        # Tokenize
        if has_context_labels:
            processed_dataset = dataset.map(
                lambda examples: prepare_fn(
                    examples=examples,
                    tokenizer=tokenizer,
                    question_column=config.question_column,
                    context_column=config.context_column,
                    max_length=config.max_length,
                    doc_stride=config.doc_stride,
                    padding=config.padding,
                    use_vietnamese_segmentation=config.use_vietnamese_segmentation,
                    segmentation_tool=config.segmentation_tool,
                    **prepare_kwargs,
                ),
                batched=True,
                remove_columns=dataset.column_names,
                desc=f"Tokenizing {split_name}",
            )
        else:
            processed_dataset = dataset.map(
                lambda examples, indices: prepare_fn(
                    examples=examples,
                    tokenizer=tokenizer,
                    question_column=config.question_column,
                    context_column=config.context_column,
                    max_length=config.max_length,
                    doc_stride=config.doc_stride,
                    padding=config.padding,
                    use_vietnamese_segmentation=config.use_vietnamese_segmentation,
                    segmentation_tool=config.segmentation_tool,
                    example_indices=indices,
                    **prepare_kwargs,
                ),
                batched=True,
                with_indices=True,
                remove_columns=dataset.column_names,
                desc=f"Tokenizing {split_name}",
            )

        # Set PyTorch format
        if has_context_labels:
            # Training/validation loss: cần start/end positions
            processed_dataset.set_format(
                type="torch",
                columns=[
                    "input_ids",
                    "attention_mask",
                    "start_positions",
                    "end_positions",
                ],
            )
        else:
            # Evaluation: giữ offset mapping và sample_id để post-process predictions
            eval_columns = ["input_ids", "attention_mask"]
            if "offset_mapping" in processed_dataset.column_names:
                eval_columns.append("offset_mapping")
            if "sample_id" in processed_dataset.column_names:
                eval_columns.append("sample_id")
            processed_dataset.set_format(
                type="torch",
                columns=eval_columns,
            )

        processed[split_name] = processed_dataset
        logger.info(f"  → {len(processed_dataset)} features after tokenization")

    return processed


def prepare_metric_raw_examples(rows: list[dict], config) -> list[dict]:
    """Mirror eval preprocessing for raw examples used by QA metric post-processing."""
    examples = [dict(row) for row in rows]
    if not examples:
        return examples

    questions = [normalize_text(row.get(config.question_column, "")) for row in examples]
    contexts = [normalize_text(row.get(config.context_column, "")) for row in examples]
    answers = [
        {
            "text": list((row.get(config.answers_column) or {}).get("text") or []),
            "answer_start": list((row.get(config.answers_column) or {}).get("answer_start") or []),
        }
        for row in examples
    ]

    batch = {
        config.question_column: questions,
        config.context_column: contexts,
        "language": [row.get("language") for row in examples],
    }
    if getattr(config, "use_vietnamese_segmentation", True) and has_vietnamese(batch):
        contexts_orig = list(contexts)
        questions = segment_texts(questions)
        contexts = segment_texts(contexts)
        validate_answer_in_segmented(contexts_orig, contexts, answers)

    for row, question, context, answer in zip(examples, questions, contexts, answers):
        row[config.question_column] = question
        row[config.context_column] = context
        if config.answers_column in row:
            row[config.answers_column] = answer

    return examples


def load_dataset_for_inference(
    context: str,
    question: str,
    tokenizer,
    config,
) -> dict:
    """
    Chuẩn bị single sample cho inference (không yêu cầu answers).

    Args:
        context: Context text
        question: Question text
        tokenizer: HuggingFace tokenizer
        config: Config object (max_length, doc_stride, etc.)

    Returns:
        Dict với input_ids, attention_mask, offset_mapping, etc. cho inference
    """
    from .dataset import prepare_eval_features

    examples = {
        config.question_column: [question],
        config.context_column: [context],
    }

    features = prepare_eval_features(
        examples=examples,
        tokenizer=tokenizer,
        question_column=config.question_column,
        context_column=config.context_column,
        max_length=config.max_length,
        doc_stride=config.doc_stride,
        padding=config.padding,
        use_vietnamese_segmentation=config.use_vietnamese_segmentation,
        segmentation_tool=config.segmentation_tool,
    )

    # Convert to tensors
    import torch

    result = {}
    for key in ["input_ids", "attention_mask"]:
        if key in features:
            result[key] = torch.tensor([features[key][0]], dtype=torch.long)

    # Keep offset_mapping for post-processing
    if "offset_mapping" in features:
        result["offset_mapping"] = features["offset_mapping"]

    return result
