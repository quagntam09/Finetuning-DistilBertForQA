"""
Data loading utilities for QA datasets.

Chỉ hỗ trợ tải từ local JSONL files trong folder data/.

Hỗ trợ preprocessing với prepare_train_features/prepare_eval_features.
"""

from __future__ import annotations

from pathlib import Path
import logging

from datasets import DatasetDict, load_dataset

from .dataset import prepare_train_features, prepare_eval_features


logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_raw_datasets(config) -> DatasetDict:
    """
    Tải dataset QA từ local JSONL files trong folder data/.

    Dựa vào config.train_file, config.validation_file, config.test_file
    (đường dẫn tương đối tới thư mục data/ hoặc tuyệt đối).

    Args:
        config: TrainingConfig object với các tham số dataset

    Returns:
        DatasetDict với splits "train", "validation", "test" (tùy khả dụng)
    """

    def _resolve(path: str | None) -> str | None:
        if path is None:
            return None
        p = Path(path)
        if not p.is_absolute():
            p = DATA_DIR / p
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p}")
        return str(p)

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

    logger.info(f"Loading local dataset từ files: {list(data_files.keys())}")

    datasets = load_dataset(path="json", data_files=data_files, cache_dir=config.cache_dir)

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

        # Chọn hàm xử lý tùy theo split
        has_answers = config.answers_column in dataset.column_names
        has_context_labels = split_name in {"train", "validation"} and has_answers

        if has_context_labels:
            prepare_fn = prepare_train_features
            prepare_kwargs = {
                "answers_column": config.answers_column,
                "impossible_column": config.impossible_column,
            }
        else:
            prepare_fn = prepare_eval_features
            prepare_kwargs = {}

        # Tokenize (auto-detects Vietnamese via language column)
        processed_dataset = dataset.map(
            lambda examples: prepare_fn(
                examples=examples,
                tokenizer=tokenizer,
                question_column=config.question_column,
                context_column=config.context_column,
                max_length=config.max_length,
                doc_stride=config.doc_stride,
                padding=config.padding,
                **prepare_kwargs,
            ),
            batched=True,
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