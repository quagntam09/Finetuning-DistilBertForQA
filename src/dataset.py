from __future__ import annotations

import logging

from transformers import AutoTokenizer, PreTrainedTokenizerBase

try:
    from .vietnamese import (
        has_vietnamese,
        is_quality_sample,
        normalize_text,
        segment_texts,
        validate_answer_in_segmented,
    )
except ImportError:
    from vietnamese import (
        has_vietnamese,
        is_quality_sample,
        normalize_text,
        segment_texts,
        validate_answer_in_segmented,
    )

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Quality filtering (call BEFORE tokenization)
# ──────────────────────────────────────────────

def filter_qa_dataset(
    dataset,
    question_column: str,
    context_column: str,
    answers_column: str,
) -> "Dataset":
    """Remove low-quality QA samples from a HuggingFace ``Dataset``.

    Applies :func:`is_quality_sample` per-row and filters out rows that fail.
    Returns a new ``Dataset`` with the same schema (fewer rows).
    """

    def _filter_row(row) -> bool:
        q = normalize_text(row[question_column])
        c = normalize_text(row[context_column])
        ans = row.get(answers_column) if isinstance(row, dict) else row[answers_column]
        return is_quality_sample(c, q, ans)

    n_before = len(dataset)
    filtered = dataset.filter(_filter_row, batched=False)
    n_after = len(filtered)
    removed = n_before - n_after
    if removed:
        logger.info(
            "Quality filter removed %d / %d samples (%.1f%%)",
            removed, n_before, 100 * removed / n_before,
        )
    return filtered


def _ensure_pad_token(tokenizer: PreTrainedTokenizerBase, padding: str) -> None:
    if padding == "do_not_pad":
        return
    if tokenizer.pad_token is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return
    raise RuntimeError(
        "Tokenizer has no pad_token and no eos_token to reuse. "
        "Set pad_token explicitly at tokenizer construction time "
        "(e.g. tokenizer.pad_token = tokenizer.eos_token or "
        "tokenizer.add_special_tokens({'pad_token': '[PAD]'})) "
        "before the model is instantiated, or call "
        "model.resize_token_embeddings() after adding new tokens."
    )


def prepare_train_features(
    examples: dict[str, list],
    tokenizer: PreTrainedTokenizerBase,
    question_column: str,
    context_column: str,
    max_length: int,
    doc_stride: int,
    padding: str,
    answers_column: str,
    impossible_column: str,
    use_vietnamese_segmentation: bool = True,
    segmentation_tool: str | None = "underthesea",
) -> dict[str, list]:
    questions = [normalize_text(q) for q in examples[question_column]]
    contexts = [normalize_text(c) for c in examples[context_column]]
    answers = examples[answers_column]
    is_impossible = examples[impossible_column]

    if use_vietnamese_segmentation and has_vietnamese(examples):
        logger.info("Detected Vietnamese data, applying word segmentation")
        if segmentation_tool and segmentation_tool != "underthesea":
            logger.warning("Unsupported segmentation_tool=%s, using underthesea", segmentation_tool)
        contexts_orig = list(contexts)
        questions = segment_texts(questions)
        contexts = segment_texts(contexts)
        validate_answer_in_segmented(contexts_orig, contexts, answers)

    _ensure_pad_token(tokenizer, padding)

    tokenized = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        stride=doc_stride,
        padding=padding,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sample_idx = sample_mapping[i]
        answer = answers[sample_idx]
        impossible = is_impossible[sample_idx]

        if impossible or not answer["text"]:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            answer_start_char = answer["answer_start"][0]
            answer_end_char = answer_start_char + len(answer["text"][0])

            sequence_ids = tokenized.sequence_ids(i)

            context_start = 0
            while sequence_ids[context_start] != 1:
                context_start += 1
            context_end = len(sequence_ids) - 1
            while sequence_ids[context_end] != 1:
                context_end -= 1

            if offsets[context_start][0] > answer_end_char or offsets[context_end][1] < answer_start_char:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                token_start = context_start
                while token_start <= context_end and offsets[token_start][0] <= answer_start_char:
                    token_start += 1
                token_start -= 1

                token_end = context_start
                while token_end <= context_end and offsets[token_end][1] <= answer_end_char:
                    token_end += 1
                token_end -= 1

                start_positions.append(token_start)
                end_positions.append(token_end)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    return tokenized


def prepare_eval_features(
    examples: dict[str, list],
    tokenizer: PreTrainedTokenizerBase,
    question_column: str,
    context_column: str,
    max_length: int,
    doc_stride: int,
    padding: str,
    use_vietnamese_segmentation: bool = True,
    segmentation_tool: str | None = "underthesea",
) -> dict[str, list]:
    questions = [normalize_text(q) for q in examples[question_column]]
    contexts = [normalize_text(c) for c in examples[context_column]]

    if use_vietnamese_segmentation and has_vietnamese(examples):
        logger.info("Detected Vietnamese data, applying word segmentation")
        if segmentation_tool and segmentation_tool != "underthesea":
            logger.warning("Unsupported segmentation_tool=%s, using underthesea", segmentation_tool)
        questions = segment_texts(questions)
        contexts = segment_texts(contexts)

    _ensure_pad_token(tokenizer, padding)

    tokenized = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        stride=doc_stride,
        padding=padding,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    tokenized["sample_id"] = []
    for i in range(len(tokenized["input_ids"])):
        sample_idx = sample_mapping[i]
        tokenized["sample_id"].append(sample_idx)

        sequence_ids = tokenized.sequence_ids(i)
        context_start = 0
        while sequence_ids[context_start] != 1:
            context_start += 1
        context_end = len(sequence_ids) - 1
        while sequence_ids[context_end] != 1:
            context_end -= 1

        offset_mapping[i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(offset_mapping[i])
        ]

    tokenized["offset_mapping"] = offset_mapping

    return tokenized
