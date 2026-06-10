from __future__ import annotations

import logging
import re

from transformers import PreTrainedTokenizerBase

try:
    from .vietnamese import (
        get_question_words,
        has_question_word,
        has_vietnamese,
        is_quality_sample,
        normalize_text,
        segment_texts,
        validate_answer_in_segmented,
    )
except ImportError:
    from vietnamese import (
        get_question_words,
        has_question_word,
        has_vietnamese,
        is_quality_sample,
        normalize_text,
        segment_texts,
        validate_answer_in_segmented,
    )

logger = logging.getLogger(__name__)


def _question_group(question: str, language: str) -> str:
    words = get_question_words(question, language)
    return words[0] if words else "Nhóm khác"


# ──────────────────────────────────────────────
#  Quality filtering (call BEFORE tokenization)
# ──────────────────────────────────────────────

def filter_qa_dataset(
    dataset,
    question_column: str,
    context_column: str,
    answers_column: str,
    language_column: str = "language",
    filter_question_word: bool = True,
) -> "Dataset":
    """Remove low-quality QA samples from a HuggingFace ``Dataset``.

    Applies two filters per row:
      1. :func:`is_quality_sample` — answer quality (length, match context)
      2. :func:`has_question_word` — question must contain a question word

    Returns a new ``Dataset`` with the same schema (fewer rows).
    """

    def _filter_row(row) -> bool:
        q = normalize_text(row[question_column])
        c = normalize_text(row[context_column])
        lang = row.get(language_column) if isinstance(row, dict) else row[language_column]
        ans = row.get(answers_column) if isinstance(row, dict) else row[answers_column]

        if not is_quality_sample(c, ans):
            return False
        if filter_question_word and not has_question_word(q, lang):
            return False
        return True

    n_before = len(dataset)
    filtered = dataset.filter(_filter_row, batched=False)
    n_after = len(filtered)
    removed = n_before - n_after
    if removed:
        logger.info(
            "Filter removed %d / %d samples (%.1f%%)",
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


def _tokens_with_word_offsets(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> tuple[list[str], list[tuple[int, int]]]:
    """Tokenize slow-tokenizer text and approximate offsets at word granularity."""
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for match in re.finditer(r"\S+", text):
        word = match.group()
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            continue
        tokens.extend(word_tokens)
        offsets.extend([(match.start(), match.end())] * len(word_tokens))
    return tokens, offsets


def _pad_slow_feature(
    input_ids: list[int],
    attention_mask: list[int],
    offset_mapping: list[tuple[int, int] | None] | None,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    padding: str,
) -> tuple[list[int], list[int], list[tuple[int, int] | None] | None]:
    if padding != "max_length":
        return input_ids, attention_mask, offset_mapping

    pad_len = max_length - len(input_ids)
    if pad_len <= 0:
        return input_ids[:max_length], attention_mask[:max_length], (
            offset_mapping[:max_length] if offset_mapping is not None else None
        )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("Tokenizer pad_token_id is required for max_length padding.")
    input_ids = input_ids + [pad_id] * pad_len
    attention_mask = attention_mask + [0] * pad_len
    if offset_mapping is not None:
        offset_mapping = offset_mapping + [None] * pad_len
    return input_ids, attention_mask, offset_mapping


def _append_slow_qa_features(
    features: dict[str, list],
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    context: str,
    sample_idx: int,
    max_length: int,
    doc_stride: int,
    padding: str,
    answer: dict | None = None,
    impossible: bool = False,
    question_group: str | None = None,
    include_labels: bool = True,
) -> None:
    q_tokens, _ = _tokens_with_word_offsets(tokenizer, question)
    c_tokens, c_offsets = _tokens_with_word_offsets(tokenizer, context)

    while q_tokens and max_length - len(q_tokens) - tokenizer.num_special_tokens_to_add(pair=True) < 1:
        q_tokens.pop()

    max_context_tokens = max_length - len(q_tokens) - tokenizer.num_special_tokens_to_add(pair=True)
    if max_context_tokens < 1:
        raise ValueError(
            f"max_length={max_length} leaves no room for context tokens "
            f"after tokenizing the question."
        )

    q_ids = tokenizer.convert_tokens_to_ids(q_tokens)
    step = max_context_tokens - min(doc_stride, max_context_tokens - 1)
    step = max(step, 1)
    span_starts = list(range(0, max(len(c_tokens), 1), step))
    if not c_tokens:
        span_starts = [0]

    answer_start = None
    answer_end = None
    if answer and answer.get("text") and answer["text"][0] and answer.get("answer_start"):
        answer_start = int(answer["answer_start"][0])
        answer_end = answer_start + len(answer["text"][0])

    for span_start in span_starts:
        span_end = min(span_start + max_context_tokens, len(c_tokens))
        if span_start >= len(c_tokens) and c_tokens:
            continue
        c_span_tokens = c_tokens[span_start:span_end]
        c_span_offsets = c_offsets[span_start:span_end]
        c_ids = tokenizer.convert_tokens_to_ids(c_span_tokens)

        input_ids = tokenizer.build_inputs_with_special_tokens(q_ids, c_ids)
        attention_mask = [1] * len(input_ids)

        context_start = len(input_ids) - len(c_ids) - 1
        offsets: list[tuple[int, int] | None] = [None] * len(input_ids)
        for rel_idx, offset in enumerate(c_span_offsets):
            offsets[context_start + rel_idx] = offset

        input_ids, attention_mask, offsets = _pad_slow_feature(
            input_ids,
            attention_mask,
            offsets,
            tokenizer,
            max_length,
            padding,
        )

        features["input_ids"].append(input_ids)
        features["attention_mask"].append(attention_mask)

        if include_labels:
            cls_index = input_ids.index(tokenizer.cls_token_id)
            start_position = cls_index
            end_position = cls_index
            if not impossible and answer_start is not None and answer_end is not None:
                token_start = None
                token_end = None
                for idx, offset in enumerate(offsets or []):
                    if offset is None:
                        continue
                    if token_start is None and offset[0] <= answer_start < offset[1]:
                        token_start = idx
                    if offset[0] < answer_end <= offset[1]:
                        token_end = idx
                if token_start is not None and token_end is not None:
                    start_position = token_start
                    end_position = token_end

            features["start_positions"].append(start_position)
            features["end_positions"].append(end_position)
            features["question_group"].append(question_group)
        else:
            features["offset_mapping"].append(offsets)
            features["sample_id"].append(sample_idx)

        if span_end >= len(c_tokens):
            break


def _prepare_train_features_slow(
    questions: list[str],
    contexts: list[str],
    answers: list[dict],
    is_impossible: list[bool],
    question_groups: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    doc_stride: int,
    padding: str,
) -> dict[str, list]:
    features = {
        "input_ids": [],
        "attention_mask": [],
        "start_positions": [],
        "end_positions": [],
        "question_group": [],
    }
    for idx, (question, context, answer, impossible, question_group) in enumerate(
        zip(questions, contexts, answers, is_impossible, question_groups)
    ):
        _append_slow_qa_features(
            features,
            tokenizer,
            question,
            context,
            idx,
            max_length,
            doc_stride,
            padding,
            answer=answer,
            impossible=bool(impossible),
            question_group=question_group,
            include_labels=True,
        )
    return features


def _prepare_eval_features_slow(
    questions: list[str],
    contexts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    doc_stride: int,
    padding: str,
    example_indices: list[int] | None = None,
) -> dict[str, list]:
    features = {
        "input_ids": [],
        "attention_mask": [],
        "offset_mapping": [],
        "sample_id": [],
    }
    for idx, (question, context) in enumerate(zip(questions, contexts)):
        sample_idx = idx if example_indices is None else int(example_indices[idx])
        _append_slow_qa_features(
            features,
            tokenizer,
            question,
            context,
            sample_idx,
            max_length,
            doc_stride,
            padding,
            include_labels=False,
        )
    return features


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
    languages = examples.get("language", ["en"] * len(questions))
    question_groups = [
        _question_group(question, language)
        for question, language in zip(questions, languages)
    ]

    if use_vietnamese_segmentation and has_vietnamese(examples):
        logger.info("Detected Vietnamese data, applying word segmentation")
        if segmentation_tool and segmentation_tool != "underthesea":
            logger.warning("Unsupported segmentation_tool=%s, using underthesea", segmentation_tool)
        contexts_orig = list(contexts)
        questions = segment_texts(questions)
        contexts = segment_texts(contexts)
        validate_answer_in_segmented(contexts_orig, contexts, answers)

    _ensure_pad_token(tokenizer, padding)

    if not getattr(tokenizer, "is_fast", False):
        return _prepare_train_features_slow(
            questions,
            contexts,
            answers,
            is_impossible,
            question_groups,
            tokenizer,
            max_length,
            doc_stride,
            padding,
        )

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

            if offsets[context_start][0] > answer_start_char or offsets[context_end][1] < answer_end_char:
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
    tokenized["question_group"] = [
        question_groups[sample_idx]
        for sample_idx in sample_mapping
    ]

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
    example_indices: list[int] | None = None,
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

    if not getattr(tokenizer, "is_fast", False):
        return _prepare_eval_features_slow(
            questions,
            contexts,
            tokenizer,
            max_length,
            doc_stride,
            padding,
            example_indices=example_indices,
        )

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
        if example_indices is None:
            tokenized["sample_id"].append(sample_idx)
        else:
            tokenized["sample_id"].append(example_indices[sample_idx])

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
