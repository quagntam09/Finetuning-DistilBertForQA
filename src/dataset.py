from __future__ import annotations

import logging

from transformers import PreTrainedTokenizerBase

try:
    from .vietnamese import has_vietnamese, segment_texts
except ImportError:
    from vietnamese import has_vietnamese, segment_texts

logger = logging.getLogger(__name__)


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
) -> dict[str, list]:
    questions = [q.strip() for q in examples[question_column]]
    contexts = list(examples[context_column])
    answers = examples[answers_column]
    is_impossible = examples[impossible_column]

    # Normalize to fixed keys so has_vietnamese works with custom schemas
    if has_vietnamese({"question": questions, "context": contexts}):
        logger.info("Detected Vietnamese data, applying word segmentation")
        questions = segment_texts(questions)
        contexts = segment_texts(contexts)

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
) -> dict[str, list]:
    questions = [q.strip() for q in examples[question_column]]
    contexts = list(examples[context_column])

    if has_vietnamese(examples):
        logger.info("Detected Vietnamese data, applying word segmentation")
        questions = segment_texts(questions)
        contexts = segment_texts(contexts)

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
