from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Noise patterns stripped during normalisation
_NOISE_PATTERNS = re.compile(
    r"("
    r"<[^>]+>"                              # HTML tags
    r"|https?://\S+"                        # URLs
    r"|[\\{\}\[\]|;:\"']{4,}"              # repeated punctuation
    r"|[\u200b\u200c\u200d\uFEFF]"          # zero-width characters
    r")",
    re.UNICODE,
)

# Answer quality thresholds
_MIN_ANSWER_LENGTH = 2
_MAX_ANSWER_LENGTH = 9999  # QA predicts start/end positions, length is irrelevant


def normalize_text(text: str) -> str:
    """Unicode normalisation and noise removal."""
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    text = _NOISE_PATTERNS.sub("", text)
    return text.strip()


def _is_word_char(char: str) -> bool:
    return char == "_" or char.isalnum()


def _has_answer_boundaries(context: str, start: int, answer: str) -> bool:
    end = start + len(answer)
    if start < 0 or end > len(context):
        return False

    if start > 0 and _is_word_char(context[start - 1]) and _is_word_char(answer[0]):
        return False
    if end < len(context) and _is_word_char(context[end]) and _is_word_char(answer[-1]):
        return False
    return True


def is_quality_sample(
    context: str,
    answer: dict | None,
    min_answer_len: int = _MIN_ANSWER_LENGTH,
    max_answer_len: int = _MAX_ANSWER_LENGTH,
) -> bool:
    """Check whether a single QA sample passes basic quality filters.

    Returns ``False`` (reject) when:
      - answer text is empty / too short / too long
      - answer text is not found inside the context
    """
    if answer and answer.get("text") and answer["text"][0]:
        ans = answer["text"][0].strip()
        answer_starts = answer.get("answer_start") or []
        ans_start = answer_starts[0] if answer_starts else None

        if len(ans) < min_answer_len:
            return False
        if len(ans) > max_answer_len:
            return False

        if ans_start is None:
            return False

        ctx_snippet = context[ans_start : ans_start + len(ans)]
        if ctx_snippet != ans:
            return False
        if not _has_answer_boundaries(context, ans_start, ans):
            return False
    else:
        # No answer – that's fine (is_impossible samples are valid)
        pass

    return True


def segment_texts(texts: list[str]) -> list[str]:
    """Word-segment Vietnamese texts using underthesea (format='text').

    Compounds are joined with underscores (e.g. ``sinh_viên``, ``thành_phố``)
    which helps the BERT tokenizer recognise word boundaries. The tokenizer may
    also insert spaces around punctuation, so answer offsets must be remapped
    after segmentation.

    Args:
        texts: List of Vietnamese sentences.

    Returns:
        List of segmented sentences.
    """
    try:
        from underthesea import word_tokenize
    except ImportError:
        logger.warning(
            "underthesea not installed, skip Vietnamese segmentation. "
            "Install with: pip install underthesea"
        )
        return texts

    result = []
    for text in texts:
        try:
            segmented = word_tokenize(text, format="text")
            result.append(segmented)
        except Exception:
            logger.warning("Failed to segment text, using original")
            result.append(text)
    return result


def _alignment_char(char: str) -> str:
    """Canonical single-character key used to align original and segmented text."""
    import unicodedata

    decomposed = unicodedata.normalize("NFD", char).casefold()
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _build_segmented_char_map(original: str, segmented: str) -> dict[int, int]:
    """Map original character indexes to segmented character indexes.

    Vietnamese word segmentation preserves the order of visible characters but
    can replace spaces with underscores and insert spaces around punctuation.
    For span alignment we therefore ignore spaces/underscores and match the
    remaining characters in order.
    """
    mapping: dict[int, int] = {}
    seg_idx = 0
    seg_len = len(segmented)

    for orig_idx, orig_char in enumerate(original):
        if orig_char.isspace() or orig_char == "_":
            continue

        orig_key = _alignment_char(orig_char)
        while seg_idx < seg_len and (segmented[seg_idx].isspace() or segmented[seg_idx] == "_"):
            seg_idx += 1

        while seg_idx < seg_len and _alignment_char(segmented[seg_idx]) != orig_key:
            seg_idx += 1

        if seg_idx >= seg_len:
            break

        mapping[orig_idx] = seg_idx
        seg_idx += 1

    return mapping


def _remap_answer_to_segmented(
    ctx_orig: str,
    ctx_seg: str,
    ans_text: str,
    ans_start: int,
) -> tuple[int, str] | None:
    ans_end = ans_start + len(ans_text)
    char_map = _build_segmented_char_map(ctx_orig, ctx_seg)

    answer_char_indexes = [
        idx
        for idx in range(ans_start, min(ans_end, len(ctx_orig)))
        if idx in char_map and not ctx_orig[idx].isspace() and ctx_orig[idx] != "_"
    ]
    if not answer_char_indexes:
        return None

    seg_start = char_map[answer_char_indexes[0]]
    seg_end = char_map[answer_char_indexes[-1]] + 1
    if seg_start >= seg_end:
        return None

    return seg_start, ctx_seg[seg_start:seg_end]


def _correct_original_answer_start(ctx_orig: str, ans_text: str, ans_start: int) -> int:
    if ctx_orig[ans_start : ans_start + len(ans_text)] == ans_text:
        return ans_start

    matches = [match.start() for match in re.finditer(re.escape(ans_text), ctx_orig)]
    if not matches:
        return ans_start
    return min(matches, key=lambda start: abs(start - ans_start))


def validate_answer_in_segmented(
    contexts_orig: list[str],
    contexts_seg: list[str],
    answers: list[dict],
) -> None:
    """Verify answer positions after word segmentation; fix if drifted.

    Mutates ``answers`` in-place by correcting ``answer_start`` and
    ``answer_text`` so they point to the correct location in the segmented
    context.
    """
    for ctx_orig, ctx_seg, ans in zip(contexts_orig, contexts_seg, answers):
        if not ans or not ans.get("text") or not ans["text"][0]:
            continue
        ans_text = ans["text"][0]
        if not ans.get("answer_start"):
            continue
        ans_start = ans["answer_start"][0]
        ans_start = _correct_original_answer_start(ctx_orig, ans_text, ans_start)

        remapped = _remap_answer_to_segmented(ctx_orig, ctx_seg, ans_text, ans_start)
        if remapped is not None:
            ans["answer_start"][0], ans["text"][0] = remapped
            continue

        # Fallback: search for the answer in the segmented context
        segmented_answer = segment_texts([ans_text])[0]
        pattern = re.escape(segmented_answer).replace(r"\ ", r"\s+")
        match = re.search(pattern, ctx_seg)
        if match:
            ans["answer_start"][0] = match.start()
            ans["text"][0] = match.group()
        else:
            # Cannot locate answer – mark as impossible later
            logger.debug("Answer not found in segmented context: %r", ans_text)


# ── Question word detection ──────────────────────
# Individual question words for frequency counting
_EN_QWORDS_PATTERN = re.compile(
    r"(?<![a-z])("
    r"what|when|where|which|who|whom|whose"
    r"|why|how"
    r")(?![a-z])",
    re.IGNORECASE,
)

_VI_QWORDS_PATTERN = re.compile(
    r"(?<![a-zà-ỹ])("
    r"ai là|người nào|ai"
    r"|cái gì|điều gì|gì"
    r"|ở nơi nào|nơi nào|ở đâu|đâu"
    r"|khi nào|bao giờ|lúc nào|mấy giờ"
    r"|tại sao|vì sao|do đâu"
    r"|bằng cách nào|làm thế nào|như thế nào|làm sao|thế nào|sao"
    r"|bao nhiêu|bao lâu|mấy"
    r"|nào"
    r")(?![a-zà-ỹ])",
    re.IGNORECASE,
)

def has_question_word(question: str, language: str) -> bool:
    """Check if a question contains at least one question word."""
    if language == "vi":
        return bool(_VI_QWORDS_PATTERN.search(question))
    return bool(_EN_QWORDS_PATTERN.search(question))


def get_question_words(question: str, language: str) -> list[str]:
    """Return unique question words found in the question, deduplicated by span overlap."""
    pattern = _VI_QWORDS_PATTERN if language == "vi" else _EN_QWORDS_PATTERN
    seen_words: set[str] = set()
    covered: set[int] = set()
    result: list[str] = []
    for m in pattern.finditer(question):
        # Skip if any character position is already covered by a longer match
        if any(pos in covered for pos in range(m.start(), m.end())):
            continue
        w = m.group(1).lower()
        if w not in seen_words:
            seen_words.add(w)
            result.append(w)
            covered.update(range(m.start(), m.end()))
    return result


def has_vietnamese(examples: dict[str, list], language_column: str = "language") -> bool:
    """Check if a batch of examples contains Vietnamese text.

    Looks at the ``language_column`` field if present.
    Falls back to checking for Vietnamese characters.
    """
    if language_column in examples:
        langs = set(examples[language_column])
        if "vi" in langs:
            return True

    # fallback: heuristic check for Vietnamese characters
    texts = _get_text_columns(examples)
    if not texts:
        return False
    combined = " ".join(texts)
    return _contains_vietnamese(combined)


def _get_text_columns(examples: dict[str, list]) -> list[str]:
    candidates = []
    for key in ("context", "question", "text"):
        if key in examples and examples[key]:
            val = examples[key]
            if isinstance(val, list):
                candidates.extend(str(v) for v in val if v)
    return candidates


def _contains_vietnamese(text: str) -> bool:
    """Heuristic: check for common Vietnamese characters."""
    import re
    vietnamese_chars = re.compile(
        r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩ"
        r"òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ"
        r"đÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨ"
        r"ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]"
    )
    return bool(vietnamese_chars.search(text))
