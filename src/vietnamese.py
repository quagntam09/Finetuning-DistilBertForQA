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
_MAX_ANSWER_LENGTH = 150


def normalize_text(text: str) -> str:
    """Unicode normalisation and noise removal."""
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    text = _NOISE_PATTERNS.sub("", text)
    return text.strip()


def is_quality_sample(
    context: str,
    question: str,
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
        ans_start = answer.get("answer_start", [None])[0]

        if len(ans) < min_answer_len:
            return False
        if len(ans) > max_answer_len:
            return False

        if ans_start is not None:
            ctx_snippet = context[ans_start : ans_start + len(ans)]
            if ctx_snippet != ans:
                return False
    else:
        # No answer – that's fine (is_impossible samples are valid)
        pass

    return True


def segment_texts(texts: list[str]) -> list[str]:
    """Word-segment Vietnamese texts using underthesea (format='text').

    Compounds are joined with underscores (e.g. ``sinh_viên``, ``thành_phố``)
    which helps the BERT tokenizer recognise word boundaries.
    Character positions are preserved (underscore ↔ space, 1-to-1).

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
    import re

    for ctx_orig, ctx_seg, ans in zip(contexts_orig, contexts_seg, answers):
        if not ans or not ans.get("text") or not ans["text"][0]:
            continue
        ans_text = ans["text"][0]
        ans_start = ans["answer_start"][0]

        seg_region = ctx_seg[ans_start : ans_start + len(ans_text)]
        if seg_region.replace("_", " ") == ans_text:
            # Exact position match (common case: spaces → underscores 1:1)
            ans["text"][0] = seg_region
            continue

        # Fallback: search for the answer in the segmented context
        pattern = re.escape(ans_text).replace(r"\ ", "[_ ]")
        match = re.search(pattern, ctx_seg)
        if match:
            ans["answer_start"][0] = match.start()
            ans["text"][0] = match.group()
        else:
            # Cannot locate answer – mark as impossible later
            logger.debug("Answer not found in segmented context: %r", ans_text)


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
