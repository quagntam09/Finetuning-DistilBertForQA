from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def segment_texts(texts: list[str]) -> list[str]:
    """Word-segment Vietnamese texts using underthesea.

    Args:
        texts: List of Vietnamese sentences.

    Returns:
        List of segmented sentences (words separated by spaces).
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
            tokens = word_tokenize(text)
            result.append(" ".join(tokens))
        except Exception:
            logger.warning("Failed to segment text, using original")
            result.append(text)
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
