"""Cheap, dependency-free token estimation.

Exact tokenizers are model specific and expensive to ship.  For the purpose of
context budgeting we only need a conservative estimate, so we use a hybrid
character heuristic: CJK glyphs are close to 1 token each, latin text is close
to 1 token per 3.6 characters.  The estimate deliberately over-counts a little.
"""

from __future__ import annotations

from typing import Any, Iterable

_CJK_RANGES = (
    (0x3040, 0x30FF),  # kana
    (0x3400, 0x4DBF),  # cjk ext a
    (0x4E00, 0x9FFF),  # cjk unified
    (0xAC00, 0xD7AF),  # hangul
    (0xF900, 0xFAFF),  # compatibility
    (0xFF00, 0xFFEF),  # full width forms
)


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    for low, high in _CJK_RANGES:
        if low <= code <= high:
            return True
    return False


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    cjk = 0
    for ch in text:
        if _is_cjk(ch):
            cjk += 1
    other = len(text) - cjk
    return int(cjk * 1.05 + other / 3.6) + 1


def estimate_messages(messages: Iterable[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        total += 4  # role / separators
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += estimate_tokens(str(part.get("text", "")))
    return total


def truncate_to_tokens(text: str, limit: int) -> str:
    """Hard character-level truncation that respects the token estimate."""
    if limit <= 0 or estimate_tokens(text) <= limit:
        return text
    # binary search on characters keeps this cheap for the sizes we handle
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[:mid]) <= limit:
            low = mid
        else:
            high = mid - 1
    return text[:low]
