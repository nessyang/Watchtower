"""Plate-string helpers: normalization and OCR-confusable characters.

Kept free of any image or OCR dependency so it is trivial to unit test.
"""

from __future__ import annotations

import re

# US plates are at most 8 characters; very short strings are almost always noise.
MIN_PLATE_LEN = 2
MAX_PLATE_LEN = 8

_NON_ALNUM = re.compile(r"[^A-Z0-9]")

# Character pairs that optical character recognition (human or machine) frequently
# confuses on license plates. Used to *explain* a discrepancy, never to excuse it.
_CONFUSABLE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset("0OQD"),
    frozenset("1IL"),
    frozenset("8B"),
    frozenset("5S"),
    frozenset("2Z"),
    frozenset("6G"),
    frozenset("4A"),
    frozenset("7T"),
    frozenset("UV"),
    frozenset("MN"),
)


def normalize_plate(text: str | None) -> str:
    """Uppercase and strip everything except A-Z and 0-9.

    Spaces, dashes and dots are formatting only, so "abc-1234" == "ABC 1234".
    """
    if not text:
        return ""
    return _NON_ALNUM.sub("", text.upper())


def is_plausible_plate(text: str) -> bool:
    """True if a *normalized* string has a plausible license-plate length."""
    return MIN_PLATE_LEN <= len(text) <= MAX_PLATE_LEN


def are_confusable(a: str, b: str) -> bool:
    """True if two single characters are a known OCR confusion pair."""
    if len(a) != 1 or len(b) != 1 or a == b:
        return False
    a, b = a.upper(), b.upper()
    return any(a in group and b in group for group in _CONFUSABLE_GROUPS)
