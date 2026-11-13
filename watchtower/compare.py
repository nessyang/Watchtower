"""Comparison engine: character-level differences between two plate readings."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .plate_text import are_confusable, normalize_plate

EQUAL = "equal"
REPLACE = "replace"
INSERT = "insert"  # character present in the ALPR reading but not the AI reading
DELETE = "delete"  # character present in the AI reading but not the ALPR reading


@dataclass(frozen=True)
class DiffSegment:
    """One aligned piece of the two readings."""

    op: str
    ai: str
    alpr: str
    confusable: bool = False


@dataclass(frozen=True)
class Comparison:
    ai_normalized: str
    alpr_normalized: str
    match: bool
    similarity: float  # 0.0 – 1.0, 1 - edit_distance / max_len
    edit_distance: int
    segments: tuple[DiffSegment, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def differing_positions(self) -> int:
        return sum(1 for s in self.segments if s.op != EQUAL)

    @property
    def all_differences_confusable(self) -> bool:
        diffs = [s for s in self.segments if s.op != EQUAL]
        return bool(diffs) and all(s.confusable for s in diffs)


def levenshtein(a: str, b: str) -> int:
    """Classic edit distance (insert / delete / substitute, each cost 1)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _positional_segments(ai: str, alpr: str) -> list[DiffSegment]:
    """Character-by-character alignment for equal-length readings."""
    segments: list[DiffSegment] = []
    for ca, cb in zip(ai, alpr):
        if ca == cb:
            if segments and segments[-1].op == EQUAL:
                prev = segments.pop()
                segments.append(DiffSegment(EQUAL, prev.ai + ca, prev.alpr + cb))
            else:
                segments.append(DiffSegment(EQUAL, ca, cb))
        else:
            segments.append(DiffSegment(REPLACE, ca, cb, are_confusable(ca, cb)))
    return segments


def _segments(ai: str, alpr: str, distance: int) -> list[DiffSegment]:
    # A reviewer compares equal-length plates position by position, so when that
    # explains the difference as cheaply as any alignment would (e.g. swapped
    # digits "3392" vs "3329"), show it that way.
    if len(ai) == len(alpr) and sum(a != b for a, b in zip(ai, alpr)) == distance:
        return _positional_segments(ai, alpr)

    segments: list[DiffSegment] = []
    matcher = SequenceMatcher(None, ai, alpr, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        a, b = ai[i1:i2], alpr[j1:j2]
        if op == EQUAL:
            segments.append(DiffSegment(EQUAL, a, b))
        elif op == REPLACE and len(a) == len(b):
            # Split equal-length substitutions per character so each one can be
            # highlighted and checked against the confusable-pair table.
            for ca, cb in zip(a, b):
                segments.append(DiffSegment(REPLACE, ca, cb, are_confusable(ca, cb)))
        elif op == REPLACE:
            segments.append(DiffSegment(REPLACE, a, b))
        elif op == DELETE:
            segments.append(DiffSegment(DELETE, a, ""))
        elif op == INSERT:
            segments.append(DiffSegment(INSERT, "", b))
    return segments


def compare_readings(ai_reading: str, alpr_reading: str) -> Comparison:
    """Compare two plate readings after normalization (case/spacing ignored)."""
    ai = normalize_plate(ai_reading)
    alpr = normalize_plate(alpr_reading)
    distance = levenshtein(ai, alpr)
    longest = max(len(ai), len(alpr))
    similarity = 1.0 if longest == 0 else round(1 - distance / longest, 3)
    segments = tuple(_segments(ai, alpr, distance))

    notes: list[str] = []
    comparison = Comparison(ai, alpr, ai == alpr, similarity, distance, segments)
    if not comparison.match:
        if len(ai) != len(alpr):
            notes.append(
                f"Readings have different lengths ({len(ai)} vs {len(alpr)} characters)."
            )
        pairs = [f"{s.ai}↔{s.alpr}" for s in segments if s.confusable]
        if pairs:
            notes.append(
                "Commonly confused character pair(s): "
                + ", ".join(pairs)
                + ". One system likely misread the plate — a person must check the image."
            )
    return Comparison(ai, alpr, ai == alpr, similarity, distance, segments, tuple(notes))
