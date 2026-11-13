from watchtower.compare import DELETE, EQUAL, INSERT, REPLACE, compare_readings, levenshtein
from watchtower.plate_text import are_confusable, is_plausible_plate, normalize_plate


def test_normalize_strips_formatting_and_uppercases():
    assert normalize_plate(" abc-1234 ") == "ABC1234"
    assert normalize_plate("B.X R 48·27") == "BXR4827"
    assert normalize_plate("") == ""
    assert normalize_plate(None) == ""


def test_plausible_plate_lengths():
    assert is_plausible_plate("AB")
    assert is_plausible_plate("ABCD1234")
    assert not is_plausible_plate("A")
    assert not is_plausible_plate("ABCDE12345")


def test_confusable_pairs():
    assert are_confusable("8", "B")
    assert are_confusable("o", "0")
    assert are_confusable("1", "I")
    assert not are_confusable("A", "A")
    assert not are_confusable("X", "7")
    assert not are_confusable("AB", "8")


def test_levenshtein():
    assert levenshtein("", "") == 0
    assert levenshtein("ABC", "") == 3
    assert levenshtein("KITTEN", "SITTING") == 3
    assert levenshtein("HWN3392", "HWN3329") == 2


def test_exact_match_ignores_case_and_spacing():
    cmp = compare_readings("bxr 4827", "BXR-4827")
    assert cmp.match
    assert cmp.similarity == 1.0
    assert cmp.edit_distance == 0
    assert all(s.op == EQUAL for s in cmp.segments)
    assert cmp.notes == ()


def test_confusable_substitution_is_flagged_but_still_a_mismatch():
    cmp = compare_readings("DKT8150", "DKTB150")
    assert not cmp.match
    diffs = [s for s in cmp.segments if s.op != EQUAL]
    assert len(diffs) == 1
    assert diffs[0].op == REPLACE and diffs[0].ai == "8" and diffs[0].alpr == "B"
    assert diffs[0].confusable
    assert cmp.all_differences_confusable
    assert any("Commonly confused" in n for n in cmp.notes)


def test_non_confusable_substitution():
    cmp = compare_readings("ABC1234", "ABX1234")
    diffs = [s for s in cmp.segments if s.op != EQUAL]
    assert [(d.ai, d.alpr, d.confusable) for d in diffs] == [("C", "X", False)]
    assert not cmp.all_differences_confusable
    assert cmp.similarity == round(1 - 1 / 7, 3)


def test_missing_and_extra_characters():
    missing = compare_readings("ABC1234", "ABC234")  # ALPR dropped a character
    assert [s.op for s in missing.segments if s.op != EQUAL] == [DELETE]
    assert any("different lengths" in n for n in missing.notes)

    extra = compare_readings("ABC234", "ABC1234")  # ALPR has an extra character
    assert [s.op for s in extra.segments if s.op != EQUAL] == [INSERT]


def test_transposed_characters_align_by_position():
    cmp = compare_readings("HWN3392", "HWN3329")
    diffs = [(s.op, s.ai, s.alpr) for s in cmp.segments if s.op != EQUAL]
    assert diffs == [(REPLACE, "9", "2"), (REPLACE, "2", "9")]
    assert cmp.segments[0].ai == "HWN33"  # equal run kept together


def test_shifted_reading_uses_sequence_alignment():
    # One dropped + one extra character: positional diff would flag all 7.
    cmp = compare_readings("ABC1234", "BC12345")
    assert [s.op for s in cmp.segments if s.op != EQUAL] == [DELETE, INSERT]


def test_segments_reconstruct_both_readings():
    for ai, alpr in [("HWN3392", "HWN3329"), ("AB12", "XY9876"), ("A", "")]:
        cmp = compare_readings(ai, alpr)
        assert "".join(s.ai for s in cmp.segments) == cmp.ai_normalized
        assert "".join(s.alpr for s in cmp.segments) == cmp.alpr_normalized
