import json

import numpy as np
import pytest

from watchtower.checks import (
    CheckStatus,
    CorroborationLog,
    RecordError,
    VehicleDescription,
    check_corroboration,
    check_plate_known,
    check_vehicle_features,
    find_record,
    load_records,
    run_checks,
)
from watchtower.ocr import OcrDetection, PlateReader
from watchtower.verify import verify

RECORDS = load_records()  # the shipped fictional data file
IMAGE = np.full((300, 600, 3), 200, np.uint8)
GOOD_LOG = CorroborationLog(
    methods=("Officer visually confirmed plate characters in person",),
    reviewer_id="R-102",
    note="Plate BXR 4827 read in person from 3 m in daylight.",
)


def verified(alpr, ai_text=None, conf=0.95):
    dets = [OcrDetection(ai_text, conf, (100, 100, 500, 190))] if ai_text else []
    return verify(IMAGE, alpr, PlateReader(lambda img: dets))


# --- records -----------------------------------------------------------------

def test_shipped_records_load_and_are_fictional():
    assert len(RECORDS) >= 5
    assert all(r.reason.lower().count("fictional") for r in RECORDS)
    assert find_record("bxr-4827", RECORDS).incident_id == "INC-2026-0141"
    assert find_record("ZZZ0000", RECORDS) is None


def test_find_record_prefers_active(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"records": [
        {"incident_id": "OLD", "plate": "ABC123", "status": "cleared"},
        {"incident_id": "NEW", "plate": "ABC 123", "status": "active"},
    ]}))
    assert find_record("ABC123", load_records(path)).incident_id == "NEW"


@pytest.mark.parametrize("content", ["not json", '{"nope": []}',
                                     '{"records": [{"plate": "A1"}]}',
                                     '{"records": [{"incident_id": "X", "plate": "A1", "status": "maybe"}]}'])
def test_malformed_records_raise(tmp_path, content):
    path = tmp_path / "r.json"
    path.write_text(content)
    with pytest.raises(RecordError):
        load_records(path)


def test_missing_records_file_raises(tmp_path):
    with pytest.raises(RecordError, match="not found"):
        load_records(tmp_path / "missing.json")


# --- check 1: plate known ----------------------------------------------------

def test_plate_known_met_when_active_and_independently_confirmed():
    res = verified("BXR 4827", "BXR 4827")
    assert check_plate_known(res, find_record("BXR4827", RECORDS)).status is CheckStatus.MET


def test_plate_not_met_when_misread_hits_a_listed_plate():
    # ALPR misread DKT8150 as DKTB150, which happens to be on a record.
    res = verified("DKT B150", "DKT 8150")
    check = check_plate_known(res, find_record("DKTB150", RECORDS))
    assert check.status is CheckStatus.NOT_MET
    assert "may not be the listed vehicle" in check.detail


def test_plate_not_met_when_unlisted_or_cleared():
    assert check_plate_known(verified("ABC 999", "ABC 999"), None).status is CheckStatus.NOT_MET
    cleared = find_record("MTL7731", RECORDS)
    check = check_plate_known(verified("MTL 7731", "MTL 7731"), cleared)
    assert check.status is CheckStatus.NOT_MET and "stale" in check.detail


def test_plate_needs_review_when_image_unreadable():
    res = verified("CPL 6604")  # no OCR detections
    assert check_plate_known(res, find_record("CPL6604", RECORDS)).status is CheckStatus.NEEDS_REVIEW


# --- check 2: vehicle features -----------------------------------------------

RECORD = find_record("BXR4827", RECORDS)  # Silver Honda Civic, roof rack, dented bumper


def test_features_met_with_two_matches_case_insensitive():
    obs = VehicleDescription(color="silver", features=("Roof rack",))
    assert check_vehicle_features(obs, RECORD).status is CheckStatus.MET


def test_features_any_contradiction_fails():
    obs = VehicleDescription(color="Red", make="Honda", model="Civic", features=("Roof rack",))
    check = check_vehicle_features(obs, RECORD)
    assert check.status is CheckStatus.NOT_MET and "record says Silver" in check.detail


def test_features_not_visible_needs_review():
    assert check_vehicle_features(VehicleDescription(), RECORD).status is CheckStatus.NEEDS_REVIEW
    one = VehicleDescription(make="Honda")
    assert check_vehicle_features(one, RECORD).status is CheckStatus.NEEDS_REVIEW
    assert check_vehicle_features(one, None).status is CheckStatus.NEEDS_REVIEW


def test_unlisted_observed_feature_is_not_a_contradiction():
    obs = VehicleDescription(color="Silver", make="Honda", features=("Bike rack",))
    assert check_vehicle_features(obs, RECORD).status is CheckStatus.MET


# --- check 3: corroboration --------------------------------------------------

def test_corroboration_met_when_fully_documented():
    assert check_corroboration(GOOD_LOG).status is CheckStatus.MET


def test_automated_agreement_is_not_corroboration():
    log = CorroborationLog(("Watchtower AI agreement", "ALPR alert only"), "R-102", "x" * 40)
    check = check_corroboration(log)
    assert check.status is CheckStatus.NOT_MET and "cannot corroborate themselves" in check.detail


def test_corroboration_missing_or_incomplete():
    assert check_corroboration(CorroborationLog()).status is CheckStatus.NOT_MET
    partial = CorroborationLog(GOOD_LOG.methods, reviewer_id="", note="short")
    check = check_corroboration(partial)
    assert check.status is CheckStatus.NEEDS_REVIEW
    assert "reviewer ID" in check.detail and "note" in check.detail


# --- overall summary ---------------------------------------------------------

def test_all_met_never_authorizes_action():
    summary = run_checks(verified("BXR 4827", "BXR 4827"), RECORDS,
                         VehicleDescription(color="Silver", make="Honda"), GOOD_LOG)
    assert summary.status is CheckStatus.MET
    assert "does not authorize" in summary.advice
    assert summary.to_record()["authorizes_action"] is False


def test_any_not_met_means_do_not_act():
    summary = run_checks(verified("ELM 2045", "ELM 2045"), RECORDS,
                         VehicleDescription(color="Red"), GOOD_LOG)
    assert summary.status is CheckStatus.NOT_MET
    assert summary.headline.startswith("Do not act")
    assert [c.status for c in summary.checks] == [CheckStatus.MET, CheckStatus.NOT_MET, CheckStatus.MET]


def test_review_status_when_nothing_contradicts_but_info_missing():
    summary = run_checks(verified("BXR 4827", "BXR 4827"), RECORDS, VehicleDescription(), GOOD_LOG)
    assert summary.status is CheckStatus.NEEDS_REVIEW
