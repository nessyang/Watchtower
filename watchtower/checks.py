"""Alert corroboration checks against FICTIONAL incident records.

Three automated checks a person must see satisfied before an ALPR alert is even
considered:

1. Plate known      – is there an *active* record for the plate, and does the
                      independent reading agree with it?
2. Vehicle features – do the features a reviewer observed match the record?
3. Corroboration    – has independent corroboration been documented?

These checks never authorize a stop. The best possible result is "all checks
met — a human decision-maker may now evaluate this alert".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .plate_text import normalize_plate
from .verify import Outcome, VerificationResult

DEFAULT_RECORDS_PATH = Path(__file__).resolve().parent.parent / "data" / "incidents.json"
MIN_NOTE_LENGTH = 20
MIN_FEATURE_MATCHES = 2

INDEPENDENT_METHODS = (
    "Officer visually confirmed plate characters in person",
    "Second independent image of the plate reviewed",
    "Supervisor reviewed the original ALPR image",
    "Hot-list entry re-confirmed with originating agency",
)
# Listed so the UI can show *why* they do not count.
NON_INDEPENDENT_METHODS = (
    "ALPR alert only",
    "Watchtower AI agreement",
)


class RecordError(ValueError):
    """The incident-record file is missing or malformed."""


class CheckStatus(str, Enum):
    MET = "met"
    NOT_MET = "not_met"
    NEEDS_REVIEW = "needs_review"

    @property
    def label(self) -> str:
        return {"met": "Met", "not_met": "Not met", "needs_review": "Needs review"}[self.value]


@dataclass(frozen=True)
class VehicleDescription:
    color: str = ""
    make: str = ""
    model: str = ""
    features: tuple[str, ...] = ()


@dataclass(frozen=True)
class IncidentRecord:
    incident_id: str
    plate: str  # normalized
    state: str
    status: str  # "active" | "cleared"
    reason: str
    entered: str
    vehicle: VehicleDescription

    @property
    def is_active(self) -> bool:
        return self.status == "active"


@dataclass(frozen=True)
class CorroborationLog:
    methods: tuple[str, ...] = ()
    reviewer_id: str = ""
    note: str = ""


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    detail: str


@dataclass(frozen=True)
class CheckSummary:
    checks: tuple[CheckResult, ...]
    record: IncidentRecord | None
    headline: str = ""
    advice: str = ""
    status: CheckStatus = CheckStatus.NEEDS_REVIEW

    def to_record(self) -> dict:
        return {
            "overall": self.status.value,
            "headline": self.headline,
            "incident_id": self.record.incident_id if self.record else None,
            "checks": [
                {"name": c.name, "status": c.status.value, "detail": c.detail}
                for c in self.checks
            ],
            "authorizes_action": False,
        }


# --- records -----------------------------------------------------------------

def load_records(path: Path | str = DEFAULT_RECORDS_PATH) -> list[IncidentRecord]:
    """Load fictional incident records from JSON, validating each entry."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = raw["records"]
    except FileNotFoundError as exc:
        raise RecordError(f"Incident record file not found: {path}") from exc
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RecordError(f"Incident record file is malformed: {exc}") from exc

    records = []
    for i, e in enumerate(entries):
        try:
            v = e.get("vehicle", {})
            status = str(e["status"]).lower()
            if status not in ("active", "cleared"):
                raise ValueError(f"unknown status '{status}'")
            records.append(IncidentRecord(
                incident_id=str(e["incident_id"]),
                plate=normalize_plate(e["plate"]),
                state=str(e.get("state", "")),
                status=status,
                reason=str(e.get("reason", "")),
                entered=str(e.get("entered", "")),
                vehicle=VehicleDescription(
                    color=str(v.get("color", "")), make=str(v.get("make", "")),
                    model=str(v.get("model", "")), features=tuple(v.get("features", [])),
                ),
            ))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise RecordError(f"Record #{i + 1} is invalid: {exc}") from exc
    return records


def find_record(plate: str, records: list[IncidentRecord]) -> IncidentRecord | None:
    """Return the record for a plate, preferring an active one."""
    matches = [r for r in records if r.plate == normalize_plate(plate)]
    if not matches:
        return None
    return next((r for r in matches if r.is_active), matches[0])


# --- the three checks --------------------------------------------------------

def check_plate_known(result: VerificationResult, record: IncidentRecord | None) -> CheckResult:
    name = "Suspect plate is known"
    plate = result.alpr_normalized
    if record is None:
        return CheckResult(name, CheckStatus.NOT_MET,
                           f"'{plate}' is not on any incident record, so there is no basis for an alert.")
    if not record.is_active:
        return CheckResult(name, CheckStatus.NOT_MET,
                           f"Record {record.incident_id} for '{plate}' is {record.status} "
                           f"({record.reason}). The alert is stale and must not be acted on.")
    if result.outcome is Outcome.AGREE:
        return CheckResult(name, CheckStatus.MET,
                           f"Active record {record.incident_id}, and the independent reading "
                           f"confirms the characters '{plate}'.")
    if result.outcome is Outcome.DISAGREE:
        ai = result.ai_reading.text if result.ai_reading else "?"
        return CheckResult(name, CheckStatus.NOT_MET,
                           f"The ALPR reading matches active record {record.incident_id}, but the "
                           f"independent reading is '{ai}'. The vehicle in the image may not be "
                           "the listed vehicle.")
    return CheckResult(name, CheckStatus.NEEDS_REVIEW,
                       f"Active record {record.incident_id} exists, but the plate in the image "
                       "could not be independently read. A person must confirm the characters.")


def _same(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()


def check_vehicle_features(observed: VehicleDescription, record: IncidentRecord | None) -> CheckResult:
    name = "Distinctive vehicle features match"
    if record is None:
        return CheckResult(name, CheckStatus.NEEDS_REVIEW, "No incident record to compare against.")

    matches, mismatches = [], []
    for attr in ("color", "make", "model"):
        seen, listed = getattr(observed, attr), getattr(record.vehicle, attr)
        if not seen.strip() or not listed.strip():
            continue  # not visible / not recorded: neither supports nor contradicts
        if _same(seen, listed):
            matches.append(f"{attr} {seen}")
        else:
            mismatches.append(f"{attr}: observed {seen}, record says {listed}")
    listed_features = {f.casefold(): f for f in record.vehicle.features}
    matches += [f"feature '{listed_features[f.casefold()]}'"
                for f in observed.features if f.casefold() in listed_features]

    if mismatches:
        return CheckResult(name, CheckStatus.NOT_MET,
                           "Contradiction with the record: " + "; ".join(mismatches) + ".")
    if len(matches) >= MIN_FEATURE_MATCHES:
        return CheckResult(name, CheckStatus.MET, "Matches: " + ", ".join(matches) + ".")
    return CheckResult(name, CheckStatus.NEEDS_REVIEW,
                       f"Only {len(matches)} matching feature(s) observed; at least "
                       f"{MIN_FEATURE_MATCHES} are needed and none may contradict the record.")


def check_corroboration(log: CorroborationLog) -> CheckResult:
    name = "Independent corroboration documented"
    independent = [m for m in log.methods if m in INDEPENDENT_METHODS]
    dependent = [m for m in log.methods if m not in INDEPENDENT_METHODS]
    if not independent:
        if dependent:
            return CheckResult(name, CheckStatus.NOT_MET,
                               f"{', '.join(dependent)} is not independent corroboration: "
                               "automated systems cannot corroborate themselves.")
        return CheckResult(name, CheckStatus.NOT_MET, "No independent corroboration has been documented.")

    missing = []
    if not log.reviewer_id.strip():
        missing.append("reviewer ID")
    if len(log.note.strip()) < MIN_NOTE_LENGTH:
        missing.append(f"a note of at least {MIN_NOTE_LENGTH} characters describing what was observed")
    if missing:
        return CheckResult(name, CheckStatus.NEEDS_REVIEW, "Documentation incomplete: add " + " and ".join(missing) + ".")
    return CheckResult(name, CheckStatus.MET,
                       f"Documented by {log.reviewer_id.strip()}: {', '.join(independent)}.")


def run_checks(
    result: VerificationResult,
    records: list[IncidentRecord],
    observed: VehicleDescription,
    log: CorroborationLog,
) -> CheckSummary:
    record = find_record(result.alpr_normalized, records)
    checks = (
        check_plate_known(result, record),
        check_vehicle_features(observed, record),
        check_corroboration(log),
    )
    not_met = sum(c.status is CheckStatus.NOT_MET for c in checks)
    review = sum(c.status is CheckStatus.NEEDS_REVIEW for c in checks)
    if not_met:
        status = CheckStatus.NOT_MET
        headline = f"Do not act on this alert — {not_met} of 3 checks not met"
        advice = "At least one check contradicts or fails to support the alert."
    elif review:
        status = CheckStatus.NEEDS_REVIEW
        headline = f"Incomplete — {review} of 3 checks need review"
        advice = "Gather the missing information before this alert can be evaluated."
    else:
        status = CheckStatus.MET
        headline = "All 3 checks met — ready for a human decision"
        advice = ("Watchtower does not authorize stops. A trained, accountable person must "
                  "still decide, under applicable law and policy, whether any action is justified.")
    return CheckSummary(checks, record, headline, advice, status)
