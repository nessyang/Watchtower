"""Watchtower — Streamlit dashboard.

Run with:  streamlit run app.py
All decision logic lives in the ``watchtower`` package; this file is UI only.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from watchtower.compare import EQUAL, Comparison
from watchtower.imaging import ImageError, draw_box, load_image
from watchtower.ocr import PlateReader
from watchtower.verify import (
    DEFAULT_MIN_CONFIDENCE,
    InvalidReadingError,
    Outcome,
    VerificationResult,
    verify,
)

log = logging.getLogger("watchtower")
SAMPLES_DIR = Path(__file__).parent / "samples"


@dataclass(frozen=True)
class Scenario:
    label: str
    image: str
    alpr: str
    story: str


SCENARIOS = [
    Scenario("1 · Clear plate, readings match", "plate_clear.png", "BXR 4827",
             "ALPR reading matches the plate. Still requires a human check."),
    Scenario("2 · ALPR misread (8 → B)", "plate_confusable.png", "DKT B150",
             "A classic OCR confusion: the ALPR system read the digit 8 as the letter B."),
    Scenario("3 · Transposed digits", "plate_mismatch.png", "HWN 3329",
             "The ALPR reading has two digits swapped — a different plate entirely."),
    Scenario("4 · Unreadable capture", "plate_unreadable.png", "CPL 6604",
             "Night-time motion blur. The ALPR system still produced a confident read."),
    Scenario("5 · Vehicle photo", "vehicle_scene.png", "ELM 2045",
             "A full (staged) vehicle photo instead of a cropped plate."),
]

OUTCOME_STYLE = {
    Outcome.AGREE: ("wt-agree", "✓", "Human verification required"),
    Outcome.DISAGREE: ("wt-disagree", "⚠", "Discrepancy — do not act on ALPR alert"),
    Outcome.INCONCLUSIVE: ("wt-inconclusive", "?", "Manual review required"),
}

REVIEW_CHECKLIST = [
    "Look at the image yourself and read every character of the plate.",
    "Confirm the issuing state / plate type, not just the characters.",
    "Confirm the vehicle make, model and colour match the hot-list record.",
    "Confirm the hot-list entry is current and was entered for a lawful reason.",
    "Document your independent observations before any action is taken.",
]

CSS = """
<style>
:root { --wt-ink:#0f172a; --wt-muted:#64748b; --wt-line:#e2e8f0; --wt-card:#ffffff;
        --wt-blue:#1d4ed8; --wt-red:#b91c1c; --wt-amber:#b45309; --wt-slate:#475569; }
.block-container { padding-top: 3.2rem; max-width: 1200px; }
.wt-hero { display:flex; align-items:center; gap:14px; padding:18px 22px; border-radius:14px;
           background:linear-gradient(120deg,#0b1d3a 0%,#1e3a8a 100%); color:#fff; margin-bottom:12px; }
.wt-hero .logo { font-size:34px; line-height:1; }
.wt-hero h1 { margin:0; font-size:1.9rem; font-weight:750; color:#fff; letter-spacing:-.01em; padding:0; }
.wt-hero p { margin:2px 0 0 0; color:#c7d2fe; font-size:.95rem; }
.wt-notice { border:1px solid #fde68a; background:#fffbeb; color:#78350f; border-radius:10px;
             padding:10px 14px; font-size:.88rem; margin-bottom:18px; }
.wt-verdict { border-radius:14px; padding:18px 20px; border:1px solid; margin-bottom:14px; }
.wt-verdict .badge { display:inline-block; font-size:.72rem; font-weight:700; letter-spacing:.06em;
                     text-transform:uppercase; padding:3px 9px; border-radius:999px; color:#fff; }
.wt-verdict h3 { margin:.45rem 0 .3rem 0; font-size:1.25rem; padding:0; }
.wt-verdict p { margin:0; font-size:.92rem; }
.wt-agree { background:#eff6ff; border-color:#bfdbfe; color:#1e3a8a; }
.wt-agree .badge { background:var(--wt-blue); }
.wt-disagree { background:#fef2f2; border-color:#fecaca; color:#7f1d1d; }
.wt-disagree .badge { background:var(--wt-red); }
.wt-inconclusive { background:#f8fafc; border-color:#cbd5e1; color:#1e293b; }
.wt-inconclusive .badge { background:var(--wt-slate); }
.wt-diff { display:grid; grid-template-columns:auto 1fr; gap:8px 14px; align-items:center; margin:6px 0 4px 0; }
.wt-diff .lbl { font-size:.78rem; font-weight:650; color:var(--wt-muted); text-transform:uppercase; letter-spacing:.05em; }
.wt-row { display:flex; flex-wrap:nowrap; gap:5px; }
.wt-ch { width:clamp(26px,9vw,38px); height:clamp(34px,11vw,46px); flex:none;
         display:flex; align-items:center; justify-content:center;
         font:700 clamp(1rem,5vw,1.45rem) ui-monospace,SFMono-Regular,Menlo,monospace; border-radius:7px;
         border:2px solid #cbd5e1; background:#fff; color:#0f172a; }
.wt-ch.diff { background:#fee2e2; border-color:#ef4444; color:#991b1b; }
.wt-ch.conf { background:#fef3c7; border-color:#f59e0b; color:#92400e; }
.wt-ch.gap  { background:repeating-linear-gradient(45deg,#f1f5f9,#f1f5f9 4px,#e2e8f0 4px,#e2e8f0 8px);
              border-style:dashed; color:#94a3b8; }
.wt-legend { font-size:.78rem; color:var(--wt-muted); margin-top:6px; }
.wt-legend span { display:inline-block; width:11px; height:11px; border-radius:3px; margin:0 4px -1px 10px; border:1px solid; }
[data-testid="stCheckbox"] label, [data-testid="stCheckbox"] label * {
    white-space:normal !important; overflow:visible !important; text-overflow:clip !important; }
@media (max-width: 640px) {
  .wt-diff { grid-template-columns:1fr; gap:4px; }
  .wt-row { gap:3px; margin-bottom:6px; }
  .wt-hero h1 { font-size:1.5rem; }
}
</style>
"""


@st.cache_resource(show_spinner="Loading EasyOCR model (first run downloads ~100 MB)…")
def get_reader() -> PlateReader:
    reader = PlateReader(model_dir=os.environ.get("WATCHTOWER_MODEL_DIR"))
    reader.warm_up()  # load the model once, up front
    return reader


def diff_html(cmp: Comparison) -> str:
    """Two aligned rows of character tiles with differences highlighted."""
    ai_row, alpr_row = [], []

    def tile(ch: str, cls: str = "") -> str:
        return f'<div class="wt-ch {cls}">{html.escape(ch) if ch else "·"}</div>'

    for seg in cmp.segments:
        if seg.op == EQUAL:
            ai_row += [tile(c) for c in seg.ai]
            alpr_row += [tile(c) for c in seg.alpr]
            continue
        cls = "conf" if seg.confusable else "diff"
        width = max(len(seg.ai), len(seg.alpr))
        a, b = seg.ai.ljust(width, "\0"), seg.alpr.ljust(width, "\0")
        for ca, cb in zip(a, b):
            ai_row.append(tile(ca.strip("\0"), cls if ca != "\0" else "gap"))
            alpr_row.append(tile(cb.strip("\0"), cls if cb != "\0" else "gap"))

    return (
        '<div class="wt-diff">'
        f'<div class="lbl">Watchtower AI</div><div class="wt-row">{"".join(ai_row)}</div>'
        f'<div class="lbl">ALPR system</div><div class="wt-row">{"".join(alpr_row)}</div>'
        "</div>"
        '<div class="wt-legend">'
        '<span style="background:#fff;border-color:#cbd5e1"></span>same'
        '<span style="background:#fee2e2;border-color:#ef4444"></span>different'
        '<span style="background:#fef3c7;border-color:#f59e0b"></span>commonly-confused pair'
        '<span style="background:#f1f5f9;border-color:#94a3b8"></span>missing character'
        "</div>"
    )


def render_verdict(result: VerificationResult) -> None:
    cls, icon, badge = OUTCOME_STYLE[result.outcome]
    st.markdown(
        f'<div class="wt-verdict {cls}"><span class="badge">{icon} {badge}</span>'
        f"<h3>{html.escape(result.outcome.title)}</h3>"
        f"<p>{html.escape(result.outcome.guidance)}</p></div>",
        unsafe_allow_html=True,
    )
    ai = result.ai_reading
    c1, c2, c3 = st.columns(3)
    c1.metric("Watchtower AI reading", ai.text if ai else "—")
    c2.metric("AI confidence", f"{ai.confidence:.0%}" if ai else "—",
              help=f"Readings below {result.min_confidence:.0%} are treated as inconclusive.")
    c3.metric("ALPR reading (simulated)", result.alpr_normalized)


def render_details(result: VerificationResult, image) -> None:
    st.subheader("Character comparison")
    if result.comparison is not None:
        st.markdown(diff_html(result.comparison), unsafe_allow_html=True)
        cmp = result.comparison
        st.caption(f"Similarity {cmp.similarity:.0%} · edit distance {cmp.edit_distance} · "
                   "comparison ignores case, spaces and dashes")
    else:
        st.info("No character comparison: the independent reading was missing or "
                "below the confidence threshold, so it cannot corroborate or dispute "
                "the ALPR reading.")

    st.subheader("Why this outcome")
    for reason in result.reasons:
        st.markdown(f"- {reason}")

    left, right = st.columns(2)
    with left:
        st.subheader("Reviewer checklist")
        st.caption("Watchtower never makes the decision. A person completes these steps.")
        for i, item in enumerate(REVIEW_CHECKLIST):
            st.checkbox(item, key=f"check_{i}_{st.session_state.get('result_key')}")
    with right:
        st.subheader("Image quality")
        q = result.quality
        st.caption(f"{q.width}×{q.height}px · sharpness {q.sharpness} · "
                   f"brightness {q.brightness} · contrast {q.contrast}")
        if q.warnings:
            for w in q.warnings:
                st.warning(w, icon="⚠️")
        else:
            st.success("No quality problems detected.", icon="✅")

    with st.expander("Technical details: what the AI looked at"):
        ai = result.ai_reading
        if ai is not None:
            st.image(draw_box(image, ai.box, ai.text),
                     caption=f"Region used for the independent reading ({ai.source} image pass)",
                     width="stretch")
        dets = ai.detections if ai else ()
        if dets:
            st.dataframe(
                [{"text": d.text, "confidence": round(d.confidence, 3), "box (x0,y0,x1,y1)": str(d.box)}
                 for d in dets],
                width="stretch", hide_index=True,
            )
        else:
            st.write("EasyOCR returned no text regions.")

    st.download_button(
        "Download review record (JSON)",
        data=json.dumps(result.to_record(), indent=2),
        file_name="watchtower_review_record.json",
        mime="application/json",
        help="Contains readings and scores only — no image and no personal data.",
    )


def main() -> None:
    st.set_page_config(page_title="Watchtower", page_icon="🗼", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="wt-hero"><div class="logo">🗼</div><div><h1>Watchtower</h1>'
        "<p>Independent AI verification of automated license plate reader (ALPR) results</p>"
        "</div></div>"
        '<div class="wt-notice"><b>Decision-support prototype.</b> Uses fictional plates and '
        "staged images only. Not connected to any camera, ALPR vendor or police database. "
        "Does not identify vehicle owners. Agreement between two AI systems is "
        "<b>not</b> proof of identity, and no outcome is an enforcement decision.</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Input")
        source = st.radio("Image source", ["Demo scenario", "Upload my own image"],
                          help="Demo scenarios use staged, fictional plates.")
        scenario = None
        if source == "Demo scenario":
            scenario = st.selectbox("Scenario", SCENARIOS, format_func=lambda s: s.label)
            st.caption(scenario.story)
        st.header("Settings")
        min_conf = st.slider(
            "Minimum AI confidence", 0.10, 0.95, DEFAULT_MIN_CONFIDENCE, 0.05,
            help="Independent readings below this confidence are reported as inconclusive.",
        )
        st.header("Safeguards")
        st.markdown(
            "- Images are processed in memory and never stored\n"
            "- No owner lookup, no database access\n"
            "- Every outcome requires human review\n"
            "- Discrepancies are surfaced, never auto-resolved"
        )

    img_col, result_col = st.columns([5, 6], gap="large")

    image_bytes: bytes | None = None
    with img_col:
        st.subheader("Evidence image")
        if scenario is not None:
            path = SAMPLES_DIR / scenario.image
            if path.exists():
                image_bytes = path.read_bytes()
            else:
                st.error(f"Demo image '{scenario.image}' is missing. "
                         "Run `python scripts/make_samples.py`.")
        else:
            upload = st.file_uploader(
                "Vehicle or license plate photo", type=["jpg", "jpeg", "png", "bmp", "webp"],
                help="Use staged or fictional plates only. Max 10 MB.",
            )
            if upload is not None:
                image_bytes = upload.getvalue()

        image = None
        if image_bytes is not None:
            try:
                image = load_image(image_bytes)
            except ImageError as exc:
                st.error(f"Cannot use this image: {exc}")
            else:
                st.image(image, caption="Original image", width="stretch")
        elif scenario is None:
            st.info("Upload a photo of a vehicle or license plate to begin.")

    with result_col:
        st.subheader("ALPR reading")
        alpr_text = st.text_input(
            "Simulated Flock/ALPR plate reading",
            value=scenario.alpr if scenario else "",
            key=f"alpr_{scenario.image if scenario else 'upload'}",
            placeholder="e.g. ABC 1234",
            help="Type what the (simulated) ALPR system reported. Case, spaces and dashes are ignored.",
        )
        run = st.button("Verify reading", type="primary", disabled=image is None,
                        width="stretch")

        key = None
        if image_bytes is not None:
            key = hashlib.sha256(image_bytes + alpr_text.encode() + str(min_conf).encode()).hexdigest()
        if st.session_state.get("result_key") != key:
            st.session_state.pop("result", None)  # inputs changed: stale result

        if run and image is not None:
            try:
                reader = get_reader()
                with st.spinner("Reading the plate independently…"):
                    st.session_state["result"] = verify(image, alpr_text, reader, min_conf)
                    st.session_state["result_key"] = key
            except InvalidReadingError as exc:
                st.error(str(exc))
            except Exception as exc:  # model download / runtime failures
                log.exception("Verification failed")
                st.error(f"Verification could not be completed: {exc}. "
                         "No outcome was produced — review the image manually.")

        result: VerificationResult | None = st.session_state.get("result")
        if result is not None:
            render_verdict(result)
        elif image is not None:
            st.caption("Press **Verify reading** to run an independent check.")

    if result is not None and image is not None:
        st.divider()
        render_details(result, image)


if __name__ == "__main__":
    main()
