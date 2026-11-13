"""Smoke test: the Streamlit page renders without errors (no OCR model needed)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_renders_demo_scenario():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Watchtower" in m.value for m in at.markdown)
    assert at.text_input[0].value == "BXR 4827"  # first demo scenario pre-filled
    assert at.button[0].label == "Verify reading"
    assert not at.button[0].disabled


def test_upload_mode_waits_for_image():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio[0].set_value("Upload my own image").run()
    assert not at.exception
    assert at.button[0].disabled  # nothing to verify yet
