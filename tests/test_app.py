"""Headless tests of the Streamlit interface, using Streamlit's built-in AppTest runner."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.config import LABELS, PROJECT_ROOT, RESULTS_ENV_VAR

APP_PATH = PROJECT_ROOT / "app.py"
VALID_TICKET = "I forgot my password and the reset email never arrives"


@pytest.fixture
def app(trained_results_dir, monkeypatch) -> AppTest:
    """A freshly started app session that uses the pre-trained temporary model."""
    monkeypatch.setenv(RESULTS_ENV_VAR, str(trained_results_dir))
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def submit(at: AppTest, text: str) -> AppTest:
    """Type a ticket and press Predict. Raises RuntimeError (not AssertionError) if the app itself crashes."""
    at.text_area(key="ticket_text").set_value(text)
    at.button(key="predict").click()
    at.run()
    if at.exception:
        raise RuntimeError(f"App raised: {[e.value for e in at.exception]}")
    return at


def shown_categories(at: AppTest) -> list[str]:
    return [metric.value for metric in at.metric if metric.label == "Predicted category"]


def test_app_starts_with_disclaimer_and_all_sections(app):
    assert any("not a clinical decision-making tool" in warning.value for warning in app.warning)
    assert len(app.tabs) == 5
    assert shown_categories(app) == []


def test_valid_ticket_shows_one_category_and_four_probabilities(app):
    submit(app, VALID_TICKET)
    assert not app.error
    categories = shown_categories(app)
    assert len(categories) == 1
    assert categories[0].lower() in LABELS
    assert len(app.get("progress")) == len(LABELS)


def test_empty_ticket_shows_an_error_and_no_result(app):
    submit(app, "")
    assert len(app.error) == 1
    assert "empty" in app.error[0].value
    assert shown_categories(app) == []


def test_over_long_ticket_shows_an_error(app):
    submit(app, "word " * 400)
    assert len(app.error) == 1
    assert "too long" in app.error[0].value


def test_sample_selector_fills_the_text_area(app):
    selector = app.selectbox(key="sample_choice")
    choice = selector.options[1]
    selector.select(choice).run()
    assert app.text_area(key="ticket_text").value
    assert app.text_area(key="ticket_text").value[:20] in choice


def test_evaluation_tab_shows_the_headline_metrics(app):
    labels = {metric.label for metric in app.metric}
    assert {"Accuracy", "Macro F1", "Macro precision", "Macro recall"} <= labels


def test_batch_tab_scores_the_demo_tickets(app):
    labels = {metric.label: metric.value for metric in app.metric}
    assert labels["Tickets"] == "24"
    assert labels["Rejected as invalid"] == "0"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Known defect C (see README): after a prediction error the previous successful "
    "result is still displayed as if it belonged to the latest input.",
)
def test_error_clears_the_previous_result(app):
    submit(app, VALID_TICKET)
    if len(shown_categories(app)) != 1:  # RuntimeError, so a broken setup is never mistaken for the known defect
        raise RuntimeError("precondition failed: the first prediction did not render")
    submit(app, "")
    if len(app.error) != 1:
        raise RuntimeError("precondition failed: the error message did not render")
    assert shown_categories(app) == [], "a stale result is still on screen after the error"
