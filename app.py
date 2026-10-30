"""Streamlit interface for the Clinic Support Ticket Classifier.

Run with:  streamlit run app.py

The app trains a model automatically on first launch if ``results/`` has none.
Presentation only: prediction, metrics and explanations come from ``src``.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.artifacts import ModelArtifactError, ModelNotFoundError, load_model
from src.config import (
    DATA_PATH,
    DEMO_BATCH_PATH,
    EVALUATION_FILENAME,
    LABELS,
    MAX_BATCH_ROWS,
    MAX_TICKET_CHARS,
    MISCLASSIFIED_FILENAME,
    REVIEW_MIN_MARGIN,
    REVIEW_MIN_PROBABILITY,
    TEST_PREDICTIONS_FILENAME,
    results_dir,
)
from src.evaluation import plot_confusion_matrix
from src.explain import explain_prediction
from src.model import predict_dataframe, predict_ticket
from src.training import run_training
from src.validation import InvalidTicketTextError, ValidationError

DISCLAIMER = (
    "Educational prototype - not a clinical decision-making tool. It sorts fictional software "
    "support tickets and must never be used for patient care, triage of patients, or any medical decision."
)
SAMPLE_PROMPT = "Choose a sample ticket..."
LABEL_GUIDE: dict[str, str] = {
    "appointment": "Scheduling: bookings, reschedules, cancellations, reminders, slots and no-shows.",
    "billing": "Money: invoices, charges, refunds, fees, insurance claims and subscription price.",
    "login": "Access: passwords, verification codes, locked accounts, SSO, invitations and permissions.",
    "technical": "Faults: crashes, error codes, slowness, blank screens, sync failures, export and print.",
}
LABEL_COLOR = {
    "appointment": "#0F766E",
    "billing": "#B45309",
    "login": "#4338CA",
    "technical": "#BE123C",
}
STYLE = """
<style>
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1180px; }
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #E2EAE7;
    border-radius: 14px;
    padding: 0.65rem 0.85rem 0.45rem;
}
[data-testid="stMetricValue"] {
    font-family: Georgia, "Iowan Old Style", Palatino, serif;
    color: #0F3D3A;
}
[data-testid="stTabs"] button { font-weight: 600; }
.hero {
    background: linear-gradient(135deg, #0F3D3A 0%, #14635C 58%, #1F8A78 100%);
    color: #F4FBF8;
    border-radius: 18px;
    padding: 1.45rem 1.6rem 1.25rem;
    margin-bottom: 0.85rem;
}
.hero h1 {
    font-family: Georgia, "Iowan Old Style", Palatino, serif;
    font-size: 2.05rem;
    line-height: 1.15;
    font-weight: 600;
    letter-spacing: -0.02em;
    margin: 0.2rem 0 0.45rem;
    color: #ffffff;
}
.hero p { margin: 0; color: #D7EFE8; font-size: 1.02rem; max-width: 46rem; line-height: 1.45; }
.eyebrow {
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-size: 0.72rem;
    font-weight: 700;
    color: #B7E4D6;
}
.pill-row { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-top: 0.95rem; }
.pill {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 999px;
    padding: 0.16rem 0.62rem;
    font-size: 0.8rem;
    color: #F4FBF8;
}
.guide { display: grid; grid-template-columns: 1fr 1fr; gap: 0.55rem; }
.guide-card {
    background: #ffffff;
    border: 1px solid #E2EAE7;
    border-radius: 12px;
    padding: 0.7rem 0.75rem 0.65rem;
    border-top: 3px solid var(--accent, #0F6E6B);
}
.guide-card strong { display: block; margin-bottom: 0.18rem; color: #14221F; }
.guide-card p { margin: 0; color: #4E615C; font-size: 0.86rem; line-height: 1.35; }
.ticket-quote {
    background: #ffffff;
    border-left: 4px solid #0F6E6B;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin: 0.2rem 0 0.85rem;
    color: #14221F;
    line-height: 1.45;
}
.section-label {
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
    color: #5C6F6B;
    margin: 0.2rem 0 0.35rem;
}
@media (max-width: 700px) {
    .guide { grid-template-columns: 1fr; }
    .hero h1 { font-size: 1.6rem; }
}
</style>
"""


# --------------------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading model (training one first if none exists)...")
def load_bundle(directory: str) -> dict[str, Any]:
    """Load the model and saved evaluation files, training once if nothing exists yet."""
    path = Path(directory)
    try:
        pipeline, metadata = load_model(path)
    except ModelNotFoundError:
        run_training(output_dir=path)
        pipeline, metadata = load_model(path)

    evaluation_file = path / EVALUATION_FILENAME
    misclassified_file = path / MISCLASSIFIED_FILENAME
    predictions_file = path / TEST_PREDICTIONS_FILENAME
    return {
        "pipeline": pipeline,
        "metadata": metadata,
        "evaluation": json.loads(evaluation_file.read_text(encoding="utf-8")) if evaluation_file.is_file() else None,
        "misclassified": pd.read_csv(misclassified_file) if misclassified_file.is_file() else None,
        "test_predictions": pd.read_csv(predictions_file) if predictions_file.is_file() else None,
    }


@st.cache_data
def load_samples() -> dict[str, str]:
    """Map a readable label to each unlabelled demo ticket."""
    if not DEMO_BATCH_PATH.is_file():
        return {}
    frame = pd.read_csv(DEMO_BATCH_PATH, dtype=str).dropna()
    return {f"{row.ticket_id}: {row.text[:72]}": row.text for row in frame.itertuples()}


@st.cache_data
def load_labelled_tickets() -> pd.DataFrame:
    """Return the hand-written labelled tickets for the library view."""
    if not DATA_PATH.is_file():
        return pd.DataFrame(columns=["ticket_id", "text", "category"])
    return pd.read_csv(DATA_PATH, dtype=str)


def _apply_sample() -> None:
    choice = st.session_state.get("sample_choice")
    samples = load_samples()
    if choice in samples:
        st.session_state["ticket_text"] = samples[choice]


def _label_guide() -> None:
    cards = []
    for label in LABELS:
        color = LABEL_COLOR[label]
        cards.append(
            f'<div class="guide-card" style="--accent:{color}">'
            f"<strong>{html.escape(label.title())}</strong>"
            f"<p>{html.escape(LABEL_GUIDE[label])}</p></div>"
        )
    st.markdown(f'<div class="guide">{"".join(cards)}</div>', unsafe_allow_html=True)


def _percent_columns(frame: pd.DataFrame) -> dict[str, Any]:
    """Progress bars for probability columns. Values stay on the 0–1 scale."""
    config: dict[str, Any] = {}
    if "text" in frame.columns:
        config["text"] = st.column_config.TextColumn("Ticket text", width="large")
    if "ticket_id" in frame.columns:
        config["ticket_id"] = st.column_config.TextColumn("Ticket ID")
    if "true_category" in frame.columns:
        config["true_category"] = st.column_config.TextColumn("True category")
    if "predicted_category" in frame.columns:
        config["predicted_category"] = st.column_config.TextColumn("Predicted")
    if "confidence" in frame.columns:
        config["confidence"] = st.column_config.NumberColumn("Top probability", format="percent")
    if "margin" in frame.columns:
        config["margin"] = st.column_config.NumberColumn("Margin", format="percent")
    if "needs_review" in frame.columns:
        config["needs_review"] = st.column_config.CheckboxColumn("Needs review")
    if "review_reason" in frame.columns:
        config["review_reason"] = st.column_config.TextColumn("Review reason", width="medium")
    if "correct" in frame.columns:
        config["correct"] = st.column_config.CheckboxColumn("Correct")
    if "error" in frame.columns:
        config["error"] = st.column_config.TextColumn("Error", width="medium")
    for label in LABELS:
        column = f"prob_{label}"
        if column in frame.columns:
            config[column] = st.column_config.ProgressColumn(
                label.title(), min_value=0.0, max_value=1.0, format="percent"
            )
    return config


def _option_for_row(frame: pd.DataFrame, index: int, row: pd.Series) -> str:
    if "ticket_id" in frame.columns and pd.notna(row["ticket_id"]):
        ident = str(row["ticket_id"])
    else:
        ident = f"Row {index + 1}"
    preview = str(row.get("text", "")).replace("\n", " ")[:68]
    predicted = row.get("predicted_category") or "invalid"
    return f"{ident}  ·  {predicted}  ·  {preview}"


# ---------------------------------------------------------------------------- tabs
def render_result(result: dict[str, Any]) -> None:
    """Show the predicted category, all four probability estimates and an explanation."""
    prediction = result["prediction"]
    ranked = sorted(prediction.probabilities.items(), key=lambda item: -item[1])
    runner_up = ranked[1][0].title() if len(ranked) > 1 else "—"
    quoted = html.escape(str(result.get("text", ""))).replace("\n", "<br>")

    st.markdown('<div class="section-label">Result for this ticket</div>', unsafe_allow_html=True)
    if quoted:
        st.markdown(f'<div class="ticket-quote">{quoted}</div>', unsafe_allow_html=True)

    left, right = st.columns([2, 3])
    with left:
        st.metric("Predicted category", prediction.label.title())
        st.caption(
            f"Estimated probability {prediction.confidence:.0%}. These are model estimates, "
            "not guaranteed calibrated confidence."
        )
        if prediction.needs_review:
            st.warning("Route to a person for review: " + "; ".join(prediction.review_reasons) + ".")
        else:
            st.caption("Review flag: not raised. A person should still read anything that will be acted on.")
    with right:
        for label, value in ranked:
            st.progress(min(max(value, 0.0), 1.0), text=f"{label.title()}  {value:.1%}")

    first, second, third, fourth = st.columns(4)
    first.metric("Top probability", f"{prediction.confidence:.1%}")
    second.metric("Margin to runner-up", f"{prediction.margin:.1%}")
    third.metric("Runner-up", runner_up)
    fourth.metric("Known terms", int(prediction.known_terms))

    st.subheader("Why this prediction?")
    terms = result["terms"]
    if terms:
        frame = pd.DataFrame(
            {
                "term": [t.term for t in terms],
                "contribution": [round(t.contribution, 3) for t in terms],
                "direction": [
                    "supports" if t.contribution > 0 else "counts against" if t.contribution < 0 else "neutral"
                    for t in terms
                ],
            }
        )
        chart_col, table_col = st.columns([3, 2])
        with chart_col:
            st.bar_chart(frame.set_index("term")["contribution"], height=260)
        with table_col:
            st.dataframe(frame, hide_index=True, width="stretch")
        st.caption(
            "Each value is this ticket's TF-IDF weight times the linear weight for the predicted category. "
            "Positive supports that category. Negative counts against it. This explains the linear model only, "
            "not a cause in the real world."
        )
    else:
        st.caption("None of the words in this ticket were seen during training, so there is nothing to explain.")


def classify_tab(pipeline: Any) -> None:
    st.caption(
        "Paste one fictional support ticket. The model proposes one of four categories, shows every "
        "probability, and flags uncertain tickets for a person."
    )
    form_col, guide_col = st.columns([3, 2], gap="large")
    with form_col:
        st.session_state.setdefault("ticket_text", "")
        st.selectbox(
            "Sample tickets (fictional)",
            [SAMPLE_PROMPT, *load_samples()],
            key="sample_choice",
            on_change=_apply_sample,
        )
        st.text_area(
            "Support ticket",
            key="ticket_text",
            height=150,
            placeholder="Describe the software problem in a sentence or two, e.g. 'The reminder emails are not being sent.'",
            help=f"Up to {MAX_TICKET_CHARS} characters. Do not enter real patient information.",
        )
        used = len(st.session_state.get("ticket_text") or "")
        st.caption(f"{used} / {MAX_TICKET_CHARS} characters. Do not enter real patient information.")
        predict_clicked = st.button("Predict Category", key="predict", type="primary")
    with guide_col:
        st.markdown('<div class="section-label">What the four categories mean</div>', unsafe_allow_html=True)
        _label_guide()
        st.caption("Label the primary thing the sender needs fixed, not whichever keyword appears first.")

    if predict_clicked:
        try:
            text = st.session_state["ticket_text"]
            prediction = predict_ticket(pipeline, text)
            st.session_state["result"] = {
                "prediction": prediction,
                "terms": explain_prediction(pipeline, text, prediction.label),
                "text": text,
            }
        except InvalidTicketTextError as exc:
            st.error(str(exc))

    result = st.session_state.get("result")
    if result is not None:
        render_result(result)
    else:
        st.info("Submit a ticket to see the category, all four probabilities, the review flag and the words that drove the score.")


def _inspect_frame(frame: pd.DataFrame, key: str, title: str) -> None:
    """Show every field of one selected row."""
    if frame.empty:
        return
    options = [_option_for_row(frame, index, row) for index, row in frame.iterrows()]
    choice = st.selectbox(title, options, key=key)
    row = frame.iloc[options.index(choice)]
    st.markdown("**Full ticket**")
    st.write(str(row.get("text", "")))

    summary = st.columns(4)
    if "true_category" in frame.columns:
        summary[0].metric("True category", str(row["true_category"]).title())
    summary[1].metric("Model prediction", str(row.get("predicted_category") or "—").title())
    confidence = row.get("confidence")
    summary[2].metric("Top probability", "—" if pd.isna(confidence) else f"{float(confidence):.1%}")
    if "needs_review" in frame.columns and pd.notna(row["needs_review"]):
        summary[3].metric("Human review", "Needed" if bool(row["needs_review"]) else "Not flagged")

    if "review_reason" in frame.columns and pd.notna(row["review_reason"]) and str(row["review_reason"]).strip():
        st.caption(str(row["review_reason"]))
    if "error" in frame.columns and pd.notna(row["error"]):
        st.error(str(row["error"]))

    prob_cols = [column for column in frame.columns if str(column).startswith("prob_")]
    if prob_cols:
        probabilities = pd.DataFrame(
            {
                "category": [str(column).removeprefix("prob_").title() for column in prob_cols],
                "probability": [None if pd.isna(row[column]) else float(row[column]) for column in prob_cols],
            }
        ).sort_values("probability", ascending=False, na_position="last")
        st.dataframe(
            probabilities,
            hide_index=True,
            width="stretch",
            column_config={"probability": st.column_config.NumberColumn("Probability", format="percent")},
        )


def batch_tab(pipeline: Any) -> None:
    st.caption(
        "Classify many tickets at once. Use the bundled demo batch, or upload a CSV with a `text` column "
        f"(and optionally `ticket_id`), up to {MAX_BATCH_ROWS} rows."
    )
    source = st.radio(
        "Source",
        ["Demo batch (24 unlabelled tickets)", "Upload a CSV"],
        horizontal=True,
        key="batch_source",
    )
    frame: pd.DataFrame | None = None
    if source.startswith("Demo"):
        frame = pd.read_csv(DEMO_BATCH_PATH, dtype=str)
    else:
        upload = st.file_uploader("CSV file", type=["csv"], key="batch_upload")
        if upload is not None:
            try:
                frame = pd.read_csv(upload, dtype=str)
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
                st.error(f"Could not read that file as a CSV: {exc}")
    if frame is None:
        return
    if len(frame) > MAX_BATCH_ROWS:
        st.error(f"That file has {len(frame)} rows; the limit is {MAX_BATCH_ROWS}.")
        return
    try:
        result = predict_dataframe(pipeline, frame)
    except ValidationError as exc:
        st.error(str(exc))
        return

    first, second, third = st.columns(3)
    first.metric("Tickets", len(result))
    second.metric("Flagged for review", int(result["needs_review"].sum()))
    third.metric("Rejected as invalid", int(result["error"].notna().sum()))
    counts = result["predicted_category"].value_counts().reindex(list(LABELS), fill_value=0)
    st.bar_chart(counts, height=220)

    view = st.selectbox("Show", ["All tickets", "Needs review", "Not flagged"], key="batch_view")
    shown = result
    if view == "Needs review":
        shown = result[result["needs_review"].astype(bool)]
    elif view == "Not flagged":
        shown = result[~result["needs_review"].astype(bool)]
    st.caption(f"Showing {len(shown)} of {len(result)} tickets. Every probability column is included.")
    st.dataframe(shown, hide_index=True, width="stretch", height=380, column_config=_percent_columns(shown))
    _inspect_frame(shown.reset_index(drop=True), key="batch_inspect", title="Open one batch ticket")
    st.download_button(
        "Download all predictions (CSV)",
        result.to_csv(index=False).encode("utf-8"),
        file_name="batch_predictions.csv",
        mime="text/csv",
    )


def evaluation_tab(bundle: dict[str, Any]) -> None:
    evaluation = bundle["evaluation"]
    if evaluation is None:
        st.info("No saved evaluation found. Run `python train.py` to create one.")
        return
    metrics, baseline = evaluation["metrics"], evaluation["baseline"]["metrics"]
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
    cols[1].metric("Macro F1", f"{metrics['macro_f1']:.3f}")
    cols[2].metric("Macro precision", f"{metrics['macro_precision']:.3f}")
    cols[3].metric("Macro recall", f"{metrics['macro_recall']:.3f}")
    cv = evaluation["cross_validation"]
    fold_scores = cv.get("scores") or []
    fold_text = ", ".join(f"{score:.3f}" for score in fold_scores)
    st.caption(
        f"Held-out test set: {evaluation['n_test']} tickets, evaluated once. Always guessing "
        f"'{evaluation['baseline']['predicted_category']}' scores accuracy {baseline['accuracy']:.3f} and macro F1 "
        f"{baseline['macro_f1']:.3f}. Cross-validated macro F1 on the training split: "
        f"{cv['mean']:.3f} +/- {cv['std']:.3f} ({cv['folds']} folds)."
        + (f" Fold scores: {fold_text}." if fold_text else "")
    )

    per_class = metrics.get("per_class") or {}
    if per_class:
        score_frame = pd.DataFrame(per_class).T.reindex([label for label in LABELS if label in per_class])
        st.subheader("Per-category precision, recall and F1")
        st.bar_chart(score_frame[["precision", "recall", "f1"]], height=260)

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        matrix = evaluation["confusion_matrix"]
        st.pyplot(plot_confusion_matrix(matrix["matrix"], matrix["labels"]), width="stretch")
        st.caption("Rows are the true category. Columns are the prediction. Diagonal cells are correct.")
    with right:
        st.subheader("Score table")
        report = {name: values for name, values in evaluation["classification_report"].items() if isinstance(values, dict)}
        table = pd.DataFrame(report).T.rename(columns={"f1-score": "f1"})
        table["support"] = table["support"].astype(int)
        st.dataframe(table.round(3), width="stretch")

    tuning = evaluation.get("tuning")
    if tuning:
        st.subheader("Hyper-parameter search")
        st.json(tuning)

    with st.expander("How to read these numbers"):
        st.markdown(
            "- **Accuracy**: share of test tickets labelled correctly. Misleading when categories are unbalanced.\n"
            "- **Precision**: of tickets predicted as a category, how many truly belong to it.\n"
            "- **Recall**: of tickets that truly belong to a category, how many were found.\n"
            "- **Macro F1**: F1 averaged over the four categories with equal weight, so a weak small category is not hidden.\n"
            "- **Confusion matrix**: rows are the true category, columns the prediction. Off-diagonal cells show which "
            "categories get mixed up.\n"
            "- With only a few dozen test tickets, one ticket moves accuracy by several points. Treat every number as rough."
        )
    triage = evaluation.get("triage")
    if triage:
        with st.expander("Human-review flag on the test set"):
            flagged, kept = triage["accuracy_flagged"], triage["accuracy_not_flagged"]
            st.write(
                f"{triage['n_flagged']} of {evaluation['n_test']} test tickets were flagged for review. "
                f"Accuracy when flagged: {'n/a' if flagged is None else f'{flagged:.0%}'}; "
                f"when not flagged: {'n/a' if kept is None else f'{kept:.0%}'}."
            )
            correct_confidence = triage.get("mean_confidence_correct")
            incorrect_confidence = triage.get("mean_confidence_incorrect")
            if correct_confidence is not None or incorrect_confidence is not None:
                st.caption(
                    "Mean top probability when correct: "
                    + ("n/a" if correct_confidence is None else f"{correct_confidence:.0%}")
                    + ". When incorrect: "
                    + ("n/a" if incorrect_confidence is None else f"{incorrect_confidence:.0%}")
                    + "."
                )

    predictions = bundle.get("test_predictions")
    if predictions is not None:
        st.subheader("Every held-out test ticket")
        st.caption(f"{len(predictions)} tickets scored once after training. This split is not used to fit the model.")
        st.dataframe(
            predictions,
            hide_index=True,
            width="stretch",
            height=360,
            column_config=_percent_columns(predictions),
        )
        _inspect_frame(predictions, key="test_inspect", title="Open one test ticket")


def misclassified_tab(bundle: dict[str, Any]) -> None:
    wrong = bundle["misclassified"]
    if wrong is None:
        st.info("No misclassification file found. Run `python train.py` to create one.")
        return
    if wrong.empty:
        st.success("No test tickets were misclassified in this split. With so few tickets, that proves little.")
        return
    st.write(f"{len(wrong)} test tickets were misclassified, most confidently wrong first.")
    filter_col, search_col = st.columns([1, 2])
    with filter_col:
        chosen = st.multiselect("Filter by true category", list(LABELS), key="mis_filter")
    with search_col:
        query = st.text_input("Search text or ticket ID", key="mis_query", placeholder="password, invoice, T-014")
    shown = wrong[wrong["true_category"].isin(chosen)] if chosen else wrong
    if query.strip():
        needle = query.strip()
        text_match = shown["text"].astype(str).str.contains(needle, case=False, na=False)
        id_match = shown["ticket_id"].astype(str).str.contains(needle, case=False, na=False) if "ticket_id" in shown.columns else False
        shown = shown[text_match | id_match]
    st.caption(f"{len(shown)} of {len(wrong)} misclassified tickets.")
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        height=340,
        column_config=_percent_columns(shown),
    )
    if shown.empty:
        st.info("No misclassified tickets match this filter.")
    else:
        _inspect_frame(shown.reset_index(drop=True), key="mis_inspect", title="Open one misclassified ticket")
    pairs = wrong.groupby(["true_category", "predicted_category"]).size().sort_values(ascending=False)
    st.caption("Most common mix-ups: " + "; ".join(f"{true} → {pred} ({count})" for (true, pred), count in pairs.head(3).items()))


def about_tab(bundle: dict[str, Any]) -> None:
    meta = bundle["metadata"]
    hyper = meta.get("hyperparameters") or {}
    st.markdown(
        """
**What this is.** A local teaching project. TF-IDF features and Logistic Regression sort short, fictional
software-support tickets into appointment, billing, login or technical.

**Probabilities** come from `predict_proba`. They are estimates, not guaranteed calibrated confidence.
The review flag is a simple rule: low top probability, a small gap to the runner-up, or no known words.

**Never** enter real patient information, and never use this for clinical decisions.
"""
    )
    _label_guide()

    st.subheader("This model")
    ngram = hyper.get("ngram_range")
    rows = [
        ("Dataset file", meta.get("dataset_file", "—")),
        ("Labelled tickets", meta.get("n_examples", "—")),
        ("Training tickets", meta.get("n_train", "—")),
        ("Test tickets", meta.get("n_test", "—")),
        ("Seed", meta.get("seed", "—")),
        ("Test fraction", meta.get("test_size", "—")),
        ("N-gram range", "—" if ngram is None else f"{ngram[0]}–{ngram[1]} words"),
        ("Logistic regression C", hyper.get("C", "—")),
        ("Tuned on training data", "yes" if meta.get("tuned") else "no"),
        ("Saved (UTC)", meta.get("saved_at_utc", "—")),
        ("Python", meta.get("python_version", "—")),
        ("scikit-learn", meta.get("sklearn_version", "—")),
    ]
    details = pd.DataFrame(rows, columns=["Detail", "Value"])
    details["Value"] = details["Value"].map(lambda value: "—" if value is None else str(value))
    st.dataframe(details, hide_index=True, width="stretch")
    fingerprint = meta.get("dataset_sha256")
    if fingerprint:
        st.caption("Dataset fingerprint (SHA-256 of the labelled rows)")
        st.code(str(fingerprint), language=None)
    test_ids = meta.get("test_ticket_ids") or []
    if test_ids:
        with st.expander(f"Test ticket IDs ({len(test_ids)})"):
            st.write(", ".join(str(ticket_id) for ticket_id in test_ids))

    st.subheader("Labelled ticket library")
    library = load_labelled_tickets()
    if library.empty:
        st.info("No labelled dataset file was found.")
        return
    counts = library["category"].str.lower().value_counts().reindex(list(LABELS), fill_value=0)
    st.bar_chart(counts, height=220)
    library_filter, library_search = st.columns([1, 2])
    with library_filter:
        categories = st.multiselect("Categories", list(LABELS), key="library_categories")
    with library_search:
        query = st.text_input("Search ticket text or ID", key="library_query", placeholder="reminder, T-001")
    shown = library
    if categories and "category" in shown.columns:
        shown = shown[shown["category"].str.lower().isin(categories)]
    if query.strip() and {"text", "ticket_id"} <= set(shown.columns):
        needle = query.strip()
        shown = shown[
            shown["text"].astype(str).str.contains(needle, case=False, na=False)
            | shown["ticket_id"].astype(str).str.contains(needle, case=False, na=False)
        ]
    st.caption(f"{len(shown)} of {len(library)} labelled tickets.")
    st.dataframe(shown, hide_index=True, width="stretch", height=420)


def _render_header(meta: dict[str, Any]) -> None:
    pills = [
        f"{meta.get('n_examples', '—')} labelled tickets",
        f"{meta.get('n_train', '—')} train",
        f"{meta.get('n_test', '—')} test",
        *[label.title() for label in LABELS],
    ]
    pill_html = "".join(f'<span class="pill">{html.escape(str(pill))}</span>' for pill in pills)
    st.markdown(
        f"""
<div class="hero">
  <div class="eyebrow">Local teaching model</div>
  <h1>Clinic Support Ticket Classifier</h1>
  <p>Sort fictional clinic-software support tickets, see every probability, and flag the ones a person should read.</p>
  <div class="pill-row">{pill_html}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------- main
def main() -> None:
    st.set_page_config(page_title="Clinic Support Ticket Classifier", page_icon="📋", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)

    try:
        bundle = load_bundle(str(results_dir()))
    except (ModelArtifactError, ValidationError) as exc:
        st.error(f"Could not load or train the model: {exc}")
        st.stop()

    meta = bundle["metadata"]
    _render_header(meta)
    st.warning(DISCLAIMER)

    with st.sidebar:
        st.header("Model")
        st.write(f"Trained on {meta['n_train']} tickets; tested on {meta['n_test']}.")
        st.write(f"Seed {meta['seed']}, scikit-learn {meta['sklearn_version']}.")
        hyper = meta.get("hyperparameters") or {}
        if hyper:
            ngram = hyper.get("ngram_range")
            st.caption(
                "TF-IDF n-grams "
                + ("—" if not ngram else f"{ngram[0]}–{ngram[1]}")
                + f", logistic regression C = {hyper.get('C', '—')}."
            )
        st.caption(f"Saved {meta['saved_at_utc']} (UTC)")
        st.divider()
        st.subheader("Review rule")
        st.caption(
            f"A ticket is flagged when the top probability is under {REVIEW_MIN_PROBABILITY:.0%}, "
            f"the top two categories are within {REVIEW_MIN_MARGIN:.0%}, "
            "or none of its words were in the training vocabulary."
        )
        st.divider()
        st.subheader("Categories")
        for label in LABELS:
            st.markdown(f"**{label.title()}**  \n{LABEL_GUIDE[label]}")
        if st.button("Reload model files", key="reload"):
            st.cache_resource.clear()
            st.rerun()

    classify, batch, evaluation, misclassified, about = st.tabs(
        ["Classify", "Batch", "Evaluation", "Misclassifications", "About"]
    )
    with classify:
        classify_tab(bundle["pipeline"])
    with batch:
        batch_tab(bundle["pipeline"])
    with evaluation:
        evaluation_tab(bundle)
    with misclassified:
        misclassified_tab(bundle)
    with about:
        about_tab(bundle)


main()
