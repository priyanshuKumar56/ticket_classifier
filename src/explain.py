"""Explain a prediction by showing which terms pushed the score up or down."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.pipeline import Pipeline

from src.validation import validate_ticket_text


@dataclass(frozen=True)
class TermContribution:
    """One term's contribution to a category's linear score (logit).

    ``contribution = tfidf_weight_in_this_ticket * coefficient_for_category``.
    Positive values support the category; negative values count against it.
    """

    term: str
    contribution: float


def explain_prediction(
    pipeline: Pipeline, text: object, label: str | None = None, top_k: int = 8
) -> list[TermContribution]:
    """Return the ``top_k`` terms with the largest absolute effect on ``label``.

    If ``label`` is omitted, the model's predicted category is explained.
    Only terms known to the vectoriser can contribute, so a ticket made
    entirely of unknown words returns an empty list.
    """
    cleaned = validate_ticket_text(text)
    vectorizer, classifier = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    features = vectorizer.transform([cleaned])
    classes = [str(c) for c in classifier.classes_]
    if label is None:
        label = classes[int(classifier.predict_proba(features)[0].argmax())]
    if label not in classes:
        raise ValueError(f"Unknown category {label!r}. Known categories: {classes}.")

    class_index = classes.index(label)
    names = vectorizer.get_feature_names_out()
    coo = features.tocoo()
    contributions = [
        TermContribution(str(names[col]), float(weight * classifier.coef_[class_index, col]))
        for col, weight in zip(coo.col, coo.data)
    ]
    contributions.sort(key=lambda item: abs(item.contribution), reverse=True)
    return contributions[:top_k]
