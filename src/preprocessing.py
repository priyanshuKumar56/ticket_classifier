"""Text preprocessing applied to every ticket before TF-IDF vectorisation.

The function is a module-level callable (not a lambda) so a pipeline that
references it can be pickled and loaded again by joblib.
"""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lower-case a ticket and mask volatile tokens.

    URLs, e-mail addresses and plain numbers (error codes, invoice numbers,
    amounts) are replaced by placeholder words so the model learns that
    "a number was mentioned" rather than memorising a specific value.
    Whitespace is collapsed and trimmed.

    >>> normalize_text("Error 500 on  https://x.example/report")
    'error numtoken on urltoken'
    """
    text = text.lower()
    text = _URL_RE.sub(" urltoken ", text)
    text = _EMAIL_RE.sub(" emailtoken ", text)
    text = _NUMBER_RE.sub(" numtoken ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()
