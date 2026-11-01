"""Score headlines. Lexicon always; VADER / FinBERT if installed — never required."""

from __future__ import annotations

from gig.nlp.lexicon import lexicon_score


def headline_sentiment(text: str) -> float:
    """
    Point-in-time safe: scoring uses only the headline string, no future prices.

    Order: VADER (if installed) → lexicon. Transformers are opt-in via
    `transformer_sentiment` so CI never downloads a 400MB model.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        vs = SentimentIntensityAnalyzer().polarity_scores(text or "")
        return float(vs["compound"])
    except Exception:
        return lexicon_score(text)


def transformer_sentiment(text: str, *, local_directory: str | None = None) -> float | None:
    """Explicit, local-only FinBERT inference; absence is never a neutral score.

    Without a caller-selected local checkpoint return ``None`` without importing
    Transformers. A supplied checkpoint must satisfy ``LocalFinBERT``'s manifest
    and safetensors contract; load/inference failures propagate to the caller.
    This helper never downloads weights or silently substitutes another model.
    """
    if local_directory is None:
        return None
    from gig.nlp.workflow import LocalFinBERT

    return float(LocalFinBERT(local_directory).score(text)["score"])
