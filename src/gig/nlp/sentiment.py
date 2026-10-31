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


def transformer_sentiment(text: str) -> float | None:
    """Optional ProsusAI/finbert. Returns None if transformers/torch are missing."""
    try:
        from transformers import pipeline
    except Exception:
        return None
    nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    out = nlp(text[:512])[0]
    label = str(out["label"]).lower()
    score = float(out["score"])
    if "pos" in label:
        return score
    if "neg" in label:
        return -score
    return 0.0
