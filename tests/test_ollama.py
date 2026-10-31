"""Ollama research desk — status is always safe; brief needs a live daemon."""

from gig.nlp.ollama import default_prompts, ollama_status, research_brief


def test_ollama_status_never_raises():
    status = ollama_status()
    assert "available" in status
    assert "host" in status
    assert "model" in status
    if not status["available"]:
        assert "hint" in status


def test_default_prompts_are_nonempty():
    prompts = default_prompts()
    assert len(prompts) >= 3
    assert all(isinstance(p, str) and p for p in prompts)


def test_research_brief_reports_offline_cleanly():
    out = research_brief("Summarize the book.")
    assert "available" in out
    assert "question" in out
    # Desk always answers — Ollama when up, structured snapshot otherwise.
    assert out["available"] is True
    assert out.get("answer")


def test_structured_brief_mentions_pipeline():
    from gig.nlp.ollama import structured_brief

    text = structured_brief(
        "Walk me through the math pipeline",
        {
            "cloud": {
                "asof": "2026-09-04",
                "construction": "risk_constrained",
                "names": 100,
                "book": {"gross": 2.0, "net": 0.0, "longs": 40, "shorts": 40},
                "risk": {
                    "available": True,
                    "equation": "Σ = BB′ + D",
                    "ex_ante_vol": 0.1,
                    "realized_vol": 0.12,
                    "systematic_share": 0.05,
                    "effective_names": 50,
                    "eigenvalue_share": [0.4, 0.25, 0.15],
                },
                "pipeline": [
                    {"label": "Factors", "detail": "mom · rev · ivol"},
                    {"label": "Σ = BB′ + D", "detail": "5 PCs"},
                ],
                "breaches": [],
            },
            "research": {"available": False},
        },
    )
    assert "Signal path" in text
    assert "Σ = BB′ + D" in text
