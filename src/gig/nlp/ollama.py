"""
Local LLM research desk via Ollama — with a structured fallback.

This module answers questions *about* the book, the risk report, and the last
backtest. It does not place orders and it does not emit buy/sell signals. An
LLM that outputs trades is excluded by model policy; this is a briefing layer
on top of numbers the research path already computed.

When Ollama is offline, ``structured_brief`` still answers from the same
snapshot JSON so the desk never goes silent.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from gig.config import get_settings

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2"


def _host() -> str:
    settings = get_settings()
    return (getattr(settings, "ollama_host", None) or DEFAULT_HOST).rstrip("/")


def _model() -> str:
    settings = get_settings()
    return getattr(settings, "ollama_model", None) or DEFAULT_MODEL


def _get(path: str, timeout: float = 3.0) -> Any:
    req = urllib.request.Request(_host() + path, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post(path: str, payload: dict[str, Any], timeout: float = 120.0) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _host() + path,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_status() -> dict[str, Any]:
    """Reachability and installed models. Never raises."""
    try:
        tags = _get("/api/tags", timeout=2.0)
        models = [m.get("name", "") for m in (tags.get("models") or [])]
        return {
            "available": True,
            "host": _host(),
            "model": _model(),
            "models": models,
            "model_ready": any(_model().split(":")[0] in m for m in models) if models else False,
            "desk": "ollama",
        }
    except Exception as exc:
        return {
            "available": False,
            "host": _host(),
            "model": _model(),
            "models": [],
            "model_ready": False,
            "desk": "structured",
            "hint": (
                "Install Ollama from https://ollama.com, start it, then: ollama pull "
                + _model()
                + " — until then the desk uses a structured snapshot briefing."
            ),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _snapshot_context() -> dict[str, Any]:
    from gig.service import snapshot

    payload: dict[str, Any] = {
        "status": snapshot.service_status(),
        "research": snapshot.research_snapshot(),
        "cloud": {},
    }
    try:
        cloud = snapshot._cached("cloud:brief", 90.0, lambda: snapshot.factor_cloud(limit=120))
        if cloud.get("available"):
            payload["cloud"] = {
                "asof": cloud.get("asof"),
                "construction": cloud.get("construction"),
                "names": cloud.get("names"),
                "book": cloud.get("book"),
                "breaches": cloud.get("breaches"),
                "risk": cloud.get("risk"),
                "factor_ic": cloud.get("factor_ic"),
                "pipeline": cloud.get("pipeline"),
                "top_longs": [
                    {"symbol": p["symbol"], "weight": p["weight"], "sector": p["sector"]}
                    for p in cloud.get("points", [])
                    if p.get("side") == "long"
                ][:8],
                "top_shorts": [
                    {"symbol": p["symbol"], "weight": p["weight"], "sector": p["sector"]}
                    for p in cloud.get("points", [])
                    if p.get("side") == "short"
                ][:8],
            }
    except Exception as exc:
        payload["cloud_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def _pct(x: Any, digits: int = 1) -> str:
    try:
        return f"{100 * float(x):.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _num(x: Any, digits: int = 2) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def structured_brief(question: str, ctx: dict[str, Any] | None = None) -> str:
    """Deterministic briefing from snapshot numbers — no LLM required."""
    ctx = ctx or _snapshot_context()
    cloud = ctx.get("cloud") or {}
    book = cloud.get("book") or {}
    risk = cloud.get("risk") or {}
    research = ctx.get("research") or {}
    metrics = research.get("metrics") or {}
    ic = cloud.get("factor_ic") or research.get("factor_ic") or {}
    breaches = cloud.get("breaches") or []
    pipeline = cloud.get("pipeline") or []
    q = (question or "").lower()

    lines: list[str] = []

    if any(k in q for k in ("pipeline", "model", "math", "how does", "algorithm")):
        lines.append("Signal path (live construction):")
        for step in pipeline:
            lines.append(f"  {step.get('label')}: {step.get('detail')}")
        if risk.get("equation"):
            lines.append(f"Risk identity: {risk['equation']} (PCA factors, Connor–Korajczyk style).")

    if any(k in q for k in ("book", "summar", "overview", "risk", "neutral", "five")) or not lines:
        lines.append(
            f"Book asof {cloud.get('asof', '—')} · {cloud.get('construction', '—')} · "
            f"{cloud.get('names', '—')} names in the cloud."
        )
        lines.append(
            f"Gross {_num(book.get('gross'))} · net {_num(book.get('net'), 4)} · "
            f"{book.get('longs', '—')} long / {book.get('shorts', '—')} short."
        )
        if risk.get("available"):
            lines.append(
                f"Ex-ante vol {_pct(risk.get('ex_ante_vol'))} · realized {_pct(risk.get('realized_vol'))} · "
                f"systematic {_pct(risk.get('systematic_share'))} of variance · "
                f"effective names {_num(risk.get('effective_names'), 0)}."
            )
            if risk.get("eigenvalue_share"):
                shares = ", ".join(
                    f"PC{i + 1} {_pct(s, 0)}" for i, s in enumerate(risk["eigenvalue_share"][:5])
                )
                lines.append(f"PCA scree (factor var share): {shares}.")
        if breaches:
            lines.append("Limit breaches: " + "; ".join(b.get("name", "?") for b in breaches))
        else:
            lines.append("Risk limits: clear on this cross-section.")

    if any(k in q for k in ("ic", "factor", "t-stat", "newey")):
        if ic:
            ranked = sorted(
                ic.items(),
                key=lambda kv: abs(float((kv[1] or {}).get("ic_tstat_nw") or 0)),
                reverse=True,
            )
            for name, stats in ranked[:4]:
                lines.append(
                    f"IC {name}: mean {_num((stats or {}).get('ic_mean'), 3)} · "
                    f"NW t {_num((stats or {}).get('ic_tstat_nw'), 2)} · "
                    f"IR {_num((stats or {}).get('ic_ir'), 2)}"
                )
        else:
            lines.append("No factor IC table in the snapshot yet — run a backtest.")

    if any(k in q for k in ("backtest", "sharpe", "paper", "before", "check")):
        if research.get("available"):
            lines.append(
                f"Last backtest ({research.get('source', '—')}): Sharpe {_num(metrics.get('sharpe'))} · "
                f"ann return {_pct(metrics.get('ann_return'))} · "
                f"max DD {_pct(metrics.get('max_drawdown'))} · "
                f"turnover {_num(metrics.get('ann_turnover'), 1)}x."
            )
            if research.get("source") != "synthetic":
                lines.append("Tape is current-listing Yahoo — survivorship bias applies; research only.")
        else:
            lines.append("No stored backtest. Run: python -m gig backtest --source yahoo")
        lines.append(
            "Before paper trading: refresh lake (`python scripts/refresh_liquid.py`), "
            "`python -m gig trade plan`, then `trade run --yes` (or use serve --trade-every)."
        )

    longs = cloud.get("top_longs") or []
    shorts = cloud.get("top_shorts") or []
    if any(k in q for k in ("long", "short", "name", "holding")) and (longs or shorts):
        if longs:
            lines.append(
                "Top longs: "
                + ", ".join(f"{r['symbol']} {_pct(r.get('weight'), 1)}" for r in longs[:5])
            )
        if shorts:
            lines.append(
                "Top shorts: "
                + ", ".join(f"{r['symbol']} {_pct(r.get('weight'), 1)}" for r in shorts[:5])
            )

    lines.append("Source: structured snapshot (not an LLM). Numbers are from the live research path.")
    return "\n".join(lines)


SYSTEM = """You are the research desk for GIG Trading Algorithm.
You explain factor books, risk, and backtests from the JSON context you are given.
Rules:
- Never invent tickers, Sharpe ratios, or exposures that are not in the context.
- Never recommend placing a live order. Paper trading is already gated by the CLI.
- Prefer short, precise language. Lead with the conclusion.
- If the context is missing a number, say so.
- Call out high systematic risk, limit breaches, and survivorship bias when present.
"""


def research_brief(question: str, *, model: str | None = None) -> dict[str, Any]:
    """
    Ask the local model a question grounded in the current terminal snapshot.

    Always returns an answer: Ollama when available, otherwise ``structured_brief``.
    """
    status = ollama_status()
    ctx = _snapshot_context()
    local = structured_brief(question, ctx)

    if not status["available"]:
        return {
            **status,
            "available": True,
            "ollama": False,
            "source": "structured",
            "answer": local,
            "question": question,
        }

    chosen = model or _model()
    context = json.dumps(ctx, indent=2, default=str)[:12000]
    prompt = (
        f"{SYSTEM}\n\nCONTEXT (JSON):\n{context}\n\n"
        f"OPERATOR QUESTION:\n{question.strip()}\n\n"
        "Answer from the context only."
    )
    try:
        raw = _post(
            "/api/generate",
            {"model": chosen, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}},
            timeout=180.0,
        )
        answer = (raw.get("response") or "").strip()
        if not answer:
            answer = local
            source = "structured"
            ollama_ok = False
        else:
            source = "ollama"
            ollama_ok = True
        return {
            "available": True,
            "ollama": ollama_ok,
            "source": source,
            "host": _host(),
            "model": chosen,
            "question": question,
            "answer": answer,
            "eval_count": raw.get("eval_count"),
            "total_duration_ns": raw.get("total_duration"),
        }
    except Exception as exc:
        return {
            "available": True,
            "ollama": False,
            "source": "structured",
            "host": _host(),
            "model": chosen,
            "question": question,
            "answer": local,
            "error": f"{type(exc).__name__}: {exc}",
            "hint": status.get("hint") or f"ollama pull {chosen}",
        }


def default_prompts() -> list[str]:
    return [
        "Summarize the book and ex-ante risk in five bullets.",
        "Walk me through the math pipeline behind this book.",
        "Which factor has the strongest Newey–West t-stat?",
        "Is the book factor-neutral? Quote systematic share.",
        "What should I check before paper-trading today?",
    ]
