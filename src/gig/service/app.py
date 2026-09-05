"""
HTTP adapter for the terminal UI.

The app is a thin shell over `gig.service.snapshot`: every route returns a
payload that module already built. FastAPI is an optional extra, so it is
imported inside `create_app()` and the package stays importable without it.

Read-only by design. There is no order-entry route, because a research UI that
can send orders is a production system with none of the controls of one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

STATIC_DIR = Path(__file__).parent / "static"


def create_app(cloud_limit: int = 220, cloud_ttl: float = 90.0) -> Any:
    try:
        from fastapi import Body, FastAPI, Query, Request
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - exercised by the CLI message
        raise ImportError(
            'FastAPI is not installed. Run: pip install -e ".[web]"'
        ) from exc

    from gig import __version__
    from gig.service import snapshot

    app = FastAPI(
        title="GIG Trading Algorithm",
        version=__version__,
        description="Read-only research terminal: factor space, book, risk limits, tape, Ollama desk.",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    @app.middleware("http")
    async def no_store(request, call_next):
        """
        Never let a browser cache the UI or an API payload.

        A cached app.js silently pins the user to an old build, which looks
        exactly like the code being broken. Research payloads are snapshots of a
        moving cross-section and are equally wrong to reuse.
        """
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response

    def cached_cloud(limit: int) -> dict[str, Any]:
        return snapshot._cached(
            f"cloud:{limit}", cloud_ttl, lambda: snapshot.factor_cloud(limit=limit)
        )

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "version": __version__}

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return snapshot.service_status()

    @app.get("/api/research")
    def research() -> dict[str, Any]:
        return snapshot.research_snapshot()

    @app.get("/api/experiments")
    def experiments(limit: int = Query(40, ge=1, le=500)) -> dict[str, Any]:
        return snapshot.experiment_history(limit=limit)

    @app.get("/api/cloud")
    def cloud(limit: int = Query(cloud_limit, ge=20, le=1200)) -> dict[str, Any]:
        return cached_cloud(limit)

    @app.get("/api/book")
    def book(limit: int = Query(cloud_limit, ge=20, le=1200)) -> dict[str, Any]:
        payload = cached_cloud(limit)
        if not payload.get("available"):
            return {"available": False, "longs": [], "shorts": []}
        held = [p for p in payload["points"] if p["side"] != "flat"]
        held.sort(key=lambda p: (p["score"] if p["score"] is not None else 0.0), reverse=True)
        return {
            "available": True,
            "asof": payload["asof"],
            "book": payload["book"],
            "longs": [p for p in held if p["side"] == "long"],
            "shorts": [p for p in held if p["side"] == "short"][::-1],
        }

    @app.get("/api/tape")
    def tape(limit: int = Query(40, ge=1, le=200)) -> dict[str, Any]:
        return snapshot.live_tape(limit=limit)

    @app.post("/api/cache/clear")
    def cache_clear() -> dict[str, Any]:
        snapshot.clear_cache()
        return {"cleared": True}

    @app.get("/api/paper")
    def paper(limit: int = Query(120, ge=20, le=600)) -> dict[str, Any]:
        """Alpaca paper account vs target. Read-only — never submits."""
        settings = __import__("gig.config", fromlist=["get_settings"]).get_settings()
        if not settings.keys_status().get("alpaca"):
            return {
                "available": False,
                "hint": "Set ALPACA_API_KEY / ALPACA_SECRET_KEY and use the paper URL",
            }
        try:
            from gig.execution.trader import account_status

            return {"available": True, **account_status(limit=limit)}
        except Exception as exc:
            return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    @app.get("/api/ollama/status")
    def ollama_status_route() -> dict[str, Any]:
        from gig.nlp.ollama import default_prompts, ollama_status

        return {**ollama_status(), "prompts": default_prompts()}

    @app.post("/api/ollama/brief")
    async def ollama_brief(request: Request, body: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        from gig.nlp.ollama import research_brief

        payload = body or {}
        # Some clients POST raw text; fall back to the query string.
        question = str(payload.get("question") or request.query_params.get("q") or "").strip()
        if not question:
            return {"available": False, "error": "question is required"}
        return research_brief(question, model=payload.get("model"))

    @app.get("/")
    def index() -> Any:
        page = STATIC_DIR / "index.html"
        if not page.exists():
            return JSONResponse({"error": "UI assets missing"}, status_code=500)
        return FileResponse(page)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    cloud_limit: int = 220,
    trade_every: int = 0,
    trade_limit: int = 120,
    trade_force: bool = False,
    trade_yes: bool = False,
) -> int:
    """Run the terminal with uvicorn. Optionally embed the paper loop in-process."""
    try:
        import uvicorn
    except ImportError:
        print('uvicorn is not installed. Run: pip install -e ".[web]"')
        return 1

    if trade_every > 0:
        _start_paper_loop(
            every=trade_every,
            limit=trade_limit,
            force=trade_force,
            dry_run=not trade_yes,
        )

    app = create_app(cloud_limit=cloud_limit)
    print(f"GIG terminal:  http://{host}:{port}")
    print(f"API docs:      http://{host}:{port}/api/docs")
    if trade_every > 0:
        mode = "LIVE orders" if trade_yes else "dry-run"
        print(f"paper loop:    every {trade_every}s  limit={trade_limit}  {mode}  force={trade_force}")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def _start_paper_loop(*, every: int, limit: int, force: bool, dry_run: bool) -> None:
    """Background paper loop sharing this process's DuckDB connection."""
    import threading
    import time

    from gig.data.lake import db_lock
    from gig.logging import configure_logging

    log = configure_logging()

    def worker() -> None:
        from gig.execution.trader import run_once

        # Let uvicorn bind before the first heavy target build.
        time.sleep(3)
        while True:
            try:
                with db_lock():
                    run = run_once(dry_run=dry_run, limit=limit, force=force)
                log.info("paper loop\n%s", run.report())
            except Exception as exc:
                log.warning("paper loop error: %s: %s", type(exc).__name__, exc)
            time.sleep(max(60, every))

    threading.Thread(target=worker, name="gig-paper-loop", daemon=True).start()
