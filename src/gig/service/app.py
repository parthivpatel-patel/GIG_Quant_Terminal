"""
HTTP adapter for the terminal UI.

Routes expose typed snapshots and fixed-path local research workspaces. FastAPI is an optional extra, so it is
imported inside `create_app()` and the package stays importable without it.

Workspace writes admit caller evidence and run local research or simulations.
There is no brokerage order-entry route and no automatic provider connection.
"""

import inspect
import threading
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Annotated, Any

STATIC_DIR = Path(__file__).parent / "static"


def create_app(cloud_limit: int = 220, cloud_ttl: float = 90.0, *, state_dir: Path | None = None,
               finbert_dir: Path | None = None) -> Any:
    try:
        from fastapi import Body, FastAPI, Query, Request
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import FileResponse, JSONResponse, Response
        from fastapi.routing import APIRoute
        from fastapi.staticfiles import StaticFiles
        from starlette.concurrency import run_in_threadpool
        from starlette.exceptions import HTTPException
    except ImportError as exc:  # pragma: no cover - exercised by the CLI message
        raise ImportError(
            'FastAPI is not installed. Run: pip install -e ".[web]"'
        ) from exc

    from gig import __version__
    from gig.config import get_settings
    from gig.service import snapshot
    from gig.service.contracts import Contract, EnvelopeSchema, State
    from gig.service.cycle_services import CycleServices
    from gig.service.serialization import (
        DEFAULT_LIMITS,
        SerializationError,
        dumps,
        loads,
        public_error,
    )
    from gig.service.workspace import TerminalWorkspace
    from gig.timestamps import aware_timestamp

    workspace = TerminalWorkspace(state_dir if state_dir is not None else get_settings().data_dir / "terminal")
    integrations = CycleServices(workspace, finbert_dir=(finbert_dir if finbert_dir is not None
                                                       else get_settings().finbert_model_dir))
    workspace_write_lock = threading.Lock()

    class StrictJSONResponse(JSONResponse):
        def render(self, content: Any) -> bytes:
            return dumps(content).encode("utf-8")

    class StrictRoute(APIRoute):
        """Serialize the domain result before FastAPI's coercive jsonable_encoder."""
        def __init__(self, path, endpoint, **kwargs):
            @wraps(endpoint)
            async def boundary(*args, **call_kwargs):
                if inspect.iscoroutinefunction(endpoint):
                    payload = await endpoint(*args, **call_kwargs)
                else:
                    payload = await run_in_threadpool(endpoint, *args, **call_kwargs)
                if isinstance(payload, Response):
                    return payload
                if isinstance(payload, Contract):
                    return Response(payload.wire, media_type="application/json")
                return StrictJSONResponse(payload)
            super().__init__(path, endpoint=boundary, **kwargs)

    def failure(code: str, status: int = 500) -> Response:
        contract = Contract.admit(State.ERROR, None, source="api-boundary", scope="api",
                                  classification="UNAVAILABLE", error=public_error(code))
        return Response(contract.wire, media_type="application/json", status_code=status)

    app = FastAPI(
        title="GIG Trading Algorithm",
        version=__version__,
        description="Local research terminal with admitted evidence, reproducible experiments, and offline simulation. No brokerage order-entry API.",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.router.route_class = StrictRoute
    app.state.workspace = workspace

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Do not echo Pydantic's input, provider diagnostics, or exception repr.
        return failure("INVALID_REQUEST", 422)

    @app.exception_handler(SerializationError)
    async def invalid_result(request, exc):
        return failure("SERIALIZATION_ERROR", 500)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return failure("INVALID_REQUEST", exc.status_code)

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        return failure("INTERNAL_ERROR", 500)

    @app.middleware("http")
    async def no_store(request, call_next):
        """
        Never let a browser cache the UI or an API payload.

        A cached app.js silently pins the user to an old build, which looks
        exactly like the code being broken. Research payloads are snapshots of a
        moving cross-section and are equally wrong to reuse.
        """
        try:
            if request.url.path.startswith("/api/") and request.method in ("POST", "PUT", "PATCH"):
                chunks, count = [], 0
                async for chunk in request.stream():
                    count += len(chunk)
                    if count > DEFAULT_LIMITS.max_bytes:
                        return failure("INVALID_REQUEST", 413)
                    chunks.append(chunk)
                raw = b"".join(chunks)
                # Starlette's cached request replays these same original bytes
                # to FastAPI. No re-dumping, text trimming, or float coercion.
                request._body = raw
                if raw:
                    loads(raw)
            response = await call_next(request)
        except SerializationError:
            response = failure("INVALID_REQUEST", 400)
        except Exception:
            response = failure("INTERNAL_ERROR", 500)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/api/v1/terminal/{section}", response_model=EnvelopeSchema)
    def terminal_section(section: str, mode: str = Query("local", pattern="^(local|synthetic)$"),
                         as_of: str | None = Query(None, max_length=100),
                         max_age_seconds: float | None = Query(None, ge=0, le=366 * 86400),
                         engine: str = Query("default", pattern="^(default|sequence|finbert|provider|execution-engine|persisted|allocation|accounting|depth|pairs|options-research|signals-research|options-overlay|funded-account|economic-research|microstructure-research|funded-risk|funded-lifecycle|independent-statement|observed-reference)$"),
                         universe_id: str | None = Query(None, min_length=1, max_length=128),
                         instrument_id: str | None = Query(None, min_length=1, max_length=128)) -> Any:
        from gig.service.terminal import SECTIONS, section_snapshot
        if section not in SECTIONS:
            return failure("INVALID_REQUEST", 404)
        try:
            cutoff = aware_timestamp(as_of if as_of is not None else datetime.now(UTC))
        except (ValueError, TypeError, OverflowError):
            return failure("INVALID_REQUEST", 422)
        if mode == "local":
            if engine != "default":
                try:
                    return integrations.section(engine, section, cutoff, max_age_seconds,
                                                universe_id=universe_id, instrument_id=instrument_id)
                except ImportError:
                    return failure("DEPENDENCY_UNAVAILABLE", 503)
            result = workspace.section(section, cutoff, max_age_seconds)
            if result is not None:
                return result
        if engine != "default":
            return failure("INVALID_REQUEST", 422)
        return section_snapshot(section, mode)

    @app.post("/api/v1/workspace/{action}", response_model=EnvelopeSchema)
    async def workspace_action(action: str, request: Request) -> Any:
        methods = {"bars": workspace.ingest_bars, "news": workspace.ingest_news,
                   "research": workspace.run_research, "simulate": workspace.simulate,
                   "ingest": integrations.ingest, "universe": integrations.universe,
                   "train-sequence": integrations.train_sequence,
                   "predict-sequence": integrations.predict_sequence,
                   "finbert": integrations.analyze_finbert, "execution-engine": integrations.execution,
                   "research-dataset": integrations.research_dataset, "research-model": integrations.research_model,
                   "research-predict": integrations.research_predict, "research-allocate": integrations.research_allocate,
                   "research-prepare": integrations.research_prepare, "portfolio-accounting": integrations.portfolio_accounting,
                   "depth": integrations.admit_depth,
                   "pairs-research": integrations.pairs_research, "pairs-replay": integrations.pairs_replay,
                   "options-admit": integrations.options_admit, "options-research": integrations.options_research,
                   "options-replay": integrations.options_replay, "research-signals": integrations.research_signals,
                   "signals-replay": integrations.signals_replay,
                   "options-overlay": integrations.options_overlay, "options-overlay-replay": integrations.overlay_replay,
                   "funded-account": integrations.funded_account,
                   "economic-research": integrations.economic_research, "economic-replay": integrations.economic_replay,
                   "microstructure-admit": integrations.microstructure_admit,
                   "microstructure-research": integrations.microstructure_research,
                   "microstructure-replay": integrations.microstructure_replay,
                   "funded-risk": integrations.funded_risk, "funded-risk-replay": integrations.funded_risk_replay,
                   "funded-lifecycle": integrations.funded_lifecycle,
                   "statement-admit": integrations.statement_admit, "statement-reconcile": integrations.statement_reconcile,
                   "statement-replay": integrations.statement_replay, "observed-reference": integrations.observed_reference}
        if action not in methods:
            return failure("INVALID_REQUEST", 404)
        # Local mutations require JSON and reject cross-origin browser writes.
        # Reads and numerical calculations confer no broker authority.
        if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/json":
            return failure("INVALID_REQUEST", 415)
        origin = request.headers.get("origin")
        if origin is not None and origin != f"{request.url.scheme}://{request.url.netloc}":
            return failure("INVALID_REQUEST", 403)
        if not workspace_write_lock.acquire(blocking=False):
            return failure("INVALID_REQUEST", 429)
        try:
            payload = loads(await request.body())
            if not isinstance(payload, dict):
                return failure("INVALID_REQUEST", 422)
            return await run_in_threadpool(methods[action], payload)
        except SerializationError:
            return failure("INVALID_REQUEST", 400)
        except (ValueError, TypeError, KeyError, OverflowError):
            return failure("INVALID_REQUEST", 422)
        except ImportError:
            return failure("DEPENDENCY_UNAVAILABLE", 503)
        finally:
            workspace_write_lock.release()

    @app.post("/api/v1/analytics/{kind}", response_model=EnvelopeSchema)
    async def numerical_research(kind: str, request: Request) -> Any:
        from gig.service.terminal import analytics
        try:
            payload = loads(await request.body())
            if not isinstance(payload, dict):
                return failure("INVALID_REQUEST", 422)
            return await run_in_threadpool(analytics, kind, payload)
        except SerializationError:
            return failure("INVALID_REQUEST", 400)
        except (ValueError, TypeError, KeyError, OverflowError):
            return failure("INVALID_REQUEST", 422)
        except ImportError:
            return failure("DEPENDENCY_UNAVAILABLE", 503)

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
        if not settings.broker_connectivity_enabled:
            return {"available": False, "state": "UNAVAILABLE", "error": public_error("BROKER_DISABLED")}
        if not settings.keys_status().get("alpaca"):
            return {
                "available": False,
                "hint": "Set ALPACA_API_KEY / ALPACA_SECRET_KEY and use the paper URL",
            }
        try:
            from gig.data.lake import db_lock
            from gig.execution.trader import account_status

            with db_lock():
                return {"available": True, **account_status(limit=limit)}
        except Exception:
            return {"available": False, "state": "ERROR", "error": public_error()}

    @app.get("/api/name/{symbol}")
    def name_detail(symbol: str, limit: int = Query(cloud_limit, ge=20, le=1200)) -> dict[str, Any]:
        """Single-name research card from the live cloud snapshot."""
        sym = str(symbol or "").upper().strip()
        if not sym:
            return {"available": False, "error": "symbol required"}
        cloud = cached_cloud(limit)
        if not cloud.get("available"):
            return {"available": False, "hint": cloud.get("hint") or "cloud unavailable"}
        point = next((p for p in cloud.get("points") or [] if str(p.get("symbol")) == sym), None)
        if point is None:
            return {"available": False, "hint": f"{sym} not in current cloud"}
        risk = cloud.get("risk") or {}
        contrib = next(
            (c for c in (risk.get("top_contributors") or []) if str(c.get("symbol")) == sym),
            None,
        )
        return {
            "available": True,
            "asof": cloud.get("asof"),
            "construction": cloud.get("construction"),
            "names": cloud.get("names"),
            "point": point,
            "risk_contributor": contrib,
            "book": cloud.get("book"),
            "factor_ic": cloud.get("factor_ic"),
        }

    @app.get("/api/ops")
    def ops() -> dict[str, Any]:
        from gig.ops.status import ops_snapshot

        return ops_snapshot()

    @app.get("/api/attribution")
    def attribution(limit: int = Query(cloud_limit, ge=20, le=1200)) -> dict[str, Any]:
        from gig.research.attribution import attribution_snapshot

        cloud = cached_cloud(limit)
        return attribution_snapshot(
            risk=cloud.get("risk") if cloud.get("available") else None,
            factor_ic=cloud.get("factor_ic") if cloud.get("available") else None,
        )

    @app.get("/api/blotter")
    def blotter(limit: int = Query(120, ge=20, le=600)) -> dict[str, Any]:
        from gig.research.attribution import blotter_snapshot

        return blotter_snapshot(limit=limit)

    @app.get("/api/ollama/status")
    def ollama_status_route() -> dict[str, Any]:
        from gig.nlp.ollama import default_prompts, ollama_status

        return {**ollama_status(), "prompts": default_prompts()}

    @app.post("/api/ollama/brief")
    async def ollama_brief(request: Request, body: Annotated[dict[str, Any] | None, Body()] = None) -> dict[str, Any]:
        from gig.nlp.ollama import research_brief

        payload = body or {}
        # Some clients POST raw text; fall back to the query string.
        question = payload.get("question", request.query_params.get("q", ""))
        if not isinstance(question, str) or not question.strip():
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

    from gig.service.snapshot import warm_status_cache

    # Prime lake stats while DuckDB is free, before the paper loop contends.
    warm_status_cache()

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
    from gig.ops.killswitch import kill_switch
    from gig.ops.status import touch_heartbeat

    log = configure_logging()

    def worker() -> None:
        from gig.execution.trader import run_once

        # Let uvicorn bind and the status strip warm before the first heavy target build.
        time.sleep(8)
        while True:
            try:
                if kill_switch().engaged:
                    touch_heartbeat(ok=True, detail="kill switch engaged — skipping pass")
                    log.info("paper loop idle: kill switch engaged")
                else:
                    with db_lock():
                        run = run_once(dry_run=dry_run, limit=limit, force=force)
                    touch_heartbeat(
                        ok=not run.blocked,
                        detail=("blocked by risk gate" if run.blocked else "ok")
                        + (f" · submitted={run.submitted}" if not dry_run else " · dry-run"),
                        run_id=run.run_id,
                    )
                    log.info("paper loop\n%s", run.report())
            except Exception as exc:
                touch_heartbeat(ok=False, detail=f"{type(exc).__name__}: {exc}")
                log.warning("paper loop error: %s: %s", type(exc).__name__, exc)
            time.sleep(max(60, every))

    threading.Thread(target=worker, name="gig-paper-loop", daemon=True).start()
