"""FastAPI web server hosting the Browser Call Studio UI and WebSocket voice endpoint."""

from __future__ import annotations

import mimetypes
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .backend_factory import build_engine
from .call_recorder import list_recorded_calls, load_recorded_call
from .config import VoiceAgentConfig
from .session import CallSession

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Ensure AudioWorklet JS modules always serve with application/javascript on macOS.
mimetypes.add_type("application/javascript", ".js")


def create_app(
    config: VoiceAgentConfig,
    engine: Optional[Any] = None,
    load_on_startup: bool = True,
) -> FastAPI:
    """Build the FastAPI application with `/api/*`, `/static/*`, and `/ws/call`."""
    shared_engine = engine or build_engine(config.engine)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if load_on_startup and not shared_engine.is_loaded:
            shared_engine.load()
        try:
            yield
        finally:
            if engine is None:
                shared_engine.shutdown()

    app = FastAPI(title=config.app_title, lifespan=lifespan)
    app.state.config = config
    app.state.engine = shared_engine

    @app.middleware("http")
    async def disable_static_cache(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        return response

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "model_loaded": shared_engine.is_loaded,
                "model_id": config.engine.model_id,
                "backend": config.engine.backend,
            }
        )

    @app.get("/api/config")
    async def get_config() -> JSONResponse:
        return JSONResponse(config.to_public_config())

    @app.get("/api/calls")
    async def list_calls(limit: int = Query(default=50, ge=1, le=200)) -> JSONResponse:
        if not config.calls_dir:
            return JSONResponse({"calls_dir": None, "calls": []})
        return JSONResponse(
            {
                "calls_dir": str(Path(config.calls_dir).resolve()),
                "calls": list_recorded_calls(config.calls_dir, limit=limit),
            }
        )

    @app.get("/api/calls/{call_id:path}")
    async def get_call(
        call_id: str,
        include_trace: bool = True,
        trace_limit: int = Query(default=1000, ge=0, le=10000),
    ) -> JSONResponse:
        if not config.calls_dir:
            raise HTTPException(status_code=404, detail="Call recording is disabled")
        try:
            bundle = load_recorded_call(
                config.calls_dir,
                call_id,
                include_trace=include_trace,
                trace_limit=trace_limit,
            )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid call_id")
        if bundle is None:
            raise HTTPException(status_code=404, detail="Call not found")
        return JSONResponse(bundle)

    @app.websocket("/ws/call")
    async def ws_call(websocket: WebSocket) -> None:
        await websocket.accept()
        session = CallSession(websocket, config=config, engine=shared_engine)
        await session.run()

    return app


def run_server(config: VoiceAgentConfig, open_browser: bool = False) -> None:
    """Start the Uvicorn web server for the voice agent Call Studio."""
    url = f"http://{config.host}:{config.port}"
    print(f"\n  {config.app_title} ({config.company_name})", flush=True)
    print(f"  Backend: {config.engine.backend}", flush=True)
    print(f"  Model  : {config.engine.model_id}", flush=True)
    if config.engine.backend == "vllm":
        print(
            f"  vLLM   : native AsyncOmni (deploy_config={config.engine.vllm_deploy_config})",
            flush=True,
        )
    if config.calls_dir:
        print(f"  Calls  : {Path(config.calls_dir).resolve()}", flush=True)
    print(f"  Studio : {url}\n", flush=True)

    app = create_app(config)
    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
        ws_ping_interval=20.0,
        ws_ping_timeout=20.0,
    )
