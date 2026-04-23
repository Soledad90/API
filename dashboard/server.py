"""
FastAPI dashboard server cho VN Stock AI Agent.

Endpoints:
  GET  /              — Dashboard UI (index.html)
  GET  /api/status    — Trạng thái agent
  GET  /api/symbols   — Danh sách mã theo dõi
  GET  /api/history   — Lịch sử quyết định (từ memory)
  WS   /ws            — WebSocket nhận kết quả real-time
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.memory import load_memory, compute_performance_stats
from src.utils import NumpyEncoder

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manage active WebSocket connections and broadcast messages."""

    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)
        logger.info("WS connected (total: %d)", len(self.active))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)
        logger.info("WS disconnected (total: %d)", len(self.active))

    async def broadcast(self, data: dict[str, Any]) -> None:
        """Send JSON data to all connected WebSocket clients."""
        msg = json.dumps(data, ensure_ascii=False, cls=NumpyEncoder)
        dead: list[WebSocket] = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()

# Shared state (updated by the agent loop)
_agent_state: dict[str, Any] = {
    "running": False,
    "symbols": [],
    "last_scan": None,
    "results": {},
}


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Dashboard server starting…")
    yield
    logger.info("Dashboard server shutting down…")


app = FastAPI(
    title="VN Stock AI Agent Dashboard",
    version="2.0.0",
    lifespan=lifespan,
)

# Serve static files (CSS/JS assets)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# HTTP Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def index():
    html = STATIC_DIR / "index.html"
    if html.exists():
        return FileResponse(html)
    return JSONResponse({"error": "index.html not found"}, status_code=404)


@app.get("/api/status")
async def api_status():
    data = {
        "running": _agent_state["running"],
        "symbols": _agent_state["symbols"],
        "last_scan": _agent_state["last_scan"],
        "connected_clients": len(manager.active),
    }
    return Response(content=json.dumps(data, cls=NumpyEncoder), media_type="application/json")


@app.get("/api/symbols")
async def api_symbols():
    data = {"symbols": _agent_state["symbols"]}
    return Response(content=json.dumps(data, cls=NumpyEncoder), media_type="application/json")


@app.get("/api/history")
async def api_history():
    history = load_memory()
    stats = compute_performance_stats(history)
    # Return last 50 records
    recent = history[-50:] if len(history) > 50 else history
    data = {
        "history": recent,
        "stats": stats,
        "total": len(history),
    }
    return Response(content=json.dumps(data, cls=NumpyEncoder), media_type="application/json")


@app.get("/api/results")
async def api_results():
    data = _agent_state.get("results", {})
    return Response(content=json.dumps(data, cls=NumpyEncoder), media_type="application/json")


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    # Send current state immediately on connect
    if _agent_state["results"]:
        await ws.send_text(json.dumps({
            "type": "state",
            "data": _agent_state["results"],
        }, ensure_ascii=False, cls=NumpyEncoder))
    try:
        while True:
            # Keep connection alive; accept ping messages
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ---------------------------------------------------------------------------
# Broadcast helper (called by agent loop)
# ---------------------------------------------------------------------------

async def broadcast_result(result: dict[str, Any]) -> None:
    """Broadcast a single symbol's analysis result to all WS clients."""
    symbol = result.get("symbol", "?")
    _agent_state["results"][symbol] = result
    _agent_state["last_scan"] = result.get("timestamp")

    await manager.broadcast({
        "type": "update",
        "symbol": symbol,
        "data": result,
    })


def get_broadcast_fn():
    """Return a coroutine function suitable for passing to run_realtime_loop."""
    return broadcast_result
