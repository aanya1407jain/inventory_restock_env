"""
FastAPI application for the Inventory Restocking Decision System.

Session-managed HTTP server following the OpenEnv HTTP spec.
Each reset() returns an episode_id; pass it in step() to maintain state.

Endpoints:
    POST /reset    Reset environment
    POST /step     Execute an action
    GET  /state    Current state
    GET  /schema   Action / observation / state schemas
    GET  /metadata Environment metadata
    GET  /health   Health check
    POST /mcp      JSON-RPC 2.0 stub
    GET  /tasks    List all tasks
"""
import sys
import os

# This adds the current folder (server) to the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# This adds the parent folder (inventory_restock_env) to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import threading
import uuid
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse

try:
    from ..models import InventoryAction, InventoryObservation
    from .inventory_restock_env_environment import (
        InventoryRestockEnvironment, TASKS
    )
except (ModuleNotFoundError, ImportError):
    from models import InventoryAction, InventoryObservation
    from server.inventory_restock_env_environment import (
        InventoryRestockEnvironment, TASKS
    )

# ── Session store ─────────────────────────────────────────────────────────────
_sessions: Dict[str, InventoryRestockEnvironment] = {}
_lock = threading.Lock()
_DEFAULT = "default"


def _get_or_create(sid: str) -> InventoryRestockEnvironment:
    with _lock:
        if sid not in _sessions:
            _sessions[sid] = InventoryRestockEnvironment()
        return _sessions[sid]


def _obs_dict(obs: InventoryObservation, sid: str) -> dict:
    d = obs.model_dump()
    d["episode_id"] = sid
    return d


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Inventory Restocking Decision System",
    version="1.0.0",
    description=(
        "OpenEnv environment where AI agents manage inventory: "
        "identify low stock, predict demand, and optimise reorder strategies."
    ),
)


@app.get("/health")
def health():
    return {"status": "healthy", "environment": "inventory_restock_env", "tasks": len(TASKS)}


@app.get("/metadata")
def metadata():
    return {
        "name": "inventory_restock_env",
        "description": (
            "Inventory Restocking Decision System: agents monitor stock levels, "
            "predict demand, and optimize restocking decisions to minimize costs "
            "and prevent stockouts across a 5-product catalogue."
        ),
        "version": "1.0.0",
        "tasks": [t.task_id for t in TASKS],
    }


@app.get("/schema")
def schema():
    return {
        "action":      InventoryAction.model_json_schema(),
        "observation": InventoryObservation.model_json_schema(),
        "state": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "step_count": {"type": "integer"},
            },
        },
    }


@app.post("/reset")
def reset(body: Dict[str, Any] = Body(default={})):
    """
    Reset the environment.
    Optional: {"task_id": "T1_identify_low_stock", "episode_id": "my-session"}
    """
    task_id = body.get("task_id")
    sid     = body.get("episode_id") or str(uuid.uuid4())
    env     = _get_or_create(sid)
    obs     = env.reset(task_id=task_id)
    return _obs_dict(obs, sid)


@app.post("/step")
def step(body: Dict[str, Any] = Body(...)):
    """
    Execute one step.
    Body: {"action": {...}, "episode_id": "<from reset>"}
    Action fields:
      low_stock_ids  — T1: list of product IDs
      forecast       — T2: {product_id: predicted_avg_daily_demand}
      orders         — T3: {product_id: units_to_order}
    """
    action_data = body.get("action")
    if action_data is None:
        raise HTTPException(status_code=422, detail="'action' field required")

    sid = body.get("episode_id", _DEFAULT)
    try:
        action = InventoryAction(**action_data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    env = _get_or_create(sid)
    if env._task is None:
        env.reset()

    obs = env.step(action)
    return _obs_dict(obs, sid)


@app.get("/state")
def state(episode_id: str = _DEFAULT):
    env = _get_or_create(episode_id)
    s   = env.state
    return {"episode_id": s.episode_id or episode_id, "step_count": s.step_count}


@app.post("/mcp")
def mcp(body: Dict[str, Any] = Body(default={})):
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": body.get("id"),
        "result": {
            "name": "inventory_restock_env",
            "description": "Inventory Restocking Decision System OpenEnv environment",
        },
    })


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "task_id":    t.task_id,
                "description": t.description,
                "difficulty": t.difficulty,
                "max_attempts": t.max_attempts,
            }
            for t in TASKS
        ]
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point — enables: uv run --project . server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main()
