import sys
import os
import threading
import uuid
from typing import Any, Dict, Optional

# 1. Path Fixes - Absolute priority for Hugging Face imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.openapi.docs import get_swagger_ui_html

# 2. Imports - Handles both local and remote folder structures
try:
    from .inventory_restock_env_environment import InventoryRestockEnvironment, TASKS
    from .models import InventoryAction, InventoryObservation
except (ImportError, ModuleNotFoundError):
    try:
        from inventory_restock_env_environment import InventoryRestockEnvironment, TASKS
        from models import InventoryAction, InventoryObservation
    except:
        from server.inventory_restock_env_environment import InventoryRestockEnvironment, TASKS
        from models import InventoryAction, InventoryObservation

# 3. Session Store Setup
_sessions: Dict[str, Any] = {}
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

# 4. FastAPI App Initialization
app = FastAPI(
    title="Inventory Restocking Decision System",
    version="1.0.0",
    description=(
        "OpenEnv environment where AI agents manage inventory: "
        "identify low stock, predict demand, and optimise reorder strategies."
    ),
    docs_url=None,  # Disabled default to use custom UI on root
    redoc_url=None
)

# 5. UI & Health Endpoints
@app.get("/", include_in_schema=False)
async def custom_swagger_ui_html():
    """Loads Swagger UI directly on the home page to avoid blank screens."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Inventory Restocking API Docs",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
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

# 6. Core Logic Endpoints
@app.get("/schema")
def schema():
    return {
        "action": InventoryAction.model_json_schema(),
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
    task_id = body.get("task_id")
    sid = body.get("episode_id") or str(uuid.uuid4())
    env = _get_or_create(sid)
    obs = env.reset(task_id=task_id)
    return _obs_dict(obs, sid)

@app.post("/step")
def step(body: Dict[str, Any] = Body(...)):
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
    s = env.state
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
                "task_id": t.task_id,
                "description": t.description,
                "difficulty": t.difficulty,
                "max_attempts": t.max_attempts,
            }
            for t in TASKS
        ]
    }

# 7. Entry Point
def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
