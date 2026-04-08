#!/usr/bin/env python3
"""
inference.py — Inventory Restocking Decision System Baseline Agent
Updated with [START]/[STEP]/[END] structured output and Clamped Scores (0, 1).
"""
import argparse
import json
import os
import sys
import textwrap
import time
import urllib.request
import urllib.error
from typing import Optional
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

TEMPERATURE = 0.2
MAX_TOKENS  = 512

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert supply chain and inventory management AI.
    You will be given inventory data and must respond with a JSON object.
    Rules:
    - Reply with ONLY a valid JSON object — no explanation, no markdown fences
    - For T1 (identify low stock): {"low_stock_ids": ["P001", "P003"]}
    - For T2 (predict demand):     {"forecast": {"P001": 10.5, "P002": 5.0, ...}}
    - For T3 (optimize restock):   {"orders": {"P001": 50, "P002": 0, ...}}
    - All product IDs are P001–P005
    - Use only integer quantities in orders
""").strip()

def build_prompt(obs: dict) -> str:
    tid  = obs.get("task_id", "")
    desc = obs.get("task_description", "")
    inv  = obs.get("inventory", [])
    hist = obs.get("demand_history", {})
    day  = obs.get("day", 0)
    pending = obs.get("pending_orders", {})
    feedback = obs.get("last_action_feedback", "")
    score = obs.get("score", 0.0)
    attempts = obs.get("attempts", 0)

    lines = [
        f"TASK: {tid}",
        f"DESCRIPTION: {desc}",
        "",
        "CURRENT INVENTORY:",
    ]
    for p in inv:
        lines.append(
            f"  {p['id']} {p['name']}: stock={p['current_stock']} "
            f"reorder_pt={p['reorder_point']} avg_demand={p['avg_daily_demand']}/day "
            f"lead_time={p['lead_time_days']}d "
            f"hold_cost={p['holding_cost_per_day']}/unit/day order_cost={p['order_cost']}"
        )
    lines += ["", "DEMAND HISTORY (oldest→newest, last 14 days):"]
    for pid, h in hist.items():
        lines.append(f"  {pid}: {h}")

    if pending and any(pending.values()):
        lines += ["", "PENDING ORDERS (arriving on day):"]
        for pid, days in pending.items():
            if days:
                lines.append(f"  {pid}: {days}")

    if tid == "T3_optimize_restock":
        lines += [
            "",
            f"SIMULATION DAY: {day} / {obs.get('total_days', 14)}",
            f"Total holding cost so far: {obs.get('total_holding_cost', 0):.1f}",
            f"Total order cost so far: {obs.get('total_order_cost', 0):.1f}",
            f"Stockout days so far: {obs.get('total_stockout_days', 0)}",
        ]

    if feedback and attempts > 0:
        lines += ["", f"LAST FEEDBACK (attempt {attempts}, score={score:.2f}):", feedback]

    lines += ["", "Your JSON response:"]
    return "\n".join(lines)

def call_llm(client: OpenAI, prompt: str) -> dict:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    text = (completion.choices[0].message.content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}

# ── HTTP Client ───────────────────────────────────────────────────────────────
class EnvClient:
    def __init__(self, base_url: str):
        self._url = base_url.rstrip("/")
        self._episode_id: Optional[str] = None

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        req  = urllib.request.Request(
            f"{self._url}{path}", data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:200]}") from e

    def reset(self, task_id: Optional[str] = None) -> dict:
        body: dict = {}
        if task_id: body["task_id"] = task_id
        obs = self._post("/reset", body)
        self._episode_id = obs.get("episode_id")
        return obs

    def step(self, action: dict) -> dict:
        body: dict = {"action": action}
        if self._episode_id:
            body["episode_id"] = self._episode_id
        return self._post("/step", body)

    def tasks(self) -> list:
        return ["T1_identify_low_stock", "T2_predict_demand", "T3_optimize_restock"]

# ── Task runner ───────────────────────────────────────────────────────────────
def run_task(client: OpenAI, env: EnvClient, task_id: str) -> float:
    # [START] tag required by Validator
    print(f"[START] task={task_id}", flush=True)
        
    obs       = env.reset(task_id=task_id)
    max_steps = obs.get("max_attempts", 5)
    best      = 0.0
    steps_taken = 0

    for step_idx in range(1, max_steps + 1):
        if obs.get("done", False):
            break
                
        steps_taken += 1
        prompt = build_prompt(obs)
                
        try:
            action_data = call_llm(client, prompt)
            result = env.step(action_data)
        except Exception as exc:
            print(f"Error during {task_id} step {step_idx}: {exc}", file=sys.stderr)
            break

        score   = result.get("score", 0.0)
        # Ensure reward is also in a valid format
        reward  = result.get("reward", 0.0)
        best    = max(best, score)
        done    = result.get("done", False)
                
        # [STEP] tag required by Validator
        print(f"[STEP] step={step_idx} reward={reward}", flush=True)

        obs = result
        if done:
            break

    # --- CRITICAL FIX: CLAMP SCORE TO (0, 1) ---
    # This prevents the "score out of range" error (0.0 and 1.0 are rejected)
    safe_score = max(0.01, min(0.99, best))

    # [END] tag required by Validator
    print(f"[END] task={task_id} score={safe_score} steps={steps_taken}", flush=True)
    return safe_score

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Inventory Restock baseline agent")
    parser.add_argument("--url",  default=ENV_URL)
    parser.add_argument("--task", default=None)
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: Set HF_TOKEN environment variable", file=sys.stderr)
        sys.exit(1)

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EnvClient(args.url)
    
    if args.task:
        task_ids = [args.task]
    else:
        task_ids = ["T1_identify_low_stock", "T2_predict_demand", "T3_optimize_restock"]

    scores = {}
    for tid in task_ids:
        try:
            scores[tid] = run_task(llm, env, tid)
        except Exception as exc:
            print(f"Critical error on {tid}: {exc}", file=sys.stderr)
            scores[tid] = 0.01 # Use 0.01 instead of 0.0 for safety

    avg = sum(scores.values()) / len(scores) if scores else 0.01
    sys.exit(0 if avg > 0 else 1)

if __name__ == "__main__":
    main()
