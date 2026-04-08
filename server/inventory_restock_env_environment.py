"""
Inventory Restocking Decision System — Environment Implementation
Three tasks of increasing difficulty:
  T1  identify_low_stock   — Easy:   find products below reorder point
  T2  predict_demand       — Medium: forecast next-period demand from history
  T3  optimize_restock     — Hard:   run a 14-day simulation, minimize cost & stockouts
Reward shaping:
  T1 / T2 : immediate score on submission, -0.05 stagnation penalty
  T3       : per-day shaped reward = service_level_bonus - holding_cost - stockout_penalty
"""

import math
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import InventoryAction, InventoryObservation
except ImportError:
    from models import InventoryAction, InventoryObservation


# ─────────────────────────────────────────────────────────────────────────────
# Product catalogue  (deterministic — same every episode)
# ─────────────────────────────────────────────────────────────────────────────

PRODUCTS = [
    {
        "id": "P001", "name": "Widget A",
        "avg_daily_demand": 10, "demand_std": 2,
        "lead_time_days": 2,
        "reorder_point": 25,          # < 25 units → low stock
        "unit_cost": 15.0,
        "holding_cost_per_day": 0.50,
        "order_cost": 20.0,
        "max_stock": 200,
    },
    {
        "id": "P002", "name": "Widget B",
        "avg_daily_demand": 5, "demand_std": 1,
        "lead_time_days": 3,
        "reorder_point": 20,
        "unit_cost": 25.0,
        "holding_cost_per_day": 1.00,
        "order_cost": 30.0,
        "max_stock": 150,
    },
    {
        "id": "P003", "name": "Gadget X",
        "avg_daily_demand": 3, "demand_std": 1,
        "lead_time_days": 4,
        "reorder_point": 15,
        "unit_cost": 80.0,
        "holding_cost_per_day": 2.00,
        "order_cost": 50.0,
        "max_stock": 80,
    },
    {
        "id": "P004", "name": "Gadget Y",
        "avg_daily_demand": 8, "demand_std": 2,
        "lead_time_days": 2,
        "reorder_point": 20,
        "unit_cost": 20.0,
        "holding_cost_per_day": 0.80,
        "order_cost": 25.0,
        "max_stock": 180,
    },
    {
        "id": "P005", "name": "Component Z",
        "avg_daily_demand": 15, "demand_std": 3,
        "lead_time_days": 1,
        "reorder_point": 20,
        "unit_cost": 8.0,
        "holding_cost_per_day": 0.30,
        "order_cost": 15.0,
        "max_stock": 300,
    },
]

PRODUCT_MAP = {p["id"]: p for p in PRODUCTS}

# ─────────────────────────────────────────────────────────────────────────────
# Deterministic demand generator (seeded for reproducibility)
# ─────────────────────────────────────────────────────────────────────────────

def _demand_sequence(product: dict, n_days: int, seed: int = 42) -> List[int]:
    """Generate n_days of deterministic daily demand for a product."""
    rng = random.Random(seed + hash(product["id"]) % 1000)
    result = []
    for _ in range(n_days):
        d = int(rng.gauss(product["avg_daily_demand"], product["demand_std"]))
        result.append(max(0, d))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Task definitions
# ─────────────────────────────────────────────────────────────────────────────

class Task:
    def __init__(self, task_id, description, difficulty, max_attempts):
        self.task_id     = task_id
        self.description = description
        self.difficulty  = difficulty
        self.max_attempts = max_attempts


TASKS = [
    Task(
        task_id="T1_identify_low_stock",
        description=(
            "You are given the current inventory snapshot for 5 products. "
            "Identify which products are LOW ON STOCK — i.e. their current_stock "
            "is AT OR BELOW their reorder_point. "
            "Submit their product IDs in the 'low_stock_ids' field of your action."
        ),
        difficulty="easy",
        max_attempts=3,
    ),
    Task(
        task_id="T2_predict_demand",
        description=(
            "You are given 14 days of historical daily demand for each product. "
            "Predict the AVERAGE DAILY DEMAND for the next 7 days for each product. "
            "Submit your predictions in the 'forecast' field: {product_id: avg_daily_demand}. "
            "Closer predictions score higher (scored on mean absolute percentage error)."
        ),
        difficulty="medium",
        max_attempts=3,
    ),
    Task(
        task_id="T3_optimize_restock",
        description=(
            "Run a 14-day inventory simulation. Each day you observe stock levels "
            "and decide how many units to reorder for each product (field: 'orders'). "
            "Orders arrive after lead_time_days. "
            "Goal: minimize stockouts AND minimize total cost (holding + ordering). "
            "Score = weighted combination of service level and cost efficiency."
        ),
        difficulty="hard",
        max_attempts=14,
    ),
]

TASK_MAP = {t.task_id: t for t in TASKS}


# ─────────────────────────────────────────────────────────────────────────────
# Graders
# ─────────────────────────────────────────────────────────────────────────────
def _grade_t1(low_stock_ids: List[str], true_low: List[str]) -> Tuple[float, str]:
    """F1 score for low-stock identification, clamped to (0.01, 0.99)."""
    if not true_low:
        # Clamped: 1.0 becomes 0.99, 0.5 remains 0.5
        score = 0.99 if not low_stock_ids else 0.5
        return score, "No products are low on stock."
        
    predicted = set(pid.strip().upper() for pid in low_stock_ids)
    actual    = set(p.upper() for p in true_low)
    
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    f1 = (2 * precision * recall / (precision + recall) 
          if (precision + recall) > 0 else 0.0)
    
    # --- CRITICAL FIX: Clamp score strictly between 0 and 1 ---
    safe_f1 = max(0.01, min(0.99, f1))
    
    feedback = (
        f"True low-stock: {sorted(actual)}. "
        f"You predicted: {sorted(predicted)}. "
        f"TP={tp} FP={fp} FN={fn} → F1={f1:.2f}"
    )
    return round(safe_f1, 4), feedback


def _grade_t2(forecast: Dict[str, float], true_demands: Dict[str, float]) -> Tuple[float, str]:
    """Score based on MAPE, clamped to (0.01, 0.99)."""
    if not forecast:
        return 0.01, "Empty forecast submitted."
        
    errors = []
    details = []
    for pid, true_val in true_demands.items():
        pred = forecast.get(pid, 0.0)
        if true_val > 0:
            pct_err = abs(pred - true_val) / true_val
        else:
            pct_err = 0.0 if pred == 0 else 1.0
        errors.append(pct_err)
        details.append(f"{pid}: predicted={pred:.1f} actual={true_val:.1f} err={pct_err*100:.0f}%")
        
    mape  = sum(errors) / len(errors) if errors else 1.0
    score = max(0.0, 1.0 - mape) # Raw score calculation
    
    # --- CRITICAL FIX: Clamp score strictly between 0 and 1 ---
    safe_score = max(0.01, min(0.99, score))
    
    feedback = f"MAPE={mape*100:.1f}% | " + " | ".join(details)
    return round(safe_score, 4), feedback

def _grade_t3_step(stocks: Dict[str, int],
                   demands: Dict[str, int],
                   holding_cost_today: float,
                   order_cost_today: float) -> Tuple[float, str]:
    """Per-day reward for T3, clamped to (0.01, 0.99)."""
    n_products  = len(stocks)
    stockouts   = sum(1 for pid, s in stocks.items() if s <= 0)
    service_lvl = (n_products - stockouts) / n_products

    # Normalise costs
    max_daily_hold  = sum(p["max_stock"] * p["holding_cost_per_day"] for p in PRODUCTS)
    max_daily_order = sum(p["order_cost"] for p in PRODUCTS)
    norm_hold  = holding_cost_today  / max(max_daily_hold,  1)
    norm_order = order_cost_today    / max(max_daily_order, 1)

    reward = (0.6 * service_lvl) - (0.2 * norm_hold) - (0.2 * norm_order)
    
    # --- CLAMP REWARD (Safety for validator) ---
    safe_reward = max(0.01, min(0.99, reward))
    
    feedback = (
        f"Stockouts={stockouts}/{n_products}  "
        f"HoldCost={holding_cost_today:.1f}  "
        f"OrderCost={order_cost_today:.1f}  "
        f"DayReward={reward:.3f}"
    )
    return round(safe_reward, 4), feedback


def _grade_t3_final(total_holding: float,
                    total_order: float,
                    total_stockout_days: int,
                    n_products: int,
                    n_days: int) -> Tuple[float, str]:
    """Episode-end score for T3, clamped to (0.01, 0.99)."""
    max_stockout_days = n_products * n_days
    service_level = 1.0 - (total_stockout_days / max(max_stockout_days, 1))

    # EOQ-based ideal cost
    ideal_total = 0.0
    for p in PRODUCTS:
        d  = p["avg_daily_demand"]
        h  = p["holding_cost_per_day"]
        S  = p["order_cost"]
        L  = p["lead_time_days"]
        eoq        = math.sqrt(2 * d * S / max(h, 0.01))
        n_orders   = math.ceil(d * n_days / max(eoq, 1))
        safety     = d * L
        ideal_total += n_orders * S + (eoq / 2 + safety) * h * n_days
    ideal_total = max(ideal_total, 1.0)

    actual_total = total_holding + total_order
    cost_ratio  = min(actual_total / ideal_total, 3.0)
    cost_score  = max(0.0, 1.0 - (cost_ratio - 1.0) / 2.0)

    score = 0.6 * service_level + 0.4 * cost_score
    
    # --- CRITICAL FIX: Clamp score strictly between 0 and 1 ---
    safe_score = max(0.01, min(0.99, score))
    
    feedback = (
        f"ServiceLevel={service_level*100:.1f}%  "
        f"TotalCost={actual_total:.1f}  "
        f"IdealCost≈{ideal_total:.1f}  "
        f"CostScore={cost_score:.2f}  "
        f"FinalScore={score:.2f}"
    )
    return round(safe_score, 4), feedback
# ─────────────────────────────────────────────────────────────────────────────
# T1 setup: deterministic snapshot (some products below reorder point)
# ─────────────────────────────────────────────────────────────────────────────

T1_STOCK = {
    "P001": 18,   # reorder_point=25 → LOW ✓
    "P002": 45,   # reorder_point=20 → OK
    "P003": 12,   # reorder_point=15 → LOW ✓
    "P004": 20,   # reorder_point=20 → LOW ✓ (exactly at reorder point)
    "P005": 80,   # reorder_point=20 → OK
}
T1_TRUE_LOW = [pid for pid, stock in T1_STOCK.items()
               if stock <= PRODUCT_MAP[pid]["reorder_point"]]


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class InventoryRestockEnvironment(Environment):
    """
    Inventory Restocking Decision System.
    T1 (easy)   — single-step identification task
    T2 (medium) — single-step demand forecasting task
    T3 (hard)   — 14-step simulation with daily reorder decisions
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state       = State(episode_id=str(uuid4()), step_count=0)
        self._task: Optional[Task] = None
        self._task_idx    = 0
        self._attempts    = 0
        self._best_score  = 0.0

        # T2 data
        self._t2_history:  Dict[str, List[int]] = {}
        self._t2_true_avg: Dict[str, float]     = {}

        # T3 simulation state
        self._t3_day       = 0
        self._t3_stocks:   Dict[str, int]                = {}
        self._t3_pending:  Dict[str, Dict[int, int]]     = {}  # pid → {arrive_day: qty}
        self._t3_demands:  Dict[str, List[int]]          = {}
        self._t3_total_hold  = 0.0
        self._t3_total_order = 0.0
        self._t3_total_stockout = 0
        self._t3_sim_days  = 14

    # ── helpers ─────────────────────────────────────────────────────────────

    def _make_inventory_snapshot(self, stocks: Dict[str, int]) -> List[Dict[str, Any]]:
        snap = []
        for p in PRODUCTS:
            snap.append({
                "id":                   p["id"],
                "name":                 p["name"],
                "current_stock":        stocks[p["id"]],
                "reorder_point":        p["reorder_point"],
                "lead_time_days":       p["lead_time_days"],
                "avg_daily_demand":     p["avg_daily_demand"],
                "unit_cost":            p["unit_cost"],
                "holding_cost_per_day": p["holding_cost_per_day"],
                "order_cost":           p["order_cost"],
            })
        return snap

    def _pending_to_str_keys(self) -> Dict[str, Dict[str, int]]:
        return {
            pid: {str(d): q for d, q in days.items()}
            for pid, days in self._t3_pending.items()
        }

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self, task_id: Optional[str] = None) -> InventoryObservation:
        if task_id and task_id in TASK_MAP:
            self._task = TASK_MAP[task_id]
        else:
            self._task = TASKS[self._task_idx % len(TASKS)]
            self._task_idx += 1

        self._state    = State(episode_id=str(uuid4()), step_count=0)
        self._attempts = 0
        self._best_score = 0.0

        tid = self._task.task_id

        # ── T1 setup ──
        if tid == "T1_identify_low_stock":
            stocks = dict(T1_STOCK)
            history = {p["id"]: _demand_sequence(p, 14) for p in PRODUCTS}
            return InventoryObservation(
                task_id=tid,
                task_description=self._task.description,
                day=0, total_days=1,
                inventory=self._make_inventory_snapshot(stocks),
                demand_history=history,
                pending_orders={},
                score=0.0, attempts=0,
                max_attempts=self._task.max_attempts,
                last_action_feedback="Examine the inventory snapshot and identify low-stock products.",
                done=False, reward=0.0,
            )

        # ── T2 setup ──
        elif tid == "T2_predict_demand":
            # History = 14 days; truth = next 7 days avg
            history, future = {}, {}
            for p in PRODUCTS:
                hist = _demand_sequence(p, 14, seed=42)
                fut  = _demand_sequence(p, 7,  seed=99)
                history[p["id"]] = hist
                future[p["id"]]  = fut
            self._t2_history  = history
            self._t2_true_avg = {pid: round(sum(v)/len(v), 2)
                                  for pid, v in future.items()}
            stocks = {p["id"]: p["reorder_point"] * 3 for p in PRODUCTS}
            return InventoryObservation(
                task_id=tid,
                task_description=self._task.description,
                day=0, total_days=1,
                inventory=self._make_inventory_snapshot(stocks),
                demand_history=history,
                pending_orders={},
                score=0.0, attempts=0,
                max_attempts=self._task.max_attempts,
                last_action_feedback="Analyse demand history and submit your forecast.",
                done=False, reward=0.0,
            )

        # ── T3 setup ──
        else:
            self._t3_day   = 0
            self._t3_total_hold  = 0.0
            self._t3_total_order = 0.0
            self._t3_total_stockout = 0
            # Start with ~5 days of average stock for each product
            self._t3_stocks = {
                p["id"]: p["avg_daily_demand"] * 5
                for p in PRODUCTS
            }
            self._t3_pending = {p["id"]: {} for p in PRODUCTS}
            # Pre-generate all 14+4 days of demand (hidden from agent)
            self._t3_demands = {
                p["id"]: _demand_sequence(p, self._t3_sim_days + 4, seed=7)
                for p in PRODUCTS
            }
            history = {
                pid: demands[:14]
                for pid, demands in self._t3_demands.items()
            }
            return InventoryObservation(
                task_id=tid,
                task_description=self._task.description,
                day=0, total_days=self._t3_sim_days,
                inventory=self._make_inventory_snapshot(self._t3_stocks),
                demand_history=history,
                pending_orders=self._pending_to_str_keys(),
                total_holding_cost=0.0,
                total_order_cost=0.0,
                total_stockout_days=0,
                score=0.0, attempts=0,
                max_attempts=self._t3_sim_days,
                last_action_feedback="Day 0: Simulation starting. Submit your first reorder decisions.",
                done=False, reward=0.0,
            )

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action: InventoryAction) -> InventoryObservation:
        if self._task is None:
            self.reset()

        self._state.step_count += 1
        self._attempts += 1
        tid = self._task.task_id

        # ── T1 step ──
        if tid == "T1_identify_low_stock":
            score, feedback = _grade_t1(action.low_stock_ids, T1_TRUE_LOW)
            improvement = max(0.0, score - self._best_score)
            reward = improvement if improvement > 0 else -0.05
            self._best_score = max(self._best_score, score)
            done = score >= 1.0 or self._attempts >= self._task.max_attempts
            stocks = dict(T1_STOCK)
            return InventoryObservation(
                task_id=tid,
                task_description=self._task.description,
                day=0, total_days=1,
                inventory=self._make_inventory_snapshot(stocks),
                demand_history={p["id"]: _demand_sequence(p, 14) for p in PRODUCTS},
                pending_orders={},
                score=score, attempts=self._attempts,
                max_attempts=self._task.max_attempts,
                last_action_feedback=feedback,
                done=done, reward=round(reward, 4),
            )

        # ── T2 step ──
        elif tid == "T2_predict_demand":
            score, feedback = _grade_t2(action.forecast, self._t2_true_avg)
            improvement = max(0.0, score - self._best_score)
            reward = improvement if improvement > 0 else -0.05
            self._best_score = max(self._best_score, score)
            done = score >= 0.9 or self._attempts >= self._task.max_attempts
            stocks = {p["id"]: p["reorder_point"] * 3 for p in PRODUCTS}
            return InventoryObservation(
                task_id=tid,
                task_description=self._task.description,
                day=0, total_days=1,
                inventory=self._make_inventory_snapshot(stocks),
                demand_history=self._t2_history,
                pending_orders={},
                score=score, attempts=self._attempts,
                max_attempts=self._task.max_attempts,
                last_action_feedback=feedback,
                done=done, reward=round(reward, 4),
            )

        # ── T3 step ──
        else:
            day = self._t3_day

            # 1. Receive pending orders that arrive today
            order_cost_today = 0.0
            for pid in list(self._t3_pending):
                arrived = [qty for arrive_day, qty
                           in list(self._t3_pending[pid].items())
                           if arrive_day <= day]
                for arrive_day in [d for d, _ in self._t3_pending[pid].items()
                                   if d <= day]:
                    del self._t3_pending[pid][arrive_day]
                for qty in arrived:
                    self._t3_stocks[pid] = min(
                        self._t3_stocks[pid] + qty,
                        PRODUCT_MAP[pid]["max_stock"]
                    )

            # 2. Consume today's demand
            day_demands = {pid: self._t3_demands[pid][day] for pid in self._t3_stocks}
            stockouts_today = 0
            for pid, demand in day_demands.items():
                self._t3_stocks[pid] = max(0, self._t3_stocks[pid] - demand)
                if self._t3_stocks[pid] == 0 and demand > 0:
                    stockouts_today += 1
            self._t3_total_stockout += stockouts_today

            # 3. Compute holding cost
            holding_today = sum(
                self._t3_stocks[pid] * PRODUCT_MAP[pid]["holding_cost_per_day"]
                for pid in self._t3_stocks
            )
            self._t3_total_hold += holding_today

            # 4. Process agent's reorder decisions
            order_cost_today = 0.0
            for pid, qty in action.orders.items():
                if pid not in PRODUCT_MAP:
                    continue
                qty = max(0, int(qty))
                if qty == 0:
                    continue
                p = PRODUCT_MAP[pid]
                arrive_day = day + p["lead_time_days"]
                self._t3_pending[pid][arrive_day] = (
                    self._t3_pending[pid].get(arrive_day, 0) + qty
                )
                order_cost_today += p["order_cost"]
            self._t3_total_order += order_cost_today

            # 5. Per-step reward
            step_reward, step_feedback = _grade_t3_step(
                self._t3_stocks, day_demands, holding_today, order_cost_today
            )

            self._t3_day += 1
            done = self._t3_day >= self._t3_sim_days

            # 6. Final episode score
            if done:
                score, final_fb = _grade_t3_final(
                    self._t3_total_hold,
                    self._t3_total_order,
                    self._t3_total_stockout,
                    len(PRODUCTS),
                    self._t3_sim_days,
                )
                feedback = f"Day {day}: {step_feedback} | FINAL: {final_fb}"
            else:
                score    = 0.0
                feedback = f"Day {day}: {step_feedback}"

            self._best_score = max(self._best_score, score)

            # Rolling history: add today's demand to history
            rolling_history = {}
            for pid in self._t3_demands:
                start = max(0, self._t3_day - 14)
                rolling_history[pid] = self._t3_demands[pid][start:self._t3_day]

            return InventoryObservation(
                task_id=tid,
                task_description=self._task.description,
                day=self._t3_day,
                total_days=self._t3_sim_days,
                inventory=self._make_inventory_snapshot(self._t3_stocks),
                demand_history=rolling_history,
                pending_orders=self._pending_to_str_keys(),
                total_holding_cost=round(self._t3_total_hold, 2),
                total_order_cost=round(self._t3_total_order, 2),
                total_stockout_days=self._t3_total_stockout,
                score=score, attempts=self._attempts,
                max_attempts=self._t3_sim_days,
                last_action_feedback=feedback,
                done=done,
                reward=round(step_reward, 4),
            )

    @property
    def state(self) -> State:
        return self._state

    def close(self):
        pass
