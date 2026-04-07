"""
Data models for the Inventory Restocking Decision System.

The agent monitors inventory levels, predicts demand, and decides
when and how much stock to reorder — across 5 product types.
"""

from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class InventoryAction(Action):
    """
    Action submitted by the agent.

    Depending on the task:
      T1 (identify low stock)  — fill `low_stock_ids`
      T2 (predict demand)      — fill `forecast`
      T3 (optimize restock)    — fill `orders` (product_id → qty to order today)
    """
    low_stock_ids: List[str] = Field(
        default_factory=list,
        description="T1: list of product_ids the agent thinks are low on stock"
    )
    forecast: Dict[str, float] = Field(
        default_factory=dict,
        description="T2: {product_id: predicted_daily_demand} for next period"
    )
    orders: Dict[str, int] = Field(
        default_factory=dict,
        description="T3: {product_id: units_to_order_today}"
    )


class InventoryObservation(Observation):
    """
    Observation returned to the agent each step.
    Contains the full inventory snapshot and task-specific context.
    """
    # Task context
    task_id: str = Field(default="", description="Active task identifier")
    task_description: str = Field(default="", description="What the agent must do")
    day: int = Field(default=0, description="Current simulation day (T3 only)")
    total_days: int = Field(default=14, description="Total episode length (T3 only)")

    # Inventory snapshot — list of dicts, one per product
    inventory: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Per-product snapshot: id, name, current_stock, reorder_point, "
            "lead_time_days, avg_daily_demand, unit_cost, holding_cost_per_day, order_cost"
        )
    )

    # Demand history: {product_id: [day-N, ..., day-1]} last 14 days
    demand_history: Dict[str, List[int]] = Field(
        default_factory=dict,
        description="Historical daily demand per product (last 14 days, oldest first)"
    )

    # Pending orders arriving in future: {product_id: {arrive_day: qty}}
    pending_orders: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="T3: in-transit orders per product keyed by arrival day"
    )

    # Running cost tracker (T3)
    total_holding_cost: float = Field(default=0.0, description="Cumulative holding cost so far")
    total_order_cost: float = Field(default=0.0, description="Cumulative ordering cost so far")
    total_stockout_days: int = Field(default=0, description="Total product-days with zero stock")

    # Feedback from last action
    last_action_feedback: str = Field(default="", description="Human-readable feedback on last action")
    score: float = Field(default=0.0, description="Current task score 0.0–1.0")
    attempts: int = Field(default=0, description="Steps taken so far")
    max_attempts: int = Field(default=5, description="Max steps for this task")
