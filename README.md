---
title: Inventory Restocking Decision System
emoji: 📦
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Inventory Restocking Decision System

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where AI agents manage inventory for a 5-product warehouse — monitoring stock levels, predicting demand, and optimising reorder decisions.

## Tasks

| Task ID | Difficulty | Objective | Metric |
|---------|-----------|-----------|--------|
| `T1_identify_low_stock` | Easy | Flag products below reorder point | F1 Score |
| `T2_predict_demand` | Medium | Forecast avg daily demand for next 7 days | 1 − MAPE |
| `T3_optimize_restock` | Hard | Run 14-day simulation minimising cost & stockouts | Service Level + Cost Efficiency |

## Products

| ID | Name | Avg Demand | Lead Time | Hold Cost | Order Cost |
|----|------|-----------|-----------|-----------|-----------|
| P001 | Widget A | 10/day | 2 days | $0.50/unit/day | $20 |
| P002 | Widget B | 5/day | 3 days | $1.00/unit/day | $30 |
| P003 | Gadget X | 3/day | 4 days | $2.00/unit/day | $50 |
| P004 | Gadget Y | 8/day | 2 days | $0.80/unit/day | $25 |
| P005 | Component Z | 15/day | 1 day | $0.30/unit/day | $15 |

## Action Space

```json
{
  "low_stock_ids": ["P001", "P003"],      // T1: products you think are low on stock
  "forecast":      {"P001": 10.5, ...},   // T2: predicted avg daily demand per product
  "orders":        {"P001": 50, "P002": 0, ...}  // T3: units to order today
}
```

## Observation Space

```json
{
  "task_id": "T3_optimize_restock",
  "day": 5,
  "total_days": 14,
  "inventory": [{"id":"P001","current_stock":45,"reorder_point":25,...}],
  "demand_history": {"P001": [9,11,10,8,12,...]},
  "pending_orders": {"P001": {"7": 100}},
  "total_holding_cost": 312.5,
  "total_order_cost": 80.0,
  "total_stockout_days": 0,
  "score": 0.0,
  "reward": 0.48,
  "done": false
}
```

## Reward Function

- **T1 / T2:** Immediate score on each attempt. Stagnation penalty −0.05 per non-improving step.
- **T3 (per day):** `0.6 × service_level − 0.2 × norm_holding_cost − 0.2 × norm_order_cost`
- **T3 (episode end):** `0.6 × service_level + 0.4 × cost_efficiency`

## API

```bash
# Reset to Task 1
curl -X POST https://your-space.hf.space/reset \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "T1_identify_low_stock"}'

# Submit answer (use episode_id from reset)
curl -X POST https://your-space.hf.space/step \
  -H 'Content-Type: application/json' \
  -d '{"action": {"low_stock_ids": ["P001","P003","P004"]}, "episode_id": "<from_reset>"}'

# List tasks
curl https://your-space.hf.space/tasks
```

## Run Baseline

```bash
export HF_TOKEN=hf_...
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
export ENV_URL=https://your-space.hf.space
python inference.py
```

## Baseline Scores

| Task | Score |
|------|-------|
| T1 identify_low_stock (easy) | ~1.00 |
| T2 predict_demand (medium) | ~0.75 |
| T3 optimize_restock (hard) | ~0.65 |
| **Average** | **~0.80** |

## Local Development

```bash
uv sync
uv run server    # → http://localhost:7860
```

## OpenEnv Validation

```bash
openenv validate .
# [OK] : Ready for multi-mode deployment
```
