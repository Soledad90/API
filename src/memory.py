"""
Trade memory system.

Persists trade history to a local JSON file so the agent can learn from
past performance (feedback loop).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


from src.utils import NumpyEncoder



DEFAULT_MEMORY_FILE = os.path.join(
    os.path.dirname(__file__), "..", "memory", "trade_history.json"
)


def load_memory(path: str = DEFAULT_MEMORY_FILE) -> list[dict[str, Any]]:
    """
    Load trade history from disk.

    Parameters
    ----------
    path : str
        Path to the JSON memory file.

    Returns
    -------
    list[dict]
        List of past trade records.
    """
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_memory(
    record: dict[str, Any],
    path: str = DEFAULT_MEMORY_FILE,
) -> None:
    """
    Append a trade record to the memory file.

    Parameters
    ----------
    record : dict
        Trade record to append.
    path : str
        Path to the JSON memory file.
    """
    history = load_memory(path)
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    history.append(record)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, ensure_ascii=False, cls=NumpyEncoder)


def compute_performance_stats(
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute aggregate performance statistics from trade history.

    Parameters
    ----------
    history : list[dict]
        Trade records as stored by ``save_memory``.

    Returns
    -------
    dict
        ``total_trades``, ``wins``, ``losses``, ``win_rate``,
        ``avg_confidence``.
    """
    total = len(history)
    if total == 0:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_confidence": 0.0,
        }

    wins = sum(1 for r in history if r.get("outcome") == "WIN")
    losses = sum(1 for r in history if r.get("outcome") == "LOSS")
    confidences = [r.get("confidence", 0.0) for r in history]

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total, 4) if total else 0.0,
        "avg_confidence": round(sum(confidences) / len(confidences), 4),
    }


def feedback_confidence_adjustment(
    base_confidence: float,
    history: list[dict[str, Any]],
    lookback: int = 10,
) -> float:
    """
    Adjust confidence downward after a run of recent losses.

    Parameters
    ----------
    base_confidence : float
        Raw confidence from the ensemble model.
    history : list[dict]
        Full trade history.
    lookback : int
        Number of recent trades to consider.

    Returns
    -------
    float
        Adjusted confidence clamped to [0, 1].
    """
    recent = history[-lookback:] if len(history) >= lookback else history
    if not recent:
        return base_confidence

    recent_losses = sum(1 for r in recent if r.get("outcome") == "LOSS")
    loss_rate = recent_losses / len(recent)

    # Reduce confidence proportionally to recent loss rate
    adjusted = base_confidence * (1.0 - 0.5 * loss_rate)
    return round(max(0.0, min(1.0, adjusted)), 4)
