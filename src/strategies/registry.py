"""Strategy registry.  To add a strategy: write a Strategy subclass and add it to REGISTRY."""
from __future__ import annotations

from typing import Optional

from src.strategies.base import Strategy
from src.strategies.ema_crossover import EmaCrossover
from src.strategies.orb import OpeningRangeBreakout
from src.strategies.swing_breakout import SwingBreakout
from src.strategies.swing_sr import SwingSupportResistance

REGISTRY: dict[str, type[Strategy]] = {
    cls.key: cls for cls in (EmaCrossover, OpeningRangeBreakout, SwingBreakout, SwingSupportResistance)
}


def get_strategy(key: str, overrides: Optional[dict] = None) -> Strategy:
    if key not in REGISTRY:
        raise ValueError(f"Unknown strategy '{key}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[key].load(overrides)


def write_default_configs(overwrite: bool = False) -> list:
    """Create config/strategies/<key>.json for every registered strategy."""
    written = []
    for cls in REGISTRY.values():
        path = cls.config_path()
        if overwrite or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            cls().save_config(path)
            written.append(path)
    return written
