import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def isolate_strategy_configs(tmp_path_factory, monkeypatch):
    """Tests must never read or modify the user's real config/strategies/*.json files."""
    import src.strategies.base as base
    monkeypatch.setattr(base, "STRATEGY_CONFIG_DIR", tmp_path_factory.mktemp("strategy_cfg"))
