import os

import pytest

from prod.config import DEFAULT_SYMBOLS, ProdConfig, load_config


def test_prod_config_defaults_are_shadow_and_fixed_universe(tmp_path):
    config = ProdConfig(run_dir=tmp_path)

    assert config.mode == "shadow"
    assert config.venue == "binance_spot"
    assert config.symbols == DEFAULT_SYMBOLS
    assert config.prediction_ledger_path == tmp_path / "prediction_results.jsonl"


def test_prod_config_rejects_live_without_order_cap(tmp_path):
    with pytest.raises(ValueError, match="max_order_notional"):
        ProdConfig(mode="live", run_dir=tmp_path)


def test_prod_config_loads_env_overrides(tmp_path):
    env = {
        "KRONOS_PROD_MODE": "canary_live",
        "KRONOS_RUN_DIR": str(tmp_path),
        "KRONOS_MAX_ORDER_NOTIONAL": "25.5",
    }

    config = load_config(env=env)

    assert config.mode == "canary_live"
    assert config.run_dir == tmp_path
    assert config.max_order_notional == 25.5


def test_prod_config_rejects_dynamic_universe_for_v1(tmp_path):
    with pytest.raises(ValueError, match="v1 universe"):
        ProdConfig(run_dir=tmp_path, symbols=("BTCUSDT", "ETHUSDT"))
