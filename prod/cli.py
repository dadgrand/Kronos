"""Kronos production operator CLI."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from decimal import Decimal

from trading.live import LiveExecutionLoop, LiveOrderRequest
from trading.market_data import BinanceMarketDataClient
from trading.ops import JsonlOrderJournal
from trading.paper import PaperBroker, PaperRiskManager, Position

from .binance import BinanceCredentials, BinanceRestClient, BinanceSpotAdapter
from .config import load_config
from .runtime import (
    ProdPreflight,
    ShadowScheduler,
    beat,
    disable_live_state,
    enable_canary_state,
    status_record,
    write_dashboard,
)
from .inference import build_predictor


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config.write_snapshot()

    if args.command == "status":
        record = status_record(config)
        dashboard = write_dashboard(config)
        record["dashboard"] = str(dashboard.resolve())
        print_json(record)
        return 0
    if args.command == "preflight":
        print_json(ProdPreflight(config).run().to_record())
        return 0
    if args.command == "live-preflight":
        adapter = build_binance_adapter(config)
        local_broker = local_broker_from_snapshot(config, adapter.account_snapshot())
        print_json(ProdPreflight(config).run(broker_adapter=adapter, local_broker=local_broker, require_live=True).to_record())
        return 0
    if args.command == "shadow-run":
        client = BinanceMarketDataClient(base_url=args.market_data_base_url, timeout=args.timeout)
        bars, requests = client.fetch_closed_bars(config.symbols, interval=config.interval, limit=args.limit)
        records = ShadowScheduler(config, predictor=build_predictor(config, args.predictor)).write_predictions_for_closed_bars(bars)
        beat(config, status="shadow_ok", payload={"prediction_records": len(records), "request_count": len(requests)})
        print_json({"prediction_records": len(records), "heartbeat": str(config.heartbeat_path.resolve())})
        return 0
    if args.command == "enable-canary":
        path = enable_canary_state(config, operator=args.operator)
        print_json({"enabled": True, "path": str(path.resolve())})
        return 0
    if args.command == "disable-live":
        path = disable_live_state(config, reason=args.reason)
        print_json({"disabled": True, "path": str(path.resolve()), "kill_switch": str(config.kill_switch_path.resolve())})
        return 0
    if args.command == "reconcile":
        adapter = build_binance_adapter(config)
        snapshot = adapter.account_snapshot()
        print_json(
            {
                "timestamp": snapshot.timestamp.isoformat(),
                "cash": snapshot.cash,
                "positions": {symbol: str(quantity) for symbol, quantity in snapshot.positions.items()},
                "open_orders": len(snapshot.orders),
            }
        )
        return 0
    if args.command == "flatten":
        if os.environ.get("KRONOS_CONFIRM_FLATTEN") != "1" or not args.confirm:
            raise SystemExit("Flatten requires --confirm and KRONOS_CONFIRM_FLATTEN=1.")
        adapter = build_binance_adapter(config)
        snapshot = adapter.account_snapshot()
        local_broker = local_broker_from_snapshot(config, snapshot)
        report = ProdPreflight(config).run(broker_adapter=adapter, local_broker=local_broker, require_live=True)
        if not report.ok:
            raise SystemExit(json.dumps(report.to_record(), indent=2, sort_keys=True))
        loop = LiveExecutionLoop(adapter, journal=JsonlOrderJournal(config.journal_path))
        submitted = []
        nonce = int(time.time() * 1000)
        for symbol, quantity in sorted(snapshot.positions.items()):
            if Decimal(str(quantity)) <= 0:
                continue
            request = LiveOrderRequest(
                symbol=symbol,
                side="SELL",
                quantity=Decimal(str(quantity)),
                client_order_id=f"kronos-flatten-{symbol.lower()}-{nonce}",
                order_type="MARKET",
            )
            submitted.append(loop.submit_order(local_broker, request))
        print_json({"flatten_orders_submitted": len(submitted), "orders": [order.__dict__ for order in submitted]})
        return 0
    raise SystemExit(f"Unknown command: {args.command}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to prod JSON config.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    subparsers.add_parser("preflight")
    subparsers.add_parser("live-preflight")
    shadow = subparsers.add_parser("shadow-run")
    shadow.add_argument("--limit", type=int, default=512)
    shadow.add_argument("--timeout", type=float, default=20.0)
    shadow.add_argument("--market-data-base-url", default="https://data-api.binance.vision")
    shadow.add_argument("--predictor", choices=["hold", "kronos"], default="hold")
    canary = subparsers.add_parser("enable-canary")
    canary.add_argument("--operator", required=True)
    disable = subparsers.add_parser("disable-live")
    disable.add_argument("--reason", default="operator disabled live trading")
    subparsers.add_parser("reconcile")
    flatten = subparsers.add_parser("flatten")
    flatten.add_argument("--confirm", action="store_true")
    return parser


def build_binance_adapter(config):
    credentials = BinanceCredentials.from_env()
    client = BinanceRestClient(
        credentials=credentials,
        base_url=config.binance_base_url,
        recv_window_ms=config.recv_window_ms,
    )
    adapter = BinanceSpotAdapter(client, symbols=config.symbols)
    adapter.load_exchange_filters()
    return adapter


def local_broker_from_snapshot(config, snapshot):
    risk = PaperRiskManager(
        initial_equity=max(float(snapshot.cash), 1.0),
        max_drawdown=config.max_drawdown_halt,
        min_cash_fraction=config.min_cash_fraction,
    )
    broker = PaperBroker(initial_cash=max(float(snapshot.cash), 1.0), risk_manager=risk)
    broker.cash = float(snapshot.cash)
    broker.positions = {
        symbol: Position(symbol=symbol, quantity=Decimal(str(quantity)), avg_price=Decimal("0"))
        for symbol, quantity in snapshot.positions.items()
    }
    return broker


def print_json(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
