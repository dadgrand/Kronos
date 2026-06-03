from decimal import Decimal

import pandas as pd
import pytest

from prod.binance import (
    BinanceAmbiguousOrderTimeout,
    BinanceApiError,
    BinanceCredentials,
    BinanceRestClient,
    BinanceSpotAdapter,
    external_order_from_binance,
    parse_symbol_filters,
)
from trading.live import LiveOrderRequest


EXCHANGE_SYMBOL = {
    "symbol": "BTCUSDT",
    "baseAsset": "BTC",
    "quoteAsset": "USDT",
    "filters": [
        {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
        {"filterType": "LOT_SIZE", "minQty": "0.00010000", "stepSize": "0.00010000"},
        {"filterType": "MIN_NOTIONAL", "minNotional": "5.00000000"},
    ],
}


def test_binance_hmac_signature_matches_known_digest():
    client = BinanceRestClient(
        credentials=BinanceCredentials(
            "vmPUZE6mv9SD5VNHk4HlWFsOr6aKE2zvsw0MuIgwCIPy6utIco14y7Ju91duEh8A",
            "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZiP1e3UZiP1e3UZiP1e3UZiP1e3UZiP1e3UZ",
        )
    )

    payload = "symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1&recvWindow=5000&timestamp=1499827319559"
    expected = "c493d952f9100c804c9d18e9decef2b5dbad028c7462fda859df847c8e582420"

    assert client._sign(payload) == expected


def test_symbol_filters_quantize_and_validate_order():
    filters = parse_symbol_filters(EXCHANGE_SYMBOL)
    request = LiveOrderRequest(
        symbol="BTCUSDT",
        side="BUY",
        quantity=Decimal("0.0012"),
        client_order_id="client-1",
        order_type="LIMIT",
        time_in_force="GTC",
        limit_price=Decimal("67000.01"),
    )

    assert filters.quantize_quantity(Decimal("0.00129")) == Decimal("0.00120000")
    filters.validate_order(request)

    bad = LiveOrderRequest(symbol="BTCUSDT", side="BUY", quantity=Decimal("0.00001"), client_order_id="client-2")
    with pytest.raises(ValueError, match="minQty"):
        filters.validate_order(bad, reference_price=Decimal("67000"))


def test_external_order_mapping_preserves_client_id_and_decimal_quantities():
    order = external_order_from_binance(
        {
            "symbol": "BTCUSDT",
            "orderId": 123,
            "clientOrderId": "kronos-1",
            "transactTime": 1_700_000_000_000,
            "price": "0.00000000",
            "origQty": "0.01000000",
            "executedQty": "0.00500000",
            "cummulativeQuoteQty": "335.0",
            "status": "PARTIALLY_FILLED",
            "timeInForce": "GTC",
            "type": "MARKET",
            "side": "BUY",
        }
    )

    assert order.external_id == "123"
    assert order.client_order_id == "kronos-1"
    assert order.filled_quantity == Decimal("0.00500000")
    assert order.avg_fill_price == 67000.0


class FakeBinanceClient:
    def __init__(self):
        self.orders = []

    def exchange_info(self, symbols=None):
        return {"symbols": [EXCHANGE_SYMBOL]}

    def new_order(self, **params):
        self.orders.append(params)
        return {
            "symbol": params["symbol"],
            "orderId": 42,
            "clientOrderId": params["newClientOrderId"],
            "transactTime": 1_700_000_000_000,
            "price": params.get("price", "0"),
            "origQty": params["quantity"],
            "executedQty": "0.00000000",
            "status": "NEW",
            "timeInForce": params.get("timeInForce", "GTC"),
            "type": params["type"],
            "side": params["side"],
        }

    def account(self):
        return {"balances": [{"asset": "USDT", "free": "1000", "locked": "0"}]}

    def open_orders(self, symbol=None):
        return []


def test_binance_spot_adapter_submits_filter_checked_order():
    adapter = BinanceSpotAdapter(FakeBinanceClient(), symbols=["BTCUSDT"])
    request = LiveOrderRequest(symbol="BTCUSDT", side="BUY", quantity=Decimal("0.0012"), client_order_id="cid")

    order = adapter.submit_order(request)

    assert order.client_order_id == "cid"
    assert adapter.client.orders[0]["quantity"] == "0.0012"


def test_binance_spot_adapter_buffers_user_data_stream_fills():
    adapter = BinanceSpotAdapter(FakeBinanceClient(), symbols=["BTCUSDT"])
    fill = adapter.record_user_data_event(
        {
            "e": "executionReport",
            "x": "TRADE",
            "s": "BTCUSDT",
            "S": "BUY",
            "i": 42,
            "c": "cid",
            "t": 99,
            "l": "0.001",
            "L": "67000",
            "n": "0.01",
            "T": 1_700_000_000_000,
        }
    )

    assert fill.external_fill_id == "42:99"
    assert adapter.stream_fills()[0].price == Decimal("67000")
    assert adapter.stream_fills() == []


class TimeoutClient(BinanceRestClient):
    def _request_json(self, method, path, *, params=None, signed=False, api_key=False):
        raise BinanceApiError("timeout", code=-1007, status_code=504)


def test_binance_new_order_timeout_is_ambiguous():
    client = TimeoutClient(credentials=BinanceCredentials("key", "secret"))

    with pytest.raises(BinanceAmbiguousOrderTimeout):
        client.new_order(symbol="BTCUSDT", side="BUY", type="MARKET", quantity="0.001")
