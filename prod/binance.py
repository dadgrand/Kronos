"""Binance Spot adapter for Kronos live-execution contracts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from trading.live import AccountSnapshot, ExternalFill, ExternalOrder, LiveOrderRequest


class BinanceApiError(RuntimeError):
    def __init__(self, message, *, status_code=None, code=None, retry_after=None, payload=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.retry_after = retry_after
        self.payload = payload


class BinanceAmbiguousOrderTimeout(BinanceApiError):
    """Raised when Binance reports that order status is unknown."""


@dataclass(frozen=True)
class BinanceCredentials:
    api_key: str
    secret_key: str

    @classmethod
    def from_env(cls, env=None) -> "BinanceCredentials":
        env = env or os.environ
        api_key = env.get("BINANCE_API_KEY", "").strip()
        secret_key = env.get("BINANCE_SECRET_KEY", "").strip()
        if not api_key or not secret_key:
            raise ValueError("BINANCE_API_KEY and BINANCE_SECRET_KEY are required for live Binance access.")
        return cls(api_key=api_key, secret_key=secret_key)


@dataclass(frozen=True)
class SymbolFilters:
    symbol: str
    base_asset: str
    quote_asset: str
    tick_size: Decimal
    step_size: Decimal
    min_qty: Decimal
    min_notional: Decimal

    def quantize_quantity(self, quantity) -> Decimal:
        value = _decimal(quantity, "quantity")
        if self.step_size <= 0:
            return value
        steps = (value / self.step_size).to_integral_value(rounding=ROUND_DOWN)
        return steps * self.step_size

    def quantize_price(self, price) -> Decimal:
        value = _decimal(price, "price")
        if self.tick_size <= 0:
            return value
        ticks = (value / self.tick_size).to_integral_value(rounding=ROUND_DOWN)
        return ticks * self.tick_size

    def validate_order(self, request: LiveOrderRequest, *, reference_price=None) -> None:
        quantity = self.quantize_quantity(request.quantity)
        if quantity <= 0 or quantity < self.min_qty:
            raise ValueError(f"{self.symbol} quantity is below LOT_SIZE minQty.")
        price = request.limit_price or reference_price
        if price is not None:
            notional = quantity * _decimal(price, "reference_price")
            if notional < self.min_notional:
                raise ValueError(f"{self.symbol} order notional is below minNotional.")
        if request.order_type in {"LIMIT", "STOP_LIMIT"} and request.limit_price is not None:
            if self.quantize_price(request.limit_price) != request.limit_price:
                raise ValueError(f"{self.symbol} limit_price violates PRICE_FILTER tickSize.")


class BinanceRestClient:
    def __init__(
        self,
        *,
        credentials: BinanceCredentials | None = None,
        base_url="https://api.binance.com",
        recv_window_ms=5000,
        timeout=10.0,
    ):
        if recv_window_ms <= 0 or recv_window_ms > 60_000:
            raise ValueError("recv_window_ms must be in (0, 60000].")
        self.credentials = credentials
        self.base_url = str(base_url).rstrip("/")
        self.recv_window_ms = int(recv_window_ms)
        self.timeout = float(timeout)

    def server_time(self) -> int:
        payload = self.public_get("/api/v3/time")
        return int(payload["serverTime"])

    def exchange_info(self, symbols=None) -> dict:
        params = {}
        if symbols:
            params["symbols"] = json.dumps([str(symbol).upper() for symbol in symbols], separators=(",", ":"))
        return self.public_get("/api/v3/exchangeInfo", params=params)

    def account(self) -> dict:
        return self.signed_request("GET", "/api/v3/account")

    def open_orders(self, symbol=None) -> list[dict]:
        params = {"symbol": symbol} if symbol else {}
        return self.signed_request("GET", "/api/v3/openOrders", params=params)

    def new_order(self, **params) -> dict:
        return self.signed_request("POST", "/api/v3/order", params=params, ambiguous_on_timeout=True)

    def cancel_order(self, *, symbol: str, order_id=None, orig_client_order_id=None) -> dict:
        params = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = order_id
        if orig_client_order_id is not None:
            params["origClientOrderId"] = orig_client_order_id
        return self.signed_request("DELETE", "/api/v3/order", params=params)

    def start_user_data_stream(self) -> str:
        payload = self.api_key_request("POST", "/api/v3/userDataStream")
        return str(payload["listenKey"])

    def keepalive_user_data_stream(self, listen_key: str) -> dict:
        return self.api_key_request("PUT", "/api/v3/userDataStream", params={"listenKey": listen_key})

    def close_user_data_stream(self, listen_key: str) -> dict:
        return self.api_key_request("DELETE", "/api/v3/userDataStream", params={"listenKey": listen_key})

    def public_get(self, path: str, params=None):
        return self._request_json("GET", path, params=params or {}, signed=False, api_key=False)

    def api_key_request(self, method: str, path: str, params=None):
        return self._request_json(method, path, params=params or {}, signed=False, api_key=True)

    def signed_request(self, method: str, path: str, params=None, *, ambiguous_on_timeout=False):
        if self.credentials is None:
            raise ValueError("Binance credentials are required for signed requests.")
        params = dict(params or {})
        params.setdefault("recvWindow", self.recv_window_ms)
        params["timestamp"] = int(time.time() * 1000)
        query = urllib.parse.urlencode(params, doseq=True)
        params["signature"] = self._sign(query)
        try:
            return self._request_json(method, path, params=params, signed=True, api_key=True)
        except BinanceApiError as exc:
            if ambiguous_on_timeout and exc.code == -1007:
                raise BinanceAmbiguousOrderTimeout(
                    "Binance order submit timed out with unknown execution status.",
                    status_code=exc.status_code,
                    code=exc.code,
                    payload=exc.payload,
                ) from exc
            raise

    def _sign(self, query: str) -> str:
        if self.credentials is None:
            raise ValueError("Binance credentials are required for signing.")
        return hmac.new(
            self.credentials.secret_key.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _request_json(self, method: str, path: str, *, params=None, signed=False, api_key=False):
        params = {key: value for key, value in (params or {}).items() if value is not None}
        query = urllib.parse.urlencode(params, doseq=True)
        url = f"{self.base_url}{path}"
        data = None
        if method.upper() in {"GET", "DELETE"} and query:
            url = f"{url}?{query}"
        elif query:
            data = query.encode("utf-8")
        headers = {"User-Agent": "KronosProdBinance/1.0"}
        if api_key or signed:
            if self.credentials is None:
                raise ValueError("Binance credentials are required for authenticated requests.")
            headers["X-MBX-APIKEY"] = self.credentials.api_key
        request = urllib.request.Request(url, data=data, method=method.upper(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
                if not body:
                    return {}
                return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read()
            payload = _parse_json_bytes(body)
            code = payload.get("code") if isinstance(payload, dict) else None
            message = payload.get("msg") if isinstance(payload, dict) else body[:256].decode("utf-8", "replace")
            raise BinanceApiError(
                f"Binance API error {exc.code}: {message}",
                status_code=exc.code,
                code=code,
                retry_after=exc.headers.get("Retry-After"),
                payload=payload,
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise BinanceApiError(f"Binance request failed: {type(exc).__name__}: {exc}") from exc


class BinanceSpotAdapter:
    """Adapter satisfying trading.live.BrokerAdapter for Binance Spot."""

    def __init__(self, client: BinanceRestClient, *, symbols, quote_asset="USDT"):
        self.client = client
        self.symbols = tuple(str(symbol).upper() for symbol in symbols)
        self.quote_asset = quote_asset
        self._filters: dict[str, SymbolFilters] = {}
        self._order_symbols: dict[str, str] = {}
        self._fill_buffer: list[ExternalFill] = []

    def load_exchange_filters(self) -> dict[str, SymbolFilters]:
        payload = self.client.exchange_info(self.symbols)
        filters = {}
        for item in payload.get("symbols", []):
            symbol = item["symbol"].upper()
            if symbol in self.symbols:
                filters[symbol] = parse_symbol_filters(item)
        missing = sorted(set(self.symbols) - set(filters))
        if missing:
            raise ValueError(f"Binance exchangeInfo missing symbols: {missing}")
        self._filters = filters
        return filters

    def submit_order(self, request: LiveOrderRequest):
        if request.symbol not in self.symbols:
            raise ValueError(f"symbol {request.symbol!r} is outside configured universe.")
        filters = self._filters or self.load_exchange_filters()
        symbol_filter = filters[request.symbol]
        quantity = symbol_filter.quantize_quantity(request.quantity)
        if quantity != request.quantity:
            raise ValueError("Order quantity must already satisfy Binance LOT_SIZE stepSize.")
        params = {
            "symbol": request.symbol,
            "side": request.side,
            "type": request.order_type,
            "quantity": _format_decimal(quantity),
            "newClientOrderId": request.client_order_id,
            "newOrderRespType": "RESULT",
        }
        if request.order_type in {"LIMIT", "STOP_LIMIT"}:
            price = symbol_filter.quantize_price(request.limit_price)
            if price != request.limit_price:
                raise ValueError("Limit price must already satisfy Binance PRICE_FILTER tickSize.")
            params["price"] = _format_decimal(price)
            params["timeInForce"] = request.time_in_force
        if request.order_type in {"STOP", "STOP_LIMIT"}:
            params["stopPrice"] = _format_decimal(request.stop_price)
        response = self.client.new_order(**params)
        external = external_order_from_binance(response)
        self._order_symbols[external.external_id] = external.symbol
        return external

    def cancel_order(self, external_id: str):
        symbol = self._symbol_for_external_id(str(external_id))
        response = self.client.cancel_order(symbol=symbol, order_id=external_id)
        return external_order_from_binance(response)

    def account_snapshot(self) -> AccountSnapshot:
        account = self.client.account()
        balances = {
            item["asset"]: Decimal(str(item.get("free", "0"))) + Decimal(str(item.get("locked", "0")))
            for item in account.get("balances", [])
        }
        positions = {}
        for symbol in self.symbols:
            base_asset = self._base_asset(symbol)
            quantity = balances.get(base_asset, Decimal("0"))
            if quantity != 0:
                positions[symbol] = quantity
        orders = []
        for symbol in self.symbols:
            for payload in self.client.open_orders(symbol=symbol):
                external = external_order_from_binance(payload)
                self._order_symbols[external.external_id] = external.symbol
                orders.append(external)
        return AccountSnapshot(
            timestamp=pd.Timestamp.now("UTC"),
            cash=float(balances.get(self.quote_asset, Decimal("0"))),
            positions=positions,
            orders=orders,
        )

    def stream_fills(self, since=None) -> list[ExternalFill]:
        fills = list(self._fill_buffer)
        self._fill_buffer.clear()
        if since is not None:
            since_ts = pd.Timestamp(since)
            fills = [fill for fill in fills if fill.timestamp >= since_ts]
        return fills

    def record_user_data_event(self, payload: dict):
        if payload.get("e") != "executionReport":
            return None
        execution_type = str(payload.get("x", "")).upper()
        last_qty = Decimal(str(payload.get("l", "0")))
        if execution_type != "TRADE" or last_qty <= 0:
            return None
        fill = ExternalFill(
            external_fill_id=f"{payload.get('i')}:{payload.get('t')}",
            external_order_id=str(payload.get("i")),
            client_order_id=str(payload.get("c")),
            symbol=str(payload.get("s")),
            side=str(payload.get("S")),
            quantity=last_qty,
            price=Decimal(str(payload.get("L"))),
            commission=Decimal(str(payload.get("n", "0"))),
            timestamp=pd.Timestamp(int(payload.get("T")), unit="ms", tz="UTC"),
        )
        self._fill_buffer.append(fill)
        return fill

    def _symbol_for_external_id(self, external_id: str) -> str:
        if external_id in self._order_symbols:
            return self._order_symbols[external_id]
        for symbol in self.symbols:
            for order in self.client.open_orders(symbol=symbol):
                if str(order.get("orderId")) == str(external_id):
                    self._order_symbols[str(external_id)] = symbol
                    return symbol
        raise ValueError(f"Cannot resolve Binance symbol for external order id {external_id!r}.")

    def _base_asset(self, symbol: str) -> str:
        if self._filters and symbol in self._filters:
            return self._filters[symbol].base_asset
        if symbol.endswith(self.quote_asset):
            return symbol[: -len(self.quote_asset)]
        return symbol


def parse_symbol_filters(item: dict) -> SymbolFilters:
    filters = {entry["filterType"]: entry for entry in item.get("filters", [])}
    price_filter = filters.get("PRICE_FILTER", {})
    lot_size = filters.get("LOT_SIZE", {})
    notional = filters.get("MIN_NOTIONAL") or filters.get("NOTIONAL") or {}
    return SymbolFilters(
        symbol=str(item["symbol"]).upper(),
        base_asset=str(item.get("baseAsset", "")),
        quote_asset=str(item.get("quoteAsset", "")),
        tick_size=Decimal(str(price_filter.get("tickSize", "0"))),
        step_size=Decimal(str(lot_size.get("stepSize", "0"))),
        min_qty=Decimal(str(lot_size.get("minQty", "0"))),
        min_notional=Decimal(str(notional.get("minNotional", "0"))),
    )


def external_order_from_binance(payload: dict) -> ExternalOrder:
    status = str(payload.get("status", "NEW")).upper()
    submitted_at = payload.get("transactTime") or payload.get("time") or payload.get("workingTime") or int(time.time() * 1000)
    quantity = Decimal(str(payload.get("origQty", payload.get("quantity", "0"))))
    filled = Decimal(str(payload.get("executedQty", payload.get("filled_quantity", "0"))))
    return ExternalOrder(
        external_id=str(payload.get("orderId")),
        client_order_id=str(payload.get("clientOrderId")),
        symbol=str(payload.get("symbol")),
        side=str(payload.get("side")),
        quantity=quantity,
        filled_quantity=filled,
        status=_map_status(status),
        submitted_at=pd.Timestamp(int(submitted_at), unit="ms", tz="UTC"),
        avg_fill_price=_avg_fill_price(payload),
        order_type=str(payload.get("type", "MARKET")),
        time_in_force=str(payload.get("timeInForce", "DAY")),
        limit_price=_optional_decimal(payload.get("price")),
        stop_price=_optional_decimal(payload.get("stopPrice")),
        reduce_only=False,
    )


def _map_status(status: str) -> str:
    if status == "NEW":
        return "NEW"
    if status in {"PARTIALLY_FILLED", "FILLED", "CANCELED", "CANCELLED", "REJECTED", "EXPIRED"}:
        return "CANCELLED" if status == "CANCELED" else status
    return status


def _avg_fill_price(payload: dict) -> float:
    executed = Decimal(str(payload.get("executedQty", "0")))
    quote = Decimal(str(payload.get("cummulativeQuoteQty", payload.get("cumulativeQuoteQty", "0"))))
    if executed > 0 and quote > 0:
        return float(quote / executed)
    price = _optional_decimal(payload.get("price"))
    return float(price or Decimal("0"))


def _optional_decimal(value):
    if value in {None, ""}:
        return None
    parsed = Decimal(str(value))
    if parsed == 0:
        return None
    return parsed


def _decimal(value, field) -> Decimal:
    parsed = Decimal(str(value))
    if not parsed.is_finite():
        raise ValueError(f"{field} must be finite.")
    return parsed


def _format_decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _parse_json_bytes(body: bytes):
    if not body:
        return {}
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return {"msg": body[:256].decode("utf-8", "replace")}
