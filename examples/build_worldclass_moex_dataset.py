import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


FEATURE_COLS = ["open", "high", "low", "close", "volume", "amount"]
TIME_COLS = ["minute", "hour", "weekday", "day", "month"]


def parse_csv_list(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def now_utc() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def stable_dataset_id(config: dict) -> str:
    body = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()[:12]


def make_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "KronosResearchDatasetBuilder/1.0"})
    return session


def request_json(session: requests.Session, url: str, params: dict | None = None, retries: int = 4) -> dict:
    last_exc = None
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=40)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Request failed after {retries} attempts: {url} {params}") from last_exc


def block_to_df(payload: dict, block_name: str) -> pd.DataFrame:
    block = payload.get(block_name)
    if not block:
        return pd.DataFrame()
    columns = block.get("columns", [])
    rows = block.get("data", [])
    return pd.DataFrame(rows, columns=columns)


def fetch_auto_universe(session: requests.Session, board: str, top_n: int, output_dir: Path) -> list[str]:
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/{board}/securities.json"
    payload = request_json(session, url, params={"iss.meta": "off"})
    securities = block_to_df(payload, "securities")
    marketdata = block_to_df(payload, "marketdata")
    if securities.empty or marketdata.empty:
        raise RuntimeError("Could not load MOEX securities/marketdata for auto universe.")

    df = securities.merge(marketdata, on=["SECID", "BOARDID"], how="left", suffixes=("", "_mkt"))
    for col in ["VALTODAY_RUR", "VALTODAY", "ISSUECAPITALIZATION", "NUMTRADES"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["liquidity_value"] = df.get("VALTODAY_RUR", df.get("VALTODAY", 0)).fillna(0)
    df["status_ok"] = df["STATUS"].astype(str).eq("A")
    df["stock_ok"] = df["SECTYPE"].astype(str).eq("1")
    df = df[df["status_ok"] & df["stock_ok"]].copy()
    df = df.sort_values(["liquidity_value", "ISSUECAPITALIZATION", "NUMTRADES"], ascending=False)
    df.to_csv(output_dir / "auto_universe_ranked.csv", index=False)
    symbols = df["SECID"].head(top_n).astype(str).str.upper().tolist()
    if not symbols:
        raise RuntimeError("Auto universe is empty after filters.")
    return symbols


def fetch_security_metadata(session: requests.Session, symbol: str) -> dict:
    url = f"https://iss.moex.com/iss/securities/{symbol}.json"
    payload = request_json(session, url, params={"iss.meta": "off"})
    description = block_to_df(payload, "description")
    metadata = {"secid": symbol}
    if not description.empty and {"name", "value"}.issubset(description.columns):
        metadata.update(dict(zip(description["name"], description["value"])))
    return metadata


def fetch_dividends(session: requests.Session, symbol: str) -> pd.DataFrame:
    url = f"https://iss.moex.com/iss/securities/{symbol}/dividends.json"
    payload = request_json(session, url, params={"iss.meta": "off"})
    df = block_to_df(payload, "dividends")
    if df.empty:
        return pd.DataFrame(columns=["secid", "isin", "registryclosedate", "value", "currencyid"])
    df["registryclosedate"] = pd.to_datetime(df["registryclosedate"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def fetch_candles(
    session: requests.Session,
    symbol: str,
    board: str,
    interval: int,
    from_date: str,
    till_date: str,
) -> pd.DataFrame:
    url = (
        "https://iss.moex.com/iss/engines/stock/markets/shares/"
        f"boards/{board}/securities/{symbol}/candles.json"
    )
    rows = []
    columns = None
    start = 0
    while True:
        payload = request_json(
            session,
            url,
            params={
                "interval": interval,
                "from": from_date,
                "till": till_date,
                "start": start,
                "iss.meta": "off",
            },
        )
        block = payload.get("candles", {})
        columns = block.get("columns", columns)
        batch = block.get("data", [])
        if not batch:
            break
        rows.extend(batch)
        start += len(batch)
        if len(batch) < 500:
            break
    if not rows:
        return pd.DataFrame(columns=["open", "close", "high", "low", "value", "volume", "begin", "end"])
    return pd.DataFrame(rows, columns=columns)


def normalize_candles(raw: pd.DataFrame, symbol: str, board: str, interval: int) -> pd.DataFrame:
    if raw.empty:
        return raw
    df = raw.copy()
    df = df.rename(columns={"begin": "timestamp", "value": "amount"})
    df["symbol"] = symbol
    df["board"] = board
    df["interval"] = interval
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["symbol", "board", "interval", "timestamp", "end", "open", "high", "low", "close", "volume", "amount"]]
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"], keep="last")
    return df.reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = out["timestamp"]
    out["minute"] = ts.dt.minute
    out["hour"] = ts.dt.hour
    out["weekday"] = ts.dt.weekday
    out["day"] = ts.dt.day
    out["month"] = ts.dt.month
    out["year"] = ts.dt.year
    return out


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["symbol", "timestamp"]).copy()
    g = out.groupby("symbol", group_keys=False)
    out["prev_close"] = g["close"].shift(1)
    out["return_1"] = out["close"] / out["prev_close"] - 1.0
    out["log_return_1"] = np.log(out["close"] / out["prev_close"])
    out["open_to_close_return"] = out["close"] / out["open"] - 1.0
    out["high_low_range_pct"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["true_range"] = np.maximum.reduce(
        [
            (out["high"] - out["low"]).to_numpy(),
            (out["high"] - out["prev_close"]).abs().fillna(0).to_numpy(),
            (out["low"] - out["prev_close"]).abs().fillna(0).to_numpy(),
        ]
    )
    out["true_range_pct"] = out["true_range"] / out["close"].replace(0, np.nan)
    out["vwap"] = out["amount"] / out["volume"].replace(0, np.nan)
    out["amount_log1p"] = np.log1p(out["amount"].clip(lower=0))
    out["volume_log1p"] = np.log1p(out["volume"].clip(lower=0))
    out["rolling_vol_20"] = g["log_return_1"].transform(lambda s: s.rolling(20, min_periods=10).std())
    out["rolling_amount_20"] = g["amount"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    out["rolling_volume_20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    out["calendar_gap_days"] = g["timestamp"].diff().dt.total_seconds() / 86400.0
    out["gap_flag"] = out["calendar_gap_days"].fillna(1.0) > 4.0
    out["bad_ohlc_flag"] = (
        (out["high"] < out[["open", "close"]].max(axis=1))
        | (out["low"] > out[["open", "close"]].min(axis=1))
        | (out["low"] > out["high"])
    )
    out["zero_volume_flag"] = out["volume"].fillna(0) <= 0
    out["zero_amount_flag"] = out["amount"].fillna(0) <= 0
    return out


def add_dividend_features(candles: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    out = candles.copy()
    out["dividend_value"] = 0.0
    out["dividend_yield_prev_close"] = 0.0
    out["days_since_dividend"] = np.nan
    out["days_to_dividend"] = np.nan
    if dividends.empty:
        return out

    div = dividends.dropna(subset=["registryclosedate"]).copy()
    if div.empty:
        return out
    div["date"] = div["registryclosedate"].dt.normalize()

    for symbol, div_symbol in div.groupby("secid"):
        mask = out["symbol"] == symbol
        if not mask.any():
            continue
        symbol_idx = out.index[mask]
        symbol_dates = out.loc[symbol_idx, "timestamp"].dt.normalize()
        div_by_date = div_symbol.groupby("date")["value"].sum()
        out.loc[symbol_idx, "dividend_value"] = symbol_dates.map(div_by_date).fillna(0.0).to_numpy()
        prev_close = out.loc[symbol_idx, "prev_close"].replace(0, np.nan)
        out.loc[symbol_idx, "dividend_yield_prev_close"] = (
            out.loc[symbol_idx, "dividend_value"].to_numpy() / prev_close.to_numpy()
        )

        div_dates = sorted(div_symbol["date"].dropna().unique())
        if not div_dates:
            continue
        div_dates_np = np.array(div_dates, dtype="datetime64[ns]")
        dates_np = symbol_dates.to_numpy(dtype="datetime64[ns]")
        prev_pos = np.searchsorted(div_dates_np, dates_np, side="right") - 1
        next_pos = np.searchsorted(div_dates_np, dates_np, side="left")

        since = np.full(len(dates_np), np.nan)
        to = np.full(len(dates_np), np.nan)
        valid_prev = prev_pos >= 0
        valid_next = next_pos < len(div_dates_np)
        since[valid_prev] = (dates_np[valid_prev] - div_dates_np[prev_pos[valid_prev]]) / np.timedelta64(1, "D")
        to[valid_next] = (div_dates_np[next_pos[valid_next]] - dates_np[valid_next]) / np.timedelta64(1, "D")
        out.loc[symbol_idx, "days_since_dividend"] = since
        out.loc[symbol_idx, "days_to_dividend"] = to
    return out


def validate_symbol_interval(df: pd.DataFrame, symbol: str, interval: int) -> dict:
    if df.empty:
        return {
            "symbol": symbol,
            "interval": interval,
            "rows": 0,
            "status": "empty",
        }
    duplicate_count = int(df.duplicated(["symbol", "timestamp"]).sum())
    bad_ohlc = int(df["bad_ohlc_flag"].sum()) if "bad_ohlc_flag" in df else 0
    nonpositive_price = int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    zero_volume = int((df["volume"].fillna(0) <= 0).sum())
    zero_amount = int((df["amount"].fillna(0) <= 0).sum())
    missing = int(df[FEATURE_COLS].isna().sum().sum())
    return {
        "symbol": symbol,
        "interval": interval,
        "status": "ok" if bad_ohlc == 0 and nonpositive_price == 0 and duplicate_count == 0 else "warn",
        "rows": int(len(df)),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "duplicate_timestamps": duplicate_count,
        "bad_ohlc_rows": bad_ohlc,
        "nonpositive_price_rows": nonpositive_price,
        "zero_volume_rows": zero_volume,
        "zero_amount_rows": zero_amount,
        "feature_missing_cells": missing,
        "median_amount": float(df["amount"].median()),
        "p10_amount": float(df["amount"].quantile(0.10)),
        "p90_amount": float(df["amount"].quantile(0.90)),
        "mean_abs_return_1_pct": float(df["return_1"].abs().mean() * 100),
        "max_abs_return_1_pct": float(df["return_1"].abs().max() * 100),
        "gap_flags": int(df["gap_flag"].sum()) if "gap_flag" in df else 0,
    }


def make_splits(curated: pd.DataFrame, lookback: int, val_count: int, test_count: int, embargo: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    windows = []
    for symbol, df in curated.groupby("symbol"):
        df = df.sort_values("timestamp").reset_index(drop=True)
        n = len(df)
        if n < lookback + val_count + test_count + embargo + 5:
            continue
        train_end = n - val_count - test_count - embargo
        val_start = n - val_count - test_count
        val_end = n - test_count - embargo
        test_start = n - test_count
        split_specs = [
            ("train", lookback, train_end),
            ("val", val_start, val_end),
            ("test", test_start, n),
        ]
        rows.append(
            {
                "symbol": symbol,
                "rows": n,
                "lookback": lookback,
                "embargo": embargo,
                "train_targets": max(0, train_end - lookback),
                "val_targets": max(0, val_end - val_start),
                "test_targets": max(0, n - test_start),
                "train_start": str(df.loc[lookback, "timestamp"]),
                "train_end": str(df.loc[train_end - 1, "timestamp"]),
                "val_start": str(df.loc[val_start, "timestamp"]),
                "val_end": str(df.loc[val_end - 1, "timestamp"]),
                "test_start": str(df.loc[test_start, "timestamp"]),
                "test_end": str(df.loc[n - 1, "timestamp"]),
            }
        )
        for split, start, end in split_specs:
            for target_idx in range(start, end):
                if target_idx < lookback:
                    continue
                windows.append(
                    {
                        "symbol": symbol,
                        "split": split,
                        "target_idx": target_idx,
                        "context_start_idx": target_idx - lookback,
                        "context_end_idx": target_idx - 1,
                        "target_timestamp": df.loc[target_idx, "timestamp"],
                        "context_start_timestamp": df.loc[target_idx - lookback, "timestamp"],
                        "context_end_timestamp": df.loc[target_idx - 1, "timestamp"],
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(windows)


def write_dataset_card(path: Path, manifest: dict, validation: pd.DataFrame, splits: pd.DataFrame) -> None:
    symbols = ", ".join(manifest["symbols"])
    intervals = ", ".join(map(str, manifest["intervals"]))
    card = f"""# MOEX Kronos Research Dataset

Dataset id: `{manifest["dataset_id"]}`

Generated: `{manifest["generated_at_utc"]}`

Source: MOEX ISS candles and security metadata.

Universe: {symbols}

Intervals: {intervals}

Date range request: `{manifest["from_date"]}` to `{manifest["till_date"]}`

## Layers

- `raw/`: immutable fetched candles, dividends, and metadata snapshots.
- `curated/`: typed Parquet tables with canonical OHLCVA, time features, liquidity features, return features, dividend features, and data-quality flags.
- `splits/`: leakage-aware window manifest for train/val/test.
- `reports/`: validation tables and dataset summary.

## Leakage Policy

Windows use only observations strictly before the target timestamp. Splits are chronological per symbol with an embargo between train/validation/test where configured.

## Known Limits

MOEX ISS candles are delayed/public historical data, not a paid tick-level market data feed. Corporate action handling includes dividend event features but does not yet produce fully back-adjusted historical OHLC series. For production-grade alpha research, add paid tick/order-book data, survivorship-bias controlled universe membership, and independently audited corporate action adjustment factors.

## Validation Summary

Total curated rows: `{manifest["curated_rows"]}`

Symbols with warnings: `{int((validation["status"] != "ok").sum()) if not validation.empty else 0}`

## Split Summary

```
{splits.to_string(index=False) if not splits.empty else "No splits generated"}
```
"""
    path.write_text(card, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an institutional-grade MOEX dataset for Kronos research.")
    parser.add_argument("--symbols", default="YDEX,SBER,GAZP,LKOH,ROSN,NVTK,GMKN,MOEX,TATN,MTSS")
    parser.add_argument("--auto-universe", action="store_true", help="Select liquid active shares from MOEX board data.")
    parser.add_argument("--top-n", type=int, default=50, help="Number of liquid symbols to keep with --auto-universe.")
    parser.add_argument("--intervals", default="24")
    parser.add_argument("--board", default="TQBR")
    parser.add_argument("--from-date", default="2022-01-01")
    parser.add_argument("--till-date", default=dt.date.today().isoformat())
    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--val-count", type=int, default=60)
    parser.add_argument("--test-count", type=int, default=60)
    parser.add_argument("--embargo", type=int, default=5)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    args = parser.parse_args()

    session = make_session()
    symbols = parse_csv_list(args.symbols)
    intervals = [int(value.strip()) for value in args.intervals.split(",") if value.strip()]
    config = {
        "symbols": symbols,
        "auto_universe": args.auto_universe,
        "top_n": args.top_n,
        "intervals": intervals,
        "board": args.board,
        "from_date": args.from_date,
        "till_date": args.till_date,
        "lookback": args.lookback,
        "val_count": args.val_count,
        "test_count": args.test_count,
        "embargo": args.embargo,
    }
    dataset_id = stable_dataset_id(config)
    output_dir = args.dataset_root / f"moex_worldclass_{dataset_id}"
    raw_dir = output_dir / "raw"
    curated_dir = output_dir / "curated"
    reports_dir = output_dir / "reports"
    splits_dir = output_dir / "splits"
    for directory in [raw_dir, curated_dir, reports_dir, splits_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    if args.auto_universe:
        symbols = fetch_auto_universe(session, args.board, args.top_n, reports_dir)
        config["symbols"] = symbols
        (reports_dir / "selected_symbols.txt").write_text("\n".join(symbols), encoding="utf-8")
        print(f"Auto universe selected {len(symbols)} symbols:")
        print(", ".join(symbols))

    all_curated = []
    all_dividends = []
    metadata_rows = []
    validation_rows = []

    for symbol in symbols:
        print(f"Symbol {symbol}")
        metadata = fetch_security_metadata(session, symbol)
        metadata_rows.append(metadata)
        (raw_dir / f"{symbol.lower()}_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        dividends = fetch_dividends(session, symbol)
        if not dividends.empty:
            dividends.to_parquet(raw_dir / f"{symbol.lower()}_dividends.parquet", index=False)
            all_dividends.append(dividends.assign(symbol=symbol))

        for interval in intervals:
            print(f"  interval={interval}")
            raw = fetch_candles(session, symbol, args.board, interval, args.from_date, args.till_date)
            raw.to_parquet(raw_dir / f"{symbol.lower()}_{interval}_candles_raw.parquet", index=False)
            candles = normalize_candles(raw, symbol, args.board, interval)
            if candles.empty:
                validation_rows.append({"symbol": symbol, "interval": interval, "status": "empty", "rows": 0})
                continue

            candles = add_time_features(candles)
            candles = add_market_features(candles)
            candles = add_dividend_features(candles, dividends)
            candles.to_parquet(curated_dir / f"{symbol.lower()}_{interval}_curated.parquet", index=False)
            all_curated.append(candles)
            validation_rows.append(validate_symbol_interval(candles, symbol, interval))

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_parquet(raw_dir / "security_metadata.parquet", index=False)

    validation_df = pd.DataFrame(validation_rows).sort_values(["interval", "symbol"]).reset_index(drop=True)
    validation_df.to_csv(reports_dir / "validation_report.csv", index=False)
    validation_df.to_parquet(reports_dir / "validation_report.parquet", index=False)

    if all_dividends:
        dividend_df = pd.concat(all_dividends, ignore_index=True)
        dividend_df.to_parquet(raw_dir / "dividends_all.parquet", index=False)

    if not all_curated:
        raise RuntimeError("No curated rows produced.")
    curated_all = pd.concat(all_curated, ignore_index=True).sort_values(["interval", "symbol", "timestamp"])
    curated_all.to_parquet(curated_dir / "all_symbols_all_intervals.parquet", index=False)

    daily = curated_all[curated_all["interval"] == 24].copy()
    split_summary, windows = make_splits(daily, args.lookback, args.val_count, args.test_count, args.embargo)
    split_summary.to_csv(splits_dir / "split_summary.csv", index=False)
    split_summary.to_parquet(splits_dir / "split_summary.parquet", index=False)
    windows.to_csv(splits_dir / "window_manifest.csv", index=False)
    windows.to_parquet(splits_dir / "window_manifest.parquet", index=False)

    manifest = {
        **config,
        "dataset_id": dataset_id,
        "generated_at_utc": now_utc(),
        "output_dir": str(output_dir),
        "curated_rows": int(len(curated_all)),
        "validation_rows": int(len(validation_df)),
        "split_symbols": int(len(split_summary)),
        "window_rows": int(len(windows)),
        "schema_features": FEATURE_COLS,
        "schema_time_features": TIME_COLS,
        "source": "MOEX ISS",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_dataset_card(output_dir / "DATASET_CARD.md", manifest, validation_df, split_summary)

    summary = {
        "dataset_id": dataset_id,
        "output_dir": str(output_dir),
        "curated_rows": len(curated_all),
        "symbols": symbols,
        "intervals": intervals,
        "warnings": int((validation_df["status"] != "ok").sum()),
        "windows": len(windows),
    }
    print("\nDataset built:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nValidation:")
    print(validation_df.to_string(index=False))
    print("\nSplits:")
    print(split_summary.to_string(index=False))


if __name__ == "__main__":
    main()
