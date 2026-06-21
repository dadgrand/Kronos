import argparse
import concurrent.futures as futures
import datetime as dt
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_worldclass_moex_dataset import (
    add_market_features,
    add_time_features,
    fetch_auto_universe,
    normalize_candles,
    parse_csv_list,
)


PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def stable_id(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def make_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "KronosIntradayDatasetBuilder/1.0"})
    return session


def month_chunks(from_date: str, till_date: str) -> list[tuple[str, str, int, int]]:
    start = pd.Timestamp(from_date).normalize()
    end = pd.Timestamp(till_date).normalize()
    chunks = []
    current = start.replace(day=1)
    while current <= end:
        chunk_start = max(start, current)
        next_month = current + pd.offsets.MonthBegin(1)
        chunk_end = min(end, next_month - pd.Timedelta(days=1))
        chunks.append((chunk_start.date().isoformat(), chunk_end.date().isoformat(), chunk_start.year, chunk_start.month))
        current = next_month
    return chunks


def request_json(session: requests.Session, url: str, params: dict, retries: int) -> dict:
    last_exc = None
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=45)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_exc = exc
            time.sleep(0.75 * (attempt + 1))
    raise RuntimeError(f"Request failed: {url} {params}") from last_exc


def fetch_candles_chunk(
    symbol: str,
    board: str,
    interval: int,
    from_date: str,
    till_date: str,
    retries: int,
) -> pd.DataFrame:
    session = make_session()
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
            {
                "interval": interval,
                "from": from_date,
                "till": till_date,
                "start": start,
                "iss.meta": "off",
            },
            retries,
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


def add_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_time_features(df)
    out = add_market_features(out)
    out["date"] = out["timestamp"].dt.date.astype(str)
    out["session_minute"] = out["timestamp"].dt.hour * 60 + out["timestamp"].dt.minute
    out["is_opening_hour"] = out["timestamp"].dt.hour.between(6, 10)
    out["is_closing_hour"] = out["timestamp"].dt.hour.between(20, 23)
    return out


def chunk_paths(root: Path, symbol: str, interval: int, year: int, month: int) -> tuple[Path, Path, Path]:
    rel = Path(f"interval={interval}") / f"symbol={symbol}" / f"year={year:04d}" / f"month={month:02d}"
    raw_path = root / "raw" / rel / "candles_raw.parquet"
    curated_path = root / "curated" / rel / "candles_curated.parquet"
    meta_path = root / "status" / rel / "status.json"
    return raw_path, curated_path, meta_path


def process_chunk(task: dict) -> dict:
    root = Path(task["root"])
    symbol = task["symbol"]
    interval = int(task["interval"])
    year = int(task["year"])
    month = int(task["month"])
    raw_path, curated_path, meta_path = chunk_paths(root, symbol, interval, year, month)
    if curated_path.exists() and meta_path.exists() and not task["force"]:
        return json.loads(meta_path.read_text(encoding="utf-8")) | {"skipped": True}

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    curated_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    raw = fetch_candles_chunk(
        symbol=symbol,
        board=task["board"],
        interval=interval,
        from_date=task["from_date"],
        till_date=task["till_date"],
        retries=int(task["retries"]),
    )
    raw.to_parquet(raw_path, index=False)
    candles = normalize_candles(raw, symbol, task["board"], interval)
    if candles.empty:
        curated = candles
        rows = 0
        start_ts = None
        end_ts = None
        status = "empty"
    else:
        curated = add_intraday_features(candles)
        rows = int(len(curated))
        start_ts = str(curated["timestamp"].min())
        end_ts = str(curated["timestamp"].max())
        status = "ok"
    curated.to_parquet(curated_path, index=False)

    meta = {
        "symbol": symbol,
        "interval": interval,
        "year": year,
        "month": month,
        "from_date": task["from_date"],
        "till_date": task["till_date"],
        "rows": rows,
        "status": status,
        "start": start_ts,
        "end": end_ts,
        "raw_path": str(raw_path),
        "curated_path": str(curated_path),
        "skipped": False,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_symbols(args, reports_dir: Path) -> list[str]:
    session = make_session()
    if args.from_dataset:
        selected = Path(args.from_dataset) / "reports" / "selected_symbols.txt"
        if selected.exists():
            return [line.strip().upper() for line in selected.read_text(encoding="utf-8").splitlines() if line.strip()]
        manifest = Path(args.from_dataset) / "manifest.json"
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            return [s.upper() for s in data["symbols"]]
    if args.auto_universe:
        return fetch_auto_universe(session, args.board, args.top_n, reports_dir)
    return parse_csv_list(args.symbols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build resumable partitioned MOEX intraday dataset.")
    parser.add_argument("--from-date", default="2022-09-01")
    parser.add_argument("--till-date", default=dt.date.today().isoformat())
    parser.add_argument("--intervals", default="10,1")
    parser.add_argument("--symbols", default="SBER,GAZP,LKOH,YDEX")
    parser.add_argument("--auto-universe", action="store_true")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--from-dataset", type=Path, default=None)
    parser.add_argument("--board", default="TQBR")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=0, help="Debug limit; 0 means all tasks.")
    args = parser.parse_args()

    intervals = [int(item.strip()) for item in args.intervals.split(",") if item.strip()]
    config = {
        "from_date": args.from_date,
        "till_date": args.till_date,
        "intervals": intervals,
        "board": args.board,
        "auto_universe": args.auto_universe,
        "top_n": args.top_n,
        "from_dataset": str(args.from_dataset) if args.from_dataset else None,
    }
    dataset_id = stable_id(config)
    root = args.dataset_root / f"moex_intraday_{dataset_id}"
    for directory in [root / "raw", root / "curated", root / "reports", root / "status"]:
        directory.mkdir(parents=True, exist_ok=True)

    symbols = load_symbols(args, root / "reports")
    config["symbols"] = symbols
    (root / "reports" / "selected_symbols.txt").write_text("\n".join(symbols), encoding="utf-8")

    chunks = month_chunks(args.from_date, args.till_date)
    tasks = []
    for interval in intervals:
        for symbol in symbols:
            for from_date, till_date, year, month in chunks:
                tasks.append(
                    {
                        "root": str(root),
                        "symbol": symbol,
                        "interval": interval,
                        "board": args.board,
                        "from_date": from_date,
                        "till_date": till_date,
                        "year": year,
                        "month": month,
                        "retries": args.retries,
                        "force": args.force,
                    }
                )
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    log(f"Dataset root: {root}")
    log(f"Symbols: {len(symbols)}")
    log(f"Intervals: {intervals}")
    log(f"Chunks: {len(chunks)} months, tasks: {len(tasks)}, workers: {args.workers}")

    results = []
    failures = []
    started = time.time()
    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(process_chunk, task): task for task in tasks}
        for idx, future in enumerate(futures.as_completed(future_map), start=1):
            task = future_map[future]
            try:
                meta = future.result()
                results.append(meta)
                if idx == 1 or idx % 100 == 0 or idx == len(tasks):
                    elapsed = time.time() - started
                    rows = sum(int(r.get("rows", 0)) for r in results)
                    log(f"[{idx}/{len(tasks)}] rows={rows:,} elapsed={elapsed/60:.1f}m last={meta['symbol']} i{meta['interval']} {meta['year']}-{meta['month']:02d} {meta['status']}")
            except Exception as exc:
                failures.append({**task, "error": repr(exc)})
                log(f"FAILED {task['symbol']} i{task['interval']} {task['year']}-{task['month']:02d}: {exc}")

    result_df = pd.DataFrame(results)
    fail_df = pd.DataFrame(failures)
    result_df.to_csv(root / "reports" / "chunk_manifest.csv", index=False)
    result_df.to_parquet(root / "reports" / "chunk_manifest.parquet", index=False)
    fail_df.to_csv(root / "reports" / "failures.csv", index=False)

    summary_rows = []
    if not result_df.empty:
        summary_rows = (
            result_df.groupby(["interval", "symbol"])
            .agg(rows=("rows", "sum"), chunks=("rows", "size"), nonempty=("rows", lambda s: int((s > 0).sum())))
            .reset_index()
            .sort_values(["interval", "rows"], ascending=[True, False])
        )
        summary_rows.to_csv(root / "reports" / "symbol_interval_summary.csv", index=False)
        summary_rows.to_parquet(root / "reports" / "symbol_interval_summary.parquet", index=False)

    manifest = {
        **config,
        "dataset_id": dataset_id,
        "root": str(root),
        "generated_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "tasks": len(tasks),
        "completed": len(results),
        "failures": len(failures),
        "rows": int(result_df["rows"].sum()) if not result_df.empty else 0,
        "status": "complete" if not failures else "partial",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    card = [
        "# MOEX Intraday Partitioned Dataset",
        "",
        f"Dataset id: `{dataset_id}`",
        f"Range: `{args.from_date}` to `{args.till_date}`",
        f"Intervals: `{intervals}`",
        f"Symbols: `{len(symbols)}`",
        f"Rows: `{manifest['rows']}`",
        f"Failures: `{len(failures)}`",
        "",
        "Partition layout:",
        "",
        "`curated/interval=<interval>/symbol=<SECID>/year=<YYYY>/month=<MM>/candles_curated.parquet`",
    ]
    (root / "DATASET_CARD.md").write_text("\n".join(card), encoding="utf-8")

    log("\nDone:")
    log(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
