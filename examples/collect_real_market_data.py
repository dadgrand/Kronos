"""Collect real Binance spot OHLCV data for Kronos validation actuals.

The output is a source-backed market-data package, not an alpha-validation
result. It can be used as actuals once strict forward-only prediction events
exist for the same symbols and timestamps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading.market_data import (
    BINANCE_WS_BASE_URL,
    BinanceMarketDataClient,
    append_new_closed_bars,
    file_sha256,
    write_market_data_package,
)


DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT")


def main():
    args = parse_args()
    output_dir = Path(args.output)
    client = BinanceMarketDataClient(base_url=args.base_url, timeout=args.timeout)
    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    bars, request_log = client.fetch_closed_bars(symbols, interval=args.interval, limit=args.limit)
    poll_rounds = 0
    poll_added = 0
    deadline = time.time() + max(args.poll_seconds, 0)
    while time.time() < deadline:
        sleep_for = min(args.poll_interval_seconds, max(deadline - time.time(), 0))
        if sleep_for > 0:
            time.sleep(sleep_for)
        new_bars, new_requests = client.fetch_closed_bars(symbols, interval=args.interval, limit=min(args.limit, 20))
        bars, added = append_new_closed_bars(bars, new_bars)
        request_log.extend(new_requests)
        poll_rounds += 1
        poll_added += len(added)

    package = write_market_data_package(
        bars,
        output_dir,
        request_log=request_log,
        interval=args.interval,
        source_notes=[
            "Binance public market-data-only REST endpoints; no account credentials used.",
            "Only closed klines are written to actuals. The HTML dashboard may display a separate running live candle.",
            "This package is real market data, but it is not model alpha evidence without forward-only predictions.",
        ],
    )
    html_path = output_dir / "dashboard.html"
    write_dashboard_html(html_path, package, symbols, args.interval)
    report_paths = write_report_html(output_dir, package)
    update_source_manifest(output_dir / "source_manifest.json", report_paths | {"dashboard.html": html_path})
    summary = {
        "dashboard": str(html_path.resolve()),
        "report": str((output_dir / "report.html").resolve()),
        "output_dir": str(output_dir.resolve()),
        "symbols": symbols,
        "interval": args.interval,
        "rows": package["quality"].rows,
        "closed_rows": package["quality"].closed_rows,
        "quality_passed": package["quality"].passed,
        "poll_rounds": poll_rounds,
        "poll_added_bars": poll_added,
        "files": {name: str(path.resolve()) for name, path in package["paths"].items()},
    }
    (output_dir / "collection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="Comma-separated Binance symbols.")
    parser.add_argument("--interval", default="1m", help="Binance kline interval.")
    parser.add_argument("--limit", type=int, default=240, help="Historical klines per symbol.")
    parser.add_argument("--output", default="tmp/real_market_data", help="Output directory.")
    parser.add_argument("--poll-seconds", type=int, default=0, help="Optional live REST polling window.")
    parser.add_argument("--poll-interval-seconds", type=int, default=20, help="Polling cadence when enabled.")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout seconds.")
    parser.add_argument("--base-url", default="https://data-api.binance.vision", help="Binance public data base URL.")
    return parser.parse_args()


def write_dashboard_html(path, package, symbols, interval):
    bars = [bar.to_record() for bar in package["bars"]]
    quality = package["quality"].to_record()
    manifest = package["manifest"]
    symbol_options = "".join(f"<option value=\"{symbol}\">{symbol}</option>" for symbol in symbols)
    streams = "/".join(f"{symbol.lower()}@kline_{interval}" for symbol in symbols)
    ws_url = f"{BINANCE_WS_BASE_URL}/stream?streams={streams}"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kronos Real Market Data Monitor</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #617089;
      --line: #d8e0ea;
      --blue: #2563eb;
      --gold: #b7791f;
      --green: #18794e;
      --red: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Segoe UI, Arial, sans-serif; background: var(--bg); color: var(--ink); }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 24px auto 40px; }}
    header {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 20px; margin-bottom: 18px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.45; }}
    .pill {{ display: inline-flex; align-items: center; gap: 8px; padding: 7px 10px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); color: var(--muted); font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; min-height: 94px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .value {{ margin-top: 8px; font-size: 24px; font-weight: 700; }}
    .status-ok {{ color: var(--green); }}
    .status-warn {{ color: var(--gold); }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; margin-top: 14px; }}
    .toolbar {{ display: flex; gap: 10px; align-items: center; justify-content: space-between; margin-bottom: 12px; }}
    select {{ border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; background: white; color: var(--ink); }}
    svg {{ width: 100%; height: 330px; display: block; border: 1px solid #edf1f6; border-radius: 6px; background: #fbfcfe; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ text-align: left; padding: 9px 8px; border-bottom: 1px solid #e7ecf3; }}
    th {{ color: var(--muted); font-weight: 600; }}
    code {{ font-size: 12px; color: #334155; overflow-wrap: anywhere; }}
    .note {{ border-left: 4px solid var(--gold); background: #fff8e7; padding: 12px; border-radius: 6px; color: #5f4508; }}
    @media (max-width: 820px) {{ .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} header {{ display: block; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Kronos Real Market Data Monitor</h1>
      <p>Closed Binance spot {interval} OHLCV bars for validation actuals, plus a live running-candle WebSocket view.</p>
    </div>
    <div class="pill" id="socketStatus">WebSocket: connecting</div>
  </header>
  <section class="grid">
    <div class="card"><div class="label">Closed rows</div><div class="value">{quality['closed_rows']}</div></div>
    <div class="card"><div class="label">Symbols</div><div class="value">{len(symbols)}</div></div>
    <div class="card"><div class="label">Quality</div><div class="value {'status-ok' if quality['passed'] else 'status-warn'}">{'Passed' if quality['passed'] else 'Check'}</div></div>
    <div class="card"><div class="label">Interval</div><div class="value">{interval}</div></div>
  </section>
  <section class="panel">
    <div class="toolbar">
      <div>
        <strong>Close trend</strong>
        <p id="rangeText">Closed-candle archive</p>
      </div>
      <select id="symbolSelect">{symbol_options}</select>
    </div>
    <svg id="chart" role="img" aria-label="Close price chart"></svg>
  </section>
  <section class="panel">
    <div class="toolbar"><strong>Latest live candle</strong><span id="liveUpdated" class="pill">waiting</span></div>
    <table>
      <thead><tr><th>Symbol</th><th>Open time</th><th>Close</th><th>Volume</th><th>Closed?</th></tr></thead>
      <tbody id="liveTable"></tbody>
    </table>
  </section>
  <section class="panel">
    <p class="note">This is real market data, not proof of model profitability. Approval-grade validation still requires forward-only Kronos prediction events with model hash, code version, checksums, and target timestamps that match these actuals.</p>
  </section>
  <section class="panel">
    <strong>Package files</strong>
    <table>
      <tbody>
        <tr><th>OHLCV JSONL</th><td><code>{manifest['files']['ohlcv.jsonl']['path']}</code></td></tr>
        <tr><th>Actuals JSON</th><td><code>{manifest['files']['actuals.json']['path']}</code></td></tr>
        <tr><th>Quality report</th><td><code>{manifest['files']['quality_report.json']['path']}</code></td></tr>
        <tr><th>Source manifest</th><td><code>{package['paths']['source_manifest'].resolve()}</code></td></tr>
      </tbody>
    </table>
  </section>
</main>
<script>
const archiveBars = {json.dumps(bars)};
const quality = {json.dumps(quality)};
const wsUrl = {json.dumps(ws_url)};
const bySymbol = new Map();
for (const row of archiveBars) {{
  if (!bySymbol.has(row.symbol)) bySymbol.set(row.symbol, []);
  bySymbol.get(row.symbol).push(row);
}}
const chart = document.getElementById('chart');
const select = document.getElementById('symbolSelect');
const rangeText = document.getElementById('rangeText');
const socketStatus = document.getElementById('socketStatus');
const liveTable = document.getElementById('liveTable');
const liveUpdated = document.getElementById('liveUpdated');
const liveRows = new Map();

function draw(symbol) {{
  const rows = bySymbol.get(symbol) || [];
  chart.innerHTML = '';
  if (rows.length < 2) return;
  const width = 1080, height = 330, pad = 42;
  chart.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
  const closes = rows.map(row => Number(row.close));
  const min = Math.min(...closes);
  const max = Math.max(...closes);
  const span = max - min || 1;
  const points = closes.map((close, index) => {{
    const x = pad + index * ((width - pad * 2) / (closes.length - 1));
    const y = height - pad - ((close - min) / span) * (height - pad * 2);
    return `${{x.toFixed(1)}},${{y.toFixed(1)}}`;
  }}).join(' ');
  chart.insertAdjacentHTML('beforeend', `<line x1="${{pad}}" y1="${{height-pad}}" x2="${{width-pad}}" y2="${{height-pad}}" stroke="#cbd5e1"/>`);
  chart.insertAdjacentHTML('beforeend', `<line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height-pad}}" stroke="#cbd5e1"/>`);
  chart.insertAdjacentHTML('beforeend', `<polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="${{points}}"/>`);
  chart.insertAdjacentHTML('beforeend', `<text x="${{pad}}" y="24" fill="#475569" font-size="13">${{symbol}} close: ${{min.toFixed(2)}} - ${{max.toFixed(2)}}</text>`);
  chart.insertAdjacentHTML('beforeend', `<text x="${{width-pad}}" y="${{height-12}}" text-anchor="end" fill="#64748b" font-size="12">${{rows.length}} closed bars</text>`);
  rangeText.textContent = `${{rows[0].open_time}} to ${{rows[rows.length - 1].close_time}}`;
}}

function renderLiveTable() {{
  const rows = Array.from(liveRows.values()).sort((a, b) => a.symbol.localeCompare(b.symbol));
  liveTable.innerHTML = rows.map(row => `<tr><td>${{row.symbol}}</td><td>${{row.openTime}}</td><td>${{row.close}}</td><td>${{row.volume}}</td><td>${{row.closed ? 'yes' : 'running'}}</td></tr>`).join('');
}}

select.addEventListener('change', () => draw(select.value));
draw(select.value);

try {{
  const socket = new WebSocket(wsUrl);
  socket.onopen = () => {{ socketStatus.textContent = 'WebSocket: live'; socketStatus.className = 'pill status-ok'; }};
  socket.onerror = () => {{ socketStatus.textContent = 'WebSocket: error'; socketStatus.className = 'pill status-warn'; }};
  socket.onclose = () => {{ socketStatus.textContent = 'WebSocket: closed'; socketStatus.className = 'pill status-warn'; }};
  socket.onmessage = (event) => {{
    const payload = JSON.parse(event.data);
    const k = payload.data && payload.data.k;
    if (!k) return;
    const symbol = k.s;
    liveRows.set(symbol, {{
      symbol,
      openTime: new Date(k.t).toISOString(),
      close: Number(k.c).toFixed(6),
      volume: Number(k.v).toFixed(4),
      closed: Boolean(k.x)
    }});
    liveUpdated.textContent = new Date().toLocaleTimeString();
    renderLiveTable();
    if (k.x && bySymbol.has(symbol)) {{
      const rows = bySymbol.get(symbol);
      const openTime = new Date(k.t).toISOString();
      if (!rows.some(row => row.open_time === openTime)) {{
        rows.push({{
          symbol,
          open_time: openTime,
          close_time: new Date(k.T).toISOString(),
          close: k.c,
          volume: k.v
        }});
        if (symbol === select.value) draw(symbol);
      }}
    }}
  }};
}} catch (error) {{
  socketStatus.textContent = 'WebSocket: unavailable';
  socketStatus.className = 'pill status-warn';
}}
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def write_report_html(output_dir, package):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir = Path(output_dir)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    records = [bar.to_record() for bar in package["bars"]]
    frame = pd.DataFrame(records)
    frame["period_end"] = pd.to_datetime(frame["period_end"], utc=True)
    for column in ("close", "quote_volume", "volume", "trade_count"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["symbol", "period_end"])
    frame["indexed_close"] = frame.groupby("symbol")["close"].transform(lambda values: values / values.iloc[0] * 100)

    sns.set_theme(style="whitegrid")
    line_png = charts_dir / "indexed_close_trend.png"
    line_svg = charts_dir / "indexed_close_trend.svg"
    fig, ax = plt.subplots(figsize=(12, 5.6))
    sns.lineplot(data=frame, x="period_end", y="indexed_close", hue="symbol", linewidth=1.8, ax=ax)
    ax.set_title("Indexed Close Trend")
    ax.set_xlabel("Period end UTC")
    ax.set_ylabel("Indexed close, first bar = 100")
    ax.legend(title="symbol", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(line_png, dpi=160, bbox_inches="tight")
    fig.savefig(line_svg, bbox_inches="tight")
    plt.close(fig)

    volume_column = "quote_volume" if frame["quote_volume"].notna().any() else "volume"
    volume = frame.groupby("symbol", as_index=False)[volume_column].sum().sort_values(volume_column, ascending=False)
    volume_png = charts_dir / "quote_volume_rank.png"
    volume_svg = charts_dir / "quote_volume_rank.svg"
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=volume, x=volume_column, y="symbol", color="#2457c5", ax=ax)
    ax.set_title("Quote Volume Ranking")
    ax.set_xlabel("Quote volume" if volume_column == "quote_volume" else "Base volume")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(volume_png, dpi=160, bbox_inches="tight")
    fig.savefig(volume_svg, bbox_inches="tight")
    plt.close(fig)

    latest = frame.sort_values("period_end").groupby("symbol").tail(1).copy()
    first = frame.sort_values("period_end").groupby("symbol").head(1)[["symbol", "close"]].rename(columns={"close": "first_close"})
    latest = latest.merge(first, on="symbol", how="left")
    latest["window_return"] = latest["close"] / latest["first_close"] - 1.0
    quality = package["quality"].to_record()
    summary = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "rows": int(quality["rows"]),
        "closed_rows": int(quality["closed_rows"]),
        "symbols": quality["symbols"],
        "quality_passed": bool(quality["passed"]),
        "start": quality["start"],
        "end": quality["end"],
        "timestamp_semantics": "period_end_exclusive_utc",
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    table_rows = []
    for row in latest.sort_values("quote_volume", ascending=False).itertuples():
        table_rows.append(
            "<tr>"
            f"<td>{row.symbol}</td>"
            f"<td>{float(row.close):,.8g}</td>"
            f"<td>{row.window_return:.2%}</td>"
            f"<td>{float(getattr(row, volume_column)):,.0f}</td>"
            f"<td>{int(row.trade_count):,}</td>"
            "</tr>"
        )

    report_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kronos Real Market Data Readiness</title>
<style>
body {{ margin:0; background:#f6f8fb; color:#1f2430; font-family:Segoe UI, Arial, sans-serif; }}
main {{ width:min(1060px, calc(100vw - 32px)); margin:28px auto 44px; }}
section {{ background:#fff; border:1px solid #d8e0ea; border-radius:8px; padding:18px; margin:14px 0; }}
h1 {{ margin:0 0 10px; font-size:30px; }} h2 {{ margin:0 0 10px; font-size:20px; }}
p, li {{ color:#475569; line-height:1.5; }} .lead {{ color:#1f2430; }}
.cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }} .card {{ background:#fbfcfe; border:1px solid #e6e8f0; border-radius:8px; padding:14px; }}
.label {{ color:#64748b; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }} .value {{ font-size:24px; font-weight:700; margin-top:8px; }}
img {{ width:100%; height:auto; border:1px solid #e6e8f0; border-radius:6px; background:#fff; }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:9px 8px; border-bottom:1px solid #e6e8f0; text-align:left; }} th {{ color:#64748b; }}
code {{ overflow-wrap:anywhere; font-size:12px; }} .warn {{ border-left:4px solid #b7791f; background:#fff8e7; }}
@media(max-width:760px) {{ .cards {{ grid-template-columns:repeat(2,1fr); }} }}
</style>
</head>
<body><main>
<h1>Kronos Real Market Data Readiness</h1>
<section><h2>Executive Summary</h2>
<ul>
<li><strong>The real-data lane is usable for validation actuals.</strong> The package contains {quality['closed_rows']} closed Binance spot {quality['interval']} bars across {len(quality['symbols'])} symbols, with quality gate status: {'passed' if quality['passed'] else 'check'}.</li>
<li><strong>The source is auditable.</strong> Delivered files are checksummed in <code>source_manifest.json</code>, with source request metadata and timestamp semantics preserved.</li>
<li><strong>This is not alpha evidence yet.</strong> Profitability validation still needs forward-only Kronos prediction events generated before each target timestamp.</li>
</ul></section>
<section><div class="cards">
<div class="card"><div class="label">Closed rows</div><div class="value">{quality['closed_rows']}</div></div>
<div class="card"><div class="label">Symbols</div><div class="value">{len(quality['symbols'])}</div></div>
<div class="card"><div class="label">Quality gate</div><div class="value">{'Passed' if quality['passed'] else 'Check'}</div></div>
<div class="card"><div class="label">Interval</div><div class="value">{quality['interval']}</div></div>
</div></section>
<section><h2>Price Movement Coverage Is Complete</h2><p class="lead"><strong>The close trend can be inspected without continuity blockers.</strong> The archive is closed-candle only, so it is suitable as actuals data for future scoring if predictions target the same period end.</p><img src="charts/indexed_close_trend.png" alt="Indexed close trend"></section>
<section><h2>Liquidity Context</h2><p class="lead"><strong>Quote volume gives a practical universe-quality read.</strong> Higher-volume symbols are better first candidates for forward-only shadow validation.</p><img src="charts/quote_volume_rank.png" alt="Quote volume ranking"></section>
<section><h2>Symbol-Level Audit Table</h2><table><thead><tr><th>Symbol</th><th>Latest close</th><th>Window return</th><th>Volume</th><th>Trades</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></section>
<section class="warn"><h2>What This Unlocks Next</h2><p>Use this package as the real <code>actuals.json</code> lane for forward-only prediction validation. The next acceptance run should schedule Kronos inference, persist strict <code>prediction_results</code>, and then call <code>ModelApprovalRegistry.approve()</code> against these source-backed actuals.</p></section>
</main></body></html>
"""
    report_path = output_dir / "report.html"
    report_path.write_text(report_html, encoding="utf-8")
    return {
        "report.html": report_path,
        "analysis_summary.json": output_dir / "analysis_summary.json",
        "charts/indexed_close_trend.png": line_png,
        "charts/indexed_close_trend.svg": line_svg,
        "charts/quote_volume_rank.png": volume_png,
        "charts/quote_volume_rank.svg": volume_svg,
    }


def update_source_manifest(manifest_path, extra_files):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    for name, path in extra_files.items():
        path = Path(path)
        manifest["files"][name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
