import argparse
import json
from pathlib import Path

import pandas as pd


def dir_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a built MOEX research dataset.")
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    validation = pd.read_csv(dataset_dir / "reports" / "validation_report.csv")
    splits = pd.read_csv(dataset_dir / "splits" / "split_summary.csv")
    windows = pd.read_parquet(dataset_dir / "splits" / "window_manifest.parquet")
    curated = pd.read_parquet(dataset_dir / "curated" / "all_symbols_all_intervals.parquet", columns=["symbol", "interval", "timestamp"])

    rows_by_interval = curated.groupby("interval").size().reset_index(name="rows")
    symbols_by_interval = curated.groupby("interval")["symbol"].nunique().reset_index(name="symbols")
    windows_by_split = windows.groupby("split").size().reset_index(name="windows")
    warnings = validation[validation["status"] != "ok"]

    summary = {
        "dataset_id": manifest["dataset_id"],
        "dataset_dir": str(dataset_dir),
        "disk_size_bytes": dir_size(dataset_dir),
        "disk_size_mb": round(dir_size(dataset_dir) / 1024 / 1024, 2),
        "files": sum(1 for item in dataset_dir.rglob("*") if item.is_file()),
        "symbols": len(manifest["symbols"]),
        "curated_rows": manifest["curated_rows"],
        "validation_warnings": int(len(warnings)),
        "split_symbols": int(manifest["split_symbols"]),
        "window_rows": int(manifest["window_rows"]),
        "rows_by_interval": rows_by_interval.to_dict(orient="records"),
        "symbols_by_interval": symbols_by_interval.to_dict(orient="records"),
        "windows_by_split": windows_by_split.to_dict(orient="records"),
    }

    report_dir = dataset_dir / "reports"
    (report_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Dataset Audit",
        "",
        f"Dataset id: `{summary['dataset_id']}`",
        f"Disk size: `{summary['disk_size_mb']} MB`",
        f"Files: `{summary['files']}`",
        f"Symbols: `{summary['symbols']}`",
        f"Curated rows: `{summary['curated_rows']}`",
        f"Validation warnings: `{summary['validation_warnings']}`",
        "",
        "## Rows By Interval",
        "",
        markdown_table(rows_by_interval),
        "",
        "## Symbols By Interval",
        "",
        markdown_table(symbols_by_interval),
        "",
        "## Windows By Split",
        "",
        markdown_table(windows_by_split),
        "",
        "## Split Coverage",
        "",
        markdown_table(splits.head(60)),
    ]
    (report_dir / "AUDIT.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
